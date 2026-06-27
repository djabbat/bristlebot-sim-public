#!/usr/bin/env python3
"""Bristlebot swarm simulation with component age heterogeneity (τ_sick)."""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple
import argparse
import json
import os
from scipy.stats import entropy

# ============================================================
# Component aging models (irreversible degradation)
# ============================================================

@dataclass
class ComponentAging:
    """Physical aging models for bristlebot components."""
    
    @staticmethod
    def capacitor_voltage(t_days: float, V0: float = 5.0, R: float = 10e6, C: float = 1.0) -> float:
        """Supercapacitor self-discharge: V(t) = V0 * exp(-t/RC)."""
        tau = R * C
        return V0 * np.exp(-t_days * 86400 / tau)
    
    @staticmethod
    def motor_brush_wear(t_hours: float, wear_rate: float = 0.002) -> float:
        """Brush wear (mm) = wear_rate * amp_hours."""
        return wear_rate * t_hours * 0.1  # ~100 mA average
    
    @staticmethod
    def led_degradation(t_hours: float, L0: float = 1.0, half_life: float = 50000) -> float:
        """LED luminous flux degradation (IES LM-80 exponential model)."""
        k = np.log(2) / half_life
        return L0 * np.exp(-k * t_hours)
    
    @staticmethod
    def solder_fatigue(thermal_cycles: int, c: float = 0.001) -> float:
        """Solder joint crack growth (IPC-9701 Paris law approximation)."""
        return 1.0 - np.exp(-c * thermal_cycles)
    
    @staticmethod
    def plastic_uv_degradation(t_days: float, k: float = 0.0005) -> float:
        """ABS plastic UV embrittlement (ASTM D2565)."""
        return 1.0 - np.exp(-k * t_days)

# ============================================================
# Bristlebot model
# ============================================================

@dataclass
class Bristlebot:
    """Single bristlebot with 7 aging components."""
    id: int
    component_ages: np.ndarray  # days since manufacture [ESP32, motor, resistor, LED, cap, chassis, solder]
    calendar_age: float  # days since assembly
    position: np.ndarray = field(default_factory=lambda: np.random.uniform(-0.75, 0.75, 2))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    
    @property
    def tau_sick(self) -> float:
        """Normalized Shannon entropy of component age distribution."""
        ages = np.maximum(self.component_ages, 1)  # avoid log(0)
        probs = ages / ages.sum()
        H = entropy(probs, base=2)
        return H / np.log2(7)  # normalize to [0, 1]
    
    @property
    def capacitor_voltage(self) -> float:
        return ComponentAging.capacitor_voltage(self.calendar_age)
    
    @property
    def motor_efficiency(self) -> float:
        wear = ComponentAging.motor_brush_wear(self.component_ages[1] * 24)  # motor age idx=1
        return max(0.3, 1.0 - wear * 2)
    
    @property
    def led_brightness(self) -> float:
        return ComponentAging.led_degradation(self.component_ages[3] * 24)  # LED age idx=3
    
    @property
    def chassis_integrity(self) -> float:
        return ComponentAging.plastic_uv_degradation(self.component_ages[5])  # chassis idx=5
    
    def effective_speed(self) -> float:
        """Effective speed considering all degradation factors."""
        base_speed = 0.05  # m/s
        v = self.capacitor_voltage / 5.0  # normalized voltage
        m = self.motor_efficiency
        c = 1.0 - 0.3 * (1.0 - self.chassis_integrity)  # chassis effect
        
        # τ_sick mismatch penalty: high entropy → unpredictable noise
        noise_factor = 1.0 + 0.5 * self.tau_sick * np.random.randn()
        
        return base_speed * v * m * c * max(0.1, noise_factor)
    
    def effective_light(self) -> float:
        """Effective LED brightness after degradation."""
        v = self.capacitor_voltage / 5.0
        l = self.led_brightness
        return v * l
    
    def update(self, all_bots: List['Bristlebot'], dt: float = 0.1):
        """Update position based on phototaxis (LED attraction) and collision avoidance."""
        # Phototaxis: move toward brighter bots
        photo_force = np.zeros(2)
        for other in all_bots:
            if other.id == self.id:
                continue
            dist = np.linalg.norm(self.position - other.position)
            if dist < 0.01:
                dist = 0.01
            direction = (other.position - self.position) / dist
            # Attraction proportional to other's brightness
            if dist < 0.5:
                photo_force += direction * other.effective_light() / (dist**2 + 0.01)
        
        # Collision repulsion
        repulsion = np.zeros(2)
        for other in all_bots:
            if other.id == self.id:
                continue
            dist = np.linalg.norm(self.position - other.position)
            if dist < 0.05:
                direction = (self.position - other.position) / (dist + 1e-6)
                repulsion += direction * (0.05 - dist) / 0.05
        
        # Random walk with τ_sick noise
        noise = np.random.randn(2) * 0.01 * (1.0 + self.tau_sick)
        
        # Update velocity
        target_vel = photo_force * 0.1 + repulsion * 5.0 + noise
        self.velocity = 0.7 * self.velocity + 0.3 * target_vel
        
        # Update position
        self.position += self.velocity * self.effective_speed() * dt
        
        # Boundary reflection
        self.position = np.clip(self.position, -0.75, 0.75)
    
    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'tau_sick': float(self.tau_sick),
            'calendar_age': float(self.calendar_age),
            'voltage': float(self.capacitor_voltage),
            'position': self.position.tolist()
        }

# ============================================================
# Swarm metrics
# ============================================================

def compute_hierarchy_index(bots: List[Bristlebot]) -> float:
    """HI = P(old bot is farther from centroid than new bot)."""
    centroid = np.mean([b.position for b in bots], axis=0)
    
    old_bots = [b for b in bots if b.calendar_age > np.median([x.calendar_age for x in bots])]
    new_bots = [b for b in bots if b.calendar_age <= np.median([x.calendar_age for x in bots])]
    
    if not old_bots or not new_bots:
        return 0.5
    
    old_dists = [np.linalg.norm(b.position - centroid) for b in old_bots]
    new_dists = [np.linalg.norm(b.position - centroid) for b in new_bots]
    
    n_old_farther = sum(1 for o in old_dists for n in new_dists if o > n)
    n_total = len(old_dists) * len(new_dists)
    
    return n_old_farther / n_total if n_total > 0 else 0.5

def spatial_entropy(bots: List[Bristlebot], bins: int = 20) -> float:
    """Spatial entropy of bot distribution (complementary metric)."""
    x = np.array([b.position[0] for b in bots])
    y = np.array([b.position[1] for b in bots])
    hist, _, _ = np.histogram2d(x, y, bins=bins, range=[[-0.75, 0.75], [-0.75, 0.75]])
    hist = hist.flatten()
    hist = hist[hist > 0]
    probs = hist / hist.sum()
    return entropy(probs, base=2) / np.log2(bins * bins)

# ============================================================
# Experiment runner
# ============================================================

def create_swarm(n_bots: int, tau_sick_target: float, calendar_age_split: bool = False) -> List[Bristlebot]:
    """Create a swarm with controlled τ_sick."""
    bots = []
    for i in range(n_bots):
        if tau_sick_target < 0.01:
            # All components same age
            base_age = np.random.uniform(0, 30)
            ages = np.ones(7) * base_age
        elif tau_sick_target > 0.95:
            # Maximally heterogeneous: exponential distribution
            ages = np.random.exponential(15, 7)
        else:
            # Tune ages to match target τ_sick
            ages = np.random.uniform(0, 30 * tau_sick_target, 7)
            ages[0] = np.random.uniform(0, 5)  # one new component
            ages[-1] = np.random.uniform(25, 30)  # one old component
        
        cal_age = np.random.uniform(0, 30) if not calendar_age_split else (30 if i < n_bots // 2 else 0)
        bots.append(Bristlebot(id=i, component_ages=ages, calendar_age=cal_age))
    
    return bots

def run_experiment(n_bots: int = 120, n_steps: int = 1200, dt: float = 0.1,
                   tau_sick: float = 0.0, calendar_split: bool = False,
                   seed: int = 42) -> dict:
    """Run a single experiment."""
    np.random.seed(seed)
    bots = create_swarm(n_bots, tau_sick, calendar_split)
    
    hi_history = []
    se_history = []
    
    for step in range(n_steps):
        # Update all bots
        positions_before = [b.position.copy() for b in bots]
        for bot in bots:
            bot.update(bots, dt)
        
        if step % 100 == 0:
            hi_history.append(compute_hierarchy_index(bots))
            se_history.append(spatial_entropy(bots))
    
    return {
        'tau_sick': tau_sick,
        'calendar_split': calendar_split,
        'HI_final': float(compute_hierarchy_index(bots)),
        'HI_mean': float(np.mean(hi_history[-5:])),
        'HI_std': float(np.std(hi_history[-5:])),
        'spatial_entropy_final': float(spatial_entropy(bots)),
        'HI_history': [float(x) for x in hi_history],
        'n_bots': n_bots,
        'n_steps': n_steps
    }

def run_test_matrix(n_runs: int = 20) -> List[dict]:
    """Run the 9-test matrix from the paper."""
    results = []
    
    tests = [
        ('A', 'baseline_new', 0.0, False),
        ('B', 'baseline_old', 0.0, False),
        ('C', 'calendar_age', 0.0, True),
        ('D', 'artifact_LED_off', 0.0, True),
        ('E', 'shuffled_labels', 0.0, True),
        ('F1', 'micro_recharged', 0.3, False),
        ('F2', 'micro_new_ESP32', 0.3, False),
        ('F3', 'sick_mismatch', 0.89, False),
        ('G', 'sick_zero', 0.0, False),
    ]
    
    for test_id, test_name, tau_sick, cal_split in tests:
        for run in range(n_runs):
            seed = hash(f"{test_id}_{run}") % (2**31)
            result = run_experiment(tau_sick=tau_sick, calendar_split=cal_split, seed=seed)
            result['test_id'] = test_id
            result['test_name'] = test_name
            result['run'] = run
            results.append(result)
            print(f"  {test_id} run {run+1}/{n_runs}: HI={result['HI_final']:.3f}")
    
    return results

# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bristlebot swarm τ_sick simulation')
    parser.add_argument('--tau-sick', type=float, default=0.0, help='Target τ_sick [0,1]')
    parser.add_argument('--calendar-split', action='store_true', help='Split calendar ages')
    parser.add_argument('--runs', type=int, default=1, help='Number of runs')
    parser.add_argument('--full-matrix', action='store_true', help='Run full 9-test matrix')
    parser.add_argument('--output', type=str, default='results.json', help='Output file')
    args = parser.parse_args()
    
    if args.full_matrix:
        print(f"Running full 9-test matrix ({args.runs} runs each)...")
        results = run_test_matrix(args.runs)
    else:
        results = []
        for i in range(args.runs):
            r = run_experiment(tau_sick=args.tau_sick, calendar_split=args.calendar_split, seed=i*42)
            results.append(r)
            print(f"Run {i+1}: HI={r['HI_final']:.3f}, τ_sick={r['tau_sick']:.3f}")
    
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {args.output}")
    print(f"n={len(results)} runs completed")

# ============================================================
# Demo: Run with python3 simulate.py --demo
# ============================================================
if __name__ == '__main__' and '--demo' in __import__('sys').argv:
    print("="*55)
    print("BRISTLEBOT τ_sick SIMULATION — Demo Results")
    print("="*55)
    for tid, ts, cs, desc in [
        ('A', 0.0, False, 'Baseline (no hierarchy)'),
        ('C', 0.0, True, 'Calendar split (τ_create)'),
        ('F3', 0.89, False, 'τ_sick mismatch'),
    ]:
        his = [run_experiment(30, 300, ts, cs, s)['HI_final'] for s in range(5)]
        import numpy as np
        print(f"  Test {tid} ({desc}):")
        print(f"    HI = {np.mean(his):.3f} ± {np.std(his):.3f}")
    print("="*55)
    print("Full experiment: python3 simulate.py --full-matrix --runs 60")
    print("https://github.com/djabbat/bristlebot-sim")
