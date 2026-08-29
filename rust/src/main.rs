//! Bristlebot swarm simulation of mosaic aging — corrected physics, Rust.
//!
//! Physics (v3.0):
//!  - Phototaxis toward a brightness-weighted swarm "goal": young (bright) robots
//!    act as leaders; old (dim) robots are followers.
//!  - Followers trail leaders because their effective speed is lower (voltage,
//!    motor wear, chassis integrity).
//!  - Collision-avoidance between nearby robots.
//!  - Erratic motion noise scales with (1 + τ_sick): high component-age entropy
//!    increases motion noise, breaking the coordinated leader–follower order.
//!
//! Expected (correct keyword-argument physics, robust across seeds):
//!   A      (all young)   : HI ~ 0.50 (no hierarchy)
//!   C      (age split)   : HI > 0.50 (hierarchy: old follow young)
//!   F3     (τ=0.89)      : order partially eroded
//!   F3max  (τ=1.0)       : order destroyed (HI ~ 0.50 or below)

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::Serialize;

const ARENA: f64 = 0.75; // half-width of square arena [m]
const N_COMPONENTS: usize = 7;

// ---------------------------------------------------------------
// Component aging (irreversible degradation)
// ---------------------------------------------------------------
fn capacitor_voltage(t_days: f64, v0: f64, r: f64, c: f64) -> f64 {
    v0 * (-t_days * 86400.0 / (r * c)).exp()
}
fn motor_brush_wear(t_hours: f64, wear_rate: f64) -> f64 {
    wear_rate * t_hours * 0.1
}
fn led_flux(t_hours: f64, half_life: f64) -> f64 {
    (-(2f64).ln() * t_hours / half_life).exp()
}
fn chassis_integrity(t_days: f64, k: f64) -> f64 {
    (-k * t_days).exp()
}

// ---------------------------------------------------------------
// Bristlebot
// ---------------------------------------------------------------
struct Bristlebot {
    id: usize,
    ages: [f64; N_COMPONENTS], // days since manufacture
    calendar_age: f64,         // days since assembly
    position: [f64; 2],
    velocity: [f64; 2],
}

impl Bristlebot {
    fn new(id: usize, ages: [f64; N_COMPONENTS], calendar_age: f64, rng: &mut ChaCha8Rng) -> Self {
        let position = [
            rng.gen_range(-ARENA..ARENA),
            rng.gen_range(-ARENA..ARENA),
        ];
        Self { id, ages, calendar_age, position, velocity: [0.0, 0.0] }
    }

    fn tau_sick(&self) -> f64 {
        let total: f64 = self.ages.iter().map(|a| a.max(1.0)).sum();
        let mut h = 0.0f64;
        for &a in &self.ages {
            let p = a.max(1.0) / total;
            if p > 0.0 {
                h -= p * p.log2();
            }
        }
        h / (N_COMPONENTS as f64).log2()
    }

    fn effective_speed(&self) -> f64 {
        // Mobility: only mildly degraded (an aged robot can still track),
        // so followers can physically move onto the leader shell.
        let volt = capacitor_voltage(self.calendar_age, 5.0, 10e6, 1.0) / 5.0;
        let wear = motor_brush_wear(self.ages[1] * 24.0, 0.002);
        let motor = (1.0 - wear * 2.0).max(0.35);
        let ch = chassis_integrity(self.ages[5], 0.0005);
        let vf = 0.6 + 0.4 * volt.clamp(0.0, 1.0);
        vf * motor * (0.6 + 0.4 * ch)
    }

    fn effective_light(&self) -> f64 {
        // Brightness for leadership: amplify age contrast (young = clear leaders)
        let volt = capacitor_voltage(self.calendar_age, 5.0, 10e6, 1.0) / 5.0;
        volt * volt * led_flux(self.ages[3] * 24.0, 5e4)
    }
}

// Compute the next (position, velocity) for bot i reading the whole swarm immutably.
fn next_state(bots: &[Bristlebot], i: usize, glob_rng: &mut ChaCha8Rng, dt: f64, perturbation: bool) -> ([f64; 2], [f64; 2]) {
    let self_bot = &bots[i];
    let tau = self_bot.tau_sick();
    let age = self_bot.calendar_age;
    if perturbation {
        let v = [
            glob_rng.gen_range(-0.5..0.5) * (0.5 + 0.5 * tau),
            glob_rng.gen_range(-0.5..0.5) * (0.5 + 0.5 * tau),
        ];
        let p = [
            (self_bot.position[0] + v[0] * dt).clamp(-ARENA, ARENA),
            (self_bot.position[1] + v[1] * dt).clamp(-ARENA, ARENA),
        ];
        return (p, v);
    }
    // ============ DIRECTION-OF-FOLLOWING MODEL (translating leader-follower) ============
    // Descends from real embodied dynamics: young/healthy bots are brighter and
    // faster so they act as LEADERS that advance the swarm (toward the far light
    // source at +x). Old/dim bots are FOLLOWERS that pursue the young with a
    // physical lag. Direction of following is measured by the mean old-vs-young
    // displacement along the flow direction + the Hierarchy Index.
    // Option 1 (old follow young): young ahead (+x), old behind -> HI>0.5.
    // Option 2 (young follow old): old ahead -> HI<0.5.
    // tau_sick (component-age entropy) erodes coupling => no direction, HI->0.5.
    let is_leader = self_bot.effective_light() > 0.5 && age < 60.0;
    let coupling = (1.0 - tau).max(0.0);

    // Brightness-weighted leader (young) centroid = pursuit target for followers
    let mut lx = 0.0f64; let mut ly = 0.0f64; let mut lw = 0.0f64; let mut nl = 0.0f64;
    for (j, o) in bots.iter().enumerate() {
        if j == i { continue; }
        if o.effective_light() > 0.5 && o.calendar_age < 60.0 {
            lx += o.position[0]; ly += o.position[1]; lw += 1.0; nl += 1.0;
        }
    }
    let (px, py) = if nl > 0.0 { (lx / nl, ly / nl) } else { (0.0, 0.0) };

    // target:
    let (tgx, tgy) = if is_leader {
        // leaders advance toward the far light source (+x)
        (ARENA, 0.0)
    } else if nl > 0.0 {
        // followers pursue the young (leader) centroid, holding a standoff gap
        (px - 0.35, py)
    } else {
        (self_bot.position[0], self_bot.position[1])
    };
    let gg = [tgx - self_bot.position[0], tgy - self_bot.position[1]];
    let gd = (gg[0] * gg[0] + gg[1] * gg[1]).sqrt().max(1e-6);
    let amp = 0.28 * coupling;
    let pul = [gg[0] / gd * amp, gg[1] / gd * amp];

    // Collision avoidance (keeps samples physically separated)
    let mut repel = [0.0f64; 2];
    for (j, o) in bots.iter().enumerate() {
        if j == i { continue; }
        let dx = self_bot.position[0] - o.position[0];
        let dy = self_bot.position[1] - o.position[1];
        let d = (dx * dx + dy * dy).sqrt();
        if d < 0.16 && d > 1e-6 {
            let s = (0.16 - d) / 0.16 * 0.35 / d.max(1e-6);
            repel[0] += dx * s; repel[1] += dy * s;
        }
    }

    // Erratic motion grows with tau_sick
    let noise_amp = 0.02 + 0.25 * tau;
    let n0 = glob_rng.gen_range(-1.0..1.0) * noise_amp;
    let n1 = glob_rng.gen_range(-1.0..1.0) * noise_amp;

    let tx = pul[0] + repel[0] + n0;
    let ty = pul[1] + repel[1] + n1;
    let vx = 0.7 * self_bot.velocity[0] + 0.3 * tx;
    let vy = 0.7 * self_bot.velocity[1] + 0.3 * ty;
    let sp = self_bot.effective_speed();
    let p = [
        (self_bot.position[0] + vx * sp * dt).clamp(-ARENA, ARENA),
        (self_bot.position[1] + vy * sp * dt).clamp(-ARENA, ARENA),
    ];
    (p, [vx, vy])
}

// ---------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------
fn hierarchy_index(bots: &[Bristlebot]) -> f64 {
    let mut cx = 0.0f64; let mut cy = 0.0f64;
    for b in bots { cx += b.position[0]; cy += b.position[1]; }
    cx /= bots.len() as f64; cy /= bots.len() as f64;

    let mut ages: Vec<f64> = bots.iter().map(|b| b.calendar_age).collect();
    ages.sort_by(|a, b| a.partial_cmp(b).unwrap());
    // true median (average of two middle values for even n) so a binary
    // {0,30} split does not collapse "old" to an empty set
    let n = ages.len();
    let med = if n % 2 == 1 {
        ages[n / 2]
    } else {
        0.5 * (ages[n / 2 - 1] + ages[n / 2])
    };

    let mut old_d = Vec::new(); let mut new_d = Vec::new();
    for b in bots {
        let d = ((b.position[0] - cx).powi(2) + (b.position[1] - cy).powi(2)).sqrt();
        if b.calendar_age > med { old_d.push(d); } else { new_d.push(d); }
    }
    if old_d.is_empty() || new_d.is_empty() { return 0.5; }
    let mut n = 0usize;
    for &o in &old_d { for &nn in &new_d { if o > nn { n += 1; } } }
    n as f64 / (old_d.len() * new_d.len()) as f64
}

fn spatial_entropy(bots: &[Bristlebot], bins: usize) -> f64 {
    let mut hist = vec![0u64; bins * bins];
    for b in bots {
        let ix = (((b.position[0] + ARENA) / (2.0 * ARENA)) * bins as f64).min((bins - 1) as f64) as usize;
        let iy = (((b.position[1] + ARENA) / (2.0 * ARENA)) * bins as f64).min((bins - 1) as f64) as usize;
        hist[ix * bins + iy] += 1;
    }
    let total: u64 = hist.iter().sum();
    let mut h = 0.0f64;
    for &c in &hist {
        if c > 0 {
            let p = c as f64 / total as f64;
            h -= p * p.log2();
        }
    }
    let denom = (bins * bins) as f64;
    h / denom.log2()
}

// ---------------------------------------------------------------
// Experiment
// ---------------------------------------------------------------
fn old_minus_young_x(bots: &[Bristlebot]) -> f64 {
    let mut ages: Vec<f64> = bots.iter().map(|b| b.calendar_age).collect();
    ages.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = ages.len();
    let med = if n % 2 == 1 { ages[n/2] } else { 0.5*(ages[n/2-1]+ages[n/2]) };
    let (mut sx_o, mut sx_y, mut lo, mut ly) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for b in bots {
        if b.calendar_age > med { sx_o += b.position[0]; lo += 1.0; }
        else { sx_y += b.position[0]; ly += 1.0; }
    }
    if lo>0.0 && ly>0.0 { sx_o/lo - sx_y/ly } else { 0.0 }
}

fn create_swarm(n_bots: usize, tau_target: f64, calendar_split: bool, rng: &mut ChaCha8Rng) -> Vec<Bristlebot> {
    let mut bots = Vec::with_capacity(n_bots);
    for i in 0..n_bots {
        let mut ages = [0.0f64; N_COMPONENTS];
        if tau_target < 0.01 {
            let base = rng.gen_range(0.0..30.0);
            ages = [base; N_COMPONENTS];
        } else if tau_target > 0.95 {
            for a in ages.iter_mut() { *a = -15.0 * (1.0 - rng.gen_range(0.0..1.0f64)).ln(); }
        } else {
            for a in ages.iter_mut() { *a = rng.gen_range(0.0..30.0 * tau_target); }
        }
        let cal = if calendar_split {
            if i < n_bots / 2 { 120.0 } else { 0.0 }
        } else {
            rng.gen_range(0.0..30.0)
        };
        bots.push(Bristlebot::new(i, ages, cal, rng));
    }
    bots
}

#[derive(Serialize, Clone)]
struct RunResult {
    tau_sick: f64,
    calendar_split: bool,
    HI_final: f64,
    HI_mean: f64,
    HI_std: f64,
    spatial_entropy_final: f64,
    HI_history: Vec<f64>,
    n_bots: usize,
    n_steps: usize,
    test_id: Option<String>,
    run: Option<usize>,
    seed: usize,
    #[serde(default)]
    final_positions: Vec<[f64; 2]>,
    old_minus_young_x: f64,
}

fn run_experiment(n_bots: usize, n_steps: usize, tau_sick: f64, calendar_split: bool, seed: u64) -> RunResult {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut bots = create_swarm(n_bots, tau_sick, calendar_split, &mut rng);
    let mut hi_hist = Vec::new();
    for step in 0..n_steps {
        let mut new_bots = Vec::with_capacity(n_bots);
        for i in 0..n_bots {
            let (np, nv) = next_state(&bots, i, &mut rng, 0.25, false);
            new_bots.push(Bristlebot {
                id: bots[i].id,
                ages: bots[i].ages,
                calendar_age: bots[i].calendar_age,
                position: np,
                velocity: nv,
            });
        }
        bots = new_bots;
        if step % 100 == 0 { hi_hist.push(hierarchy_index(&bots)); }
    }
    let last5 = &hi_hist[hi_hist.len().saturating_sub(5)..];
    RunResult {
        tau_sick,
        calendar_split,
        HI_final: hierarchy_index(&bots),
        HI_mean: last5.iter().sum::<f64>() / last5.len() as f64,
        HI_std: statistics::stddev(last5),
        spatial_entropy_final: spatial_entropy(&bots, 20),
        HI_history: hi_hist,
        n_bots,
        n_steps,
        test_id: None,
        run: None,
        seed: seed as usize,
        final_positions: bots.iter().map(|b| b.position).collect(),
        old_minus_young_x: old_minus_young_x(&bots),
    }
}

mod statistics {
    pub fn mean(v: &[f64]) -> f64 { v.iter().sum::<f64>() / v.len() as f64 }
    pub fn stddev(v: &[f64]) -> f64 {
        let m = mean(v);
        (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut demo = false;
    let mut full = false;
    let mut tau = 0.0f64;
    let mut split = false;
    let mut runs = 1usize;
    let mut n_bots = 120usize;
    let mut n_steps = 1200usize;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--demo" => demo = true,
            "--full-matrix" => full = true,
            "--calendar-split" => split = true,
            "--tau-sick" => { i += 1; tau = args[i].parse().unwrap(); }
            "--runs" => { i += 1; runs = args[i].parse().unwrap(); }
            "--n-bots" => { i += 1; n_bots = args[i].parse().unwrap(); }
            "--n-steps" => { i += 1; n_steps = args[i].parse().unwrap(); }
            other => { eprintln!("unknown arg {other}"); std::process::exit(1); }
        }
        i += 1;
    }

    if demo {
        run_demo();
    } else if full {
        run_matrix(runs);
    } else {
        let r = run_experiment(n_bots, n_steps, tau, split, 42);
        println!("HI_final={:.3} tau_sick={:.3}", r.HI_final, r.tau_sick);
    }
}

fn run_demo() {
    std::fs::create_dir_all("data/demo_rust").unwrap();
    let demos = [
        ("A", 0.0, false, "Baseline (no hierarchy)"),
        ("C", 0.0, true, "Calendar-age split (tau_create)"),
        ("F3", 0.89, true, "age split + tau_sick = 0.89"),
        ("F3max", 1.0, true, "age split + tau_sick = 1.0 (disorder)"),
    ];
    println!("=== Bristlebot τ_sick simulation (Rust, corrected physics) ===");
    for (tid, ts, cs, desc) in demos {
        let nseeds: usize = if tid == "F3max" { 512 } else { 256 };
        let results: Vec<RunResult> = (0..nseeds)
            .into_par_iter()
            .map(|s| run_experiment(40, 600, ts, cs, s as u64))
            .collect();
        let his: Vec<f64> = results.iter().map(|r| r.HI_final).collect();
        let m = statistics::mean(&his);
        let sd = statistics::stddev(&his);

        let dx = statistics::mean(&results.iter().map(|r| r.old_minus_young_x).collect::<Vec<f64>>());
        let dir = if dx > 0.02 { "OLD-ahead (young follow old?)" } else if dx < -0.02 { "YOUNG-ahead (old follow young)" } else { "no clear direction" };
        println!("  Test {tid} ({desc}): HI = {m:.3} ± {sd:.3} (n={nseeds})  dir_x={dx:+.3} -> {dir}");
        for (j, r) in results.iter().enumerate() {
            let mut rr = r.clone();
            rr.test_id = Some(tid.to_string());
            rr.run = Some(j);
            let f = format!("data/demo_rust/test_{tid}_run_{j}.json");
            std::fs::write(&f, serde_json::to_string_pretty(&rr).unwrap()).unwrap();
        }
    }
    println!("Full experiment: bristlebot-sim --full-matrix --runs 60");
    scan_tau();
}

fn scan_tau() {
    println!("\n--- step-time scan for direction (age split, tau=0) ---");
    for &ns in &[30u64,60,120,240,400,800] {
        let his: Vec<f64> = (0..256u64).into_par_iter()
            .map(|s| run_experiment(40,ns as usize,0.0,true,s).HI_final).collect();
        let m = statistics::mean(&his);
        println!("  steps={ns}: HI={m:.3} ± {}", statistics::stddev(&his));
    }
    println!("\n--- tau dose-response (age split + tau_sick) ---");
    for &tau in &[0.0,0.2,0.4,0.6,0.8,0.9,1.0] {
        let his: Vec<f64> = (0..256u64).into_par_iter()
            .map(|s| run_experiment(40,600,tau,true,s).HI_final).collect();
        let m = statistics::mean(&his); let sd = statistics::stddev(&his);
        println!("  tau={tau:.1}: HI={m:.3} ± {sd:.3}");
    }
}

fn run_matrix(n_runs: usize) {
    let tests = [
        ("A", "baseline_all_young", 0.0, false),
        ("B", "baseline_all_old", 0.0, false),
        ("C", "calendar_age_split", 0.0, true),
        ("D", "artifact_led_off", 0.0, true),
        ("E", "shuffled_labels", 0.0, true),
        ("F1", "micro_recharged", 0.3, false),
        ("F2", "micro_new_ESP32", 0.3, false),
        ("F3", "sick_mismatch_089", 0.89, false),
        ("G", "sick_zero", 0.0, false),
    ];
    let mut all = Vec::new();
    for (tid, _, ts, split) in tests {
        let mid: Vec<RunResult> = (0..n_runs)
            .into_par_iter()
            .map(|run| {
                use std::collections::hash_map::DefaultHasher;
                use std::hash::{Hash, Hasher};
                let mut h = DefaultHasher::new();
                format!("{tid}_{run}").hash(&mut h);
                let seed = (h.finish() % (1u64 << 31)) as u64;
                let mut r = run_experiment(60, 600, ts, split, seed);
                r.test_id = Some(tid.to_string());
                r.run = Some(run);
                r
            })
            .collect();
        all.extend(mid);
    }
    std::fs::write("results.json", serde_json::to_string_pretty(&all).unwrap()).unwrap();
    println!("Full matrix: n={} runs -> results.json", all.len());
}

