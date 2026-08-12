# Bristlebot Swarm Simulator — τ_sick experiment

Simulation of a 120-bristlebot swarm to test whether component age heterogeneity (τ_sick) causes collective dysfunction. Code accompanying Tqemaladze (2026) "Entropy of Age Distribution as a Causal Driver of System Dysfunction: A Bristlebot Swarm Model of Mosaic Aging" (Scientific Reports).

## Theory

- **τ_create** — calendar age: time since assembly (conventional model of aging)
- **τ_micro** — component age: vector of manufacturing dates for 7 critical components (ESP32, motor, resistor, LED, supercapacitor, chassis, solder)
- **τ_sick** — assembly mismatch: normalized Shannon entropy of the component age distribution, H(τ) = −Σ pᵢ log₂ pᵢ
- **HI** — Hierarchy Index: P(old robot is farther from the swarm centroid than new robot). HI = 0.50 = random; HI > 0.50 = hierarchy

## Repository contents

| File | Description |
|------|-------------|
| `simulate.py` | Agent-based simulation (7 aging components, phototaxis + collision avoidance) |
| `analyze.py` | Statistical analysis: Cohen's d, Bonferroni correction, HI comparison |
| `data/demo/` | Proof-of-concept demo results — 3 tests × 5 runs (30 bots, 300 steps, seeds 0–4) |
| `LICENSE` | GNU GPL v3 |
| `requirements.txt` | numpy, scipy |

## Reproducibility

```bash
pip install -r requirements.txt

# Proof-of-concept demo (reproduces data/demo/, Table 1 in the manuscript)
python3 simulate.py --demo

# Single experiment
python3 simulate.py --tau-sick 0.89 --runs 20

# Full 9-test matrix (A, B, C, D, E, F1, F2, F3, G)
python3 simulate.py --full-matrix --runs 60
```

Raw results are in `data/demo/` — one JSON per run, including `HI_final`, `HI_mean`, `HI_std`, `spatial_entropy_final`, and the full `HI_history` time series. Analysis of these files:

```bash
python3 analyze.py --data data/demo/
```

## Proof-of-concept results (data/demo/)

| Test | Description | HI (mean ± SD, n=5) |
|------|-------------|----------------------|
| A | Baseline, all new (τ_sick ≈ 0) | 0.464 ± 0.006 |
| C | Calendar split (τ_create) | 0.643 ± 0.024 |
| F3 | τ_sick mismatch (0.89) | 0.572 ± 0.004 |

## Experimental design (manuscript summary)

The 9-test matrix isolates successive age layers:

| Phase | Tests | Layer isolated |
|-------|-------|----------------|
| 1. Baselines | A (new), B (old), G (τ_sick=0) | Null model |
| 2. Calendar age | C (split) | τ_create |
| 3. Component age | F1 (recharged), F2 (new ESP32) | τ_micro |
| 4. Assembly mismatch | F3 (τ_sick=0.89) ★ | τ_sick — the key causal test |

The τ_sick test (F3) asks the causal question: "Does age mismatch cause dysfunction?" — testing whether age heterogeneity alone drives collective system failure, independent of average age.

## Reference

Tqemaladze J. "Entropy of Age Distribution as a Causal Driver of System Dysfunction: A Bristlebot Swarm Model of Mosaic Aging" (2026), Scientific Reports.

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).
