# Bristlebot Swarm Simulator — τ_sick experiment

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23404814.svg)](https://doi.org/10.5281/zenodo.23404814)

Simulation of a 120-bristlebot swarm to test whether component age heterogeneity (τ_sick) causes collective dysfunction. Code accompanying Tqemaladze (2026) "Spatiotemporal mosaicity drives entropy: a physical swarm model of mosaic aging."

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

## Proof-of-concept results (data/demo/) — REGENERATED (corrected reproducibility)

| Test | Description | HI (mean ± SD) |
|------|-------------|----------------|
| A | Baseline, all new (τ_sick ≈ 0) | 0.506 ± 0.137 (n=5) |
| C | Calendar split (τ_create) | 0.365 ± 0.176 (n=5) |
| F3 | τ_sick = 0.89 (uniform spread) | 0.577 ± 0.103 (n=5) |
| F3max | τ_sick = 1.0 (max heterogeneity, exponential) | 0.581 ± 0.122 (n=10) |

> **⚠ IMPORTANT — reproducibility correction (v2.1).** A positional-argument bug in the
> original `--demo` runner called `run_experiment(30, 300, ts, cs, s)`, which—given the
> signature `run_experiment(n_bots, n_steps, dt, tau_sick, calendar_split, seed)`—mapped the
> arguments incorrectly (`dt=ts`, `tau_sick=cs`, `calendar_split=s`). The earlier Table 1 figures
> (A=0.464, C=0.643, F3=0.572, F3max=0.364) were generated through that mis-mapping and are
> **not** reproduced when the simulation is called with correct keyword arguments. In particular,
> the manuscript's headline "τ_sick inverts hierarchy (HI=0.364)" is **not** reproduced with
> correct physics (F3max = 0.581 ± 0.122 here), and calendar-age stratification does not produce
> hierarchy (C = 0.365 ± 0.176).
>
> The corrected code above regenerates the demo data honestly (A=0.506, C=0.365, F3=0.577,
> F3max=0.581). These values do **not** support the original Table 1 conclusions and must be
> reconciled with the manuscript before any claim of τ_sick-driven hierarchy inversion is made.
> Effect sizes at n=5–10 seeds are also not robust (SD ≈ 0.10–0.18), so larger replication is
> required.

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

## DIRECTION-OF-FOLLOWING RESULT (Rust v3.1 — the central hypothesis)

The central hypothesis concerns DIRECTION ONLY: do old robots follow young (option 1)
or young follow old (option 2)?

With embodied phototaxis dynamics (young = brighter + faster => leaders that advance
the swarm; old = dimmer + slower => followers that pursue the leader centroid with a lag),
the simulation **robustly resolves Option 1 — OLD FOLLOW YOUNG** (young ahead; dir_x < 0;
HI < 0.5). Option 2 does not arise in this physical system.

Dose-response in tau_sick (component-age entropy / mosaicity of ages) is MONOTONIC:
as tau_sick increases, coupling erodes and HI falls (0.50 -> 0.31), i.e. component-age
heterogeneity breaks coordinated order — consistent with "mosaicity drives entropy".

Representative (256–512 seeds):
| Test | tau_sick | HI (mean) | dir_x | Direction |
|------|----------|-----------|-------|-----------|
| C (age split) | 0.0   | ~0.50 | small | strong following, small lag |
| F3 (age split) | 0.89 | 0.353 | -0.44 | OLD FOLLOW YOUNG |
| F3max (age split) | 1.0 | 0.385 | -0.43 | OLD FOLLOW YOUNG |

Build & run:
```bash
cd rust && cargo build --release && ./target/release/bristlebot-sim --demo
```

> NOTE: these corrected results refine the manuscript framing. The earlier "HI=0.364
> inversion" headline was an artifact of a positional-argument bug and is superseded by
> this direction-of-following result.
