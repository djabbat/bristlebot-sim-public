# Bristlebot Swarm Simulator — τ_sick experiment

Simulation of a 120-bristlebot swarm to test whether component age heterogeneity (τ_sick) causes collective dysfunction.

## Theory
- **τ_create**: calendar age (time since assembly)
- **τ_micro**: component age vector (7 components)
- **τ_sick**: Shannon entropy of component age distribution
- **HI**: Hierarchy Index = P(old robot farther from centroid than new robot)

## Usage
```bash
python3 simulate.py --tau-sick 0.0 --runs 20
python3 analyze.py --data runs/
```

## Reference
Tqemaladze J. "Entropy of Age Distribution as a Causal Driver of System Dysfunction: A Bristlebot Swarm Model of Mosaic Aging" (2026)
