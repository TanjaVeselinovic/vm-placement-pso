# Energy-Aware VM Placement with Particle Swarm Optimization

Two PSO variants for placing virtual machines on heterogeneous physical servers with
minimum total power consumption, benchmarked against classic bin-packing heuristics.
Seminar project for the MSc in Computer Science and Informatics, Faculty of Electrical
Engineering, University of East Sarajevo, 2026.

All numbers, the results table and the figures in the accompanying paper were produced
by this code and can be regenerated with a single command.

## Contents

- [Problem](#problem)
- [Methods](#methods)
- [Quick start](#quick-start)
- [Outputs](#outputs)
- [Results](#results)
- [Project structure](#project-structure)
- [Experimental setup](#experimental-setup)
- [Reproducibility](#reproducibility)
- [References](#references)

## Problem

A set of virtual machines (VMs), each with a CPU (MIPS) and memory (MB) demand, has to be
assigned to a set of heterogeneous physical servers so that the total power consumption
of the data center is minimal while no server exceeds its CPU or memory capacity.
Server power is modelled linearly between idle and full-load consumption; inactive
servers consume nothing. The problem generalises bin packing and is NP-hard, which makes
metaheuristics a natural fit.

## Methods

| Method | Description |
|---|---|
| FF, BF, WF | First, Best and Worst Fit heuristics |
| FFD, BFD | the same heuristics after sorting VMs by decreasing CPU demand |
| **PSO – decoding** | a particle is a vector of random keys (one priority per VM); a decoder places VMs in priority order onto servers sorted by energy efficiency, so every decoded solution is feasible by construction |
| **PSO – penalty** | a particle directly encodes the server index of each VM; capacity violations are penalised in the fitness function |

Both PSO variants share the same canonical update rules with a linearly decreasing
inertia weight and one warm-started particle equivalent to the FFD solution.

## Quick start

### Option 1 Python 3.9+

macOS / Linux:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python run_experiments.py
```

Windows (Command Prompt or PowerShell):

```bat
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python run_experiments.py
```

### Option 2 — Docker

```bash
docker build -t vm-placement-pso .
docker run --rm -v "$PWD/results:/app/results" vm-placement-pso
```

On Windows PowerShell use `${PWD}` instead of `$PWD`.

The full experiment set takes about one minute on a recent laptop and a few minutes on
an older one. Progress is printed to the terminal.

### Demo on a single instance

```bash
.venv/bin/python demo.py
```

Without arguments it reproduces the instance shown in the paper (60 VMs on 30 servers)
and draws the resulting data-center layout: First Fit versus PSO, servers as columns,
idle servers greyed out. Try another instance with, for example,
`python demo.py --vm 30 --pm 15 --seed 3`. Options: `--vm`, `--pm`, `--seed`,
`--umax` (CPU utilisation cap), `--iter` (PSO iterations).

### Labels in Serbian

The figures in the paper carry Serbian labels. To reproduce them exactly, set the
environment variable `VMP_LANG=sr` before running either script, e.g.
`VMP_LANG=sr .venv/bin/python run_experiments.py`. Numbers are identical in both
languages.

## Outputs

Everything is written to `results/`, which already contains the outputs used in the paper.

| File | Content |
|---|---|
| `results.csv` | all runs, one row each |
| `energy_comparison.png` | power per method and scenario |
| `active_servers.png` | number of active servers |
| `convergence.png` | PSO convergence over iterations |
| `demo_placement.png` | example placement, First Fit vs. PSO |

Each CSV row holds the scenario, algorithm, run index, power, number of active servers,
mean utilisation, load spread, feasibility flag and running time.

## Results

Total power in watts; PSO values are the mean of ten independent runs with the standard
deviation in parentheses.

| Method | Small (20 VM / 10 PM) | Medium (80 / 40) | Large (200 / 100) |
|---|---:|---:|---:|
| FF | 755.1 | 3308.3 | 6939.7 |
| BF | 828.4 | 3638.7 | 8844.0 |
| WF | 1034.4 | 3775.0 | 8247.3 |
| FFD | 754.7 | 3308.6 | 7430.1 |
| BFD | 922.7 | 3641.4 | 8955.4 |
| **PSO – decoding** | **672.4** (0.0) | **2336.3** (33.0) | **5580.3** (43.0) |
| PSO – penalty | 744.5 (26.2) | 3308.6 (0.0) | 7430.1 (0.0) |

The decoding variant reduces power by 10.9 %, 29.4 % and 19.6 % relative to the best
heuristic, mainly by consolidating load onto fewer, more energy-efficient servers.
Decomposing the gain: energy-aware server ordering alone accounts for a large part of the
saving on the medium and large scenarios, and the PSO search adds a further 11.7–14.8 %
on top of the decoder. The penalty variant is competitive only on the small scenario;
on larger instances it never improves on its FFD warm start, because the share of
feasible assignments in the search space shrinks exponentially with the number of VMs.

## Project structure

| File | Content |
|---|---|
| `vmp/model.py` | VM, server and instance types; power model, feasibility check, metrics |
| `vmp/scenarios.py` | instance generator: server profiles and VM types |
| `vmp/heuristics.py` | FF, BF, WF, FFD, BFD |
| `vmp/pso.py` | the two PSO variants |
| `vmp/labels.py` | scenario names and plot labels (English, or Serbian with `VMP_LANG=sr`) |
| `run_experiments.py` | scenarios, repetitions, CSV and plots |
| `demo.py` | single-instance demo with a placement figure |
| `docs/implementation.md` | detailed mapping of the code to the model in the paper |

Dependencies: NumPy and Matplotlib only (`requirements.txt`).

## Experimental setup

- **Servers** (after Beloglazov and Buyya): HP ProLiant ML110 G4 (3720 MIPS, 4 GB, 86/117 W),
  ML110 G5 (5320 MIPS, 4 GB, 93.7/135 W) and an additional stronger profile
  (10640 MIPS, 8 GB, 110/190 W) for a more heterogeneous infrastructure.
- **VMs:** four types modelled on Amazon EC2 instances (2500/870, 2000/1740, 1000/1740,
  500/613 MIPS/MB).
- **Scenarios:** 20 VMs / 10 servers, 80 / 40, 200 / 100; types drawn uniformly at random
  with a fixed seed.
- **PSO:** swarm of 30, 200 iterations, inertia 0.9 → 0.4, c1 = c2 = 1.49445, velocity
  clamped to 20 % of the position range; ten runs per instance (seeds 100–109).

## Reproducibility

All randomness goes through NumPy's generator with fixed seeds, so rerunning produces
identical numbers. Verified on macOS (Python 3.9, NumPy 2.0) and in Docker
(Python 3.12, NumPy 2.5): all 75 rows of `results/results.csv` match the paper.
