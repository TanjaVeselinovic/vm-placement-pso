# Implementation notes

How each part of the code maps to the model and algorithms described in the
accompanying paper (seminar project, MSc in Computer Science and Informatics,
University of East Sarajevo, 2026). Equation numbers refer to the paper.

## 1. Problem model (`vmp/model.py`) — chapter 4

- An instance consists of a set V of n virtual machines (CPU demand in MIPS, RAM in MB)
  and a set P of m physical servers (CPU and RAM capacity, idle power P_idle, full-load
  power P_max). This is the notation of section 4.1 (table 4.1).
- A placement is stored as an integer vector of length n whose element i is the index of
  the server hosting VM i. It is equivalent to the binary matrix X = [x_ij] of section 4.2;
  constraint (4.1), every VM on exactly one server, holds by construction and is never
  checked explicitly.
- `is_feasible` checks the capacity constraints (4.2) and (4.3). CPU capacity is
  multiplied by `u_max`, which also covers the QoS constraint (4.9); for `u_max = 1`
  the two coincide.
- Overcommitment is not allowed (α_k = 1), as stated in section 4.4.
- `energy` implements the power model (4.5) and the objective (4.6): an active server
  consumes P_idle + (P_max − P_idle)·u with u = assigned CPU / CPU capacity; inactive
  servers consume nothing (assumption from 4.1).
- `capacity_violation` computes g(X) for the penalty function (4.14): the sum over
  servers of relative CPU and RAM overruns.
- `solution_metrics` returns the energy (4.6), the number of active servers, the mean
  utilisation of active servers (ū from 4.5) and the standard deviation of utilisation.
  The latter is F_bal (4.10), used as a derived indicator rather than an objective, as
  the paper states in 4.5.
- F_mig (4.11) is not implemented, consistent with the weights w₁ = 1, w₂ = w₃ = 0 from
  4.5 and the static (offline) problem variant from 4.1.

## 2. Instance generator (`vmp/scenarios.py`) — assumptions of 4.1 and setup of 6.1

- Server profiles follow the experimental setup of Beloglazov and Buyya: HP ProLiant
  ML110 G4 (2 × 1860 = 3720 MIPS, 4 GB, 86/117 W) and ML110 G5 (2 × 2660 = 5320 MIPS,
  4 GB, 93.7/135 W). A third, stronger profile (10640 MIPS, 8 GB, 110/190 W) is added
  for a more heterogeneous infrastructure.
- VM types model Amazon EC2 instances from the same setup: (2500 MIPS, 870 MB),
  (2000, 1740), (1000, 1740), (500, 613).
- Three scenarios: small 20 VM / 10 PM, medium 80 / 40, large 200 / 100. Server profiles
  and VM types are drawn uniformly at random with a fixed seed, so instances are
  reproducible.
- Scenario names and plot labels come from `vmp/labels.py`; set `VMP_LANG=sr` to get
  the Serbian labels used in the paper's figures.

## 3. Heuristics (`vmp/heuristics.py`) — sections 3.4.2 and 5.4

- FF, BF and WF exactly as described in the paper. BF and WF choose the server by the
  remaining CPU capacity after placement.
- FFD and BFD (First/Best Fit Decreasing): the same heuristics after sorting VMs by
  decreasing CPU demand, the standard stronger baseline in the bin-packing literature.

## 4. PSO (`vmp/pso.py`) — sections 5.1–5.4 (and 4.6)

Common to both variants: canonical PSO with inertia weight (Shi and Eberhart), w
decreasing linearly from 0.9 to 0.4, c1 = c2 = 1.49445, velocity clamped to ±20 % of the
position range, swarm of 30 particles, 200 iterations.

**Variant A — decoding (`pso_decoding`):**
- A particle position is a vector of n real keys in [0, 1] (random keys, Bean 1994).
- Decoding: keys are sorted in decreasing order to obtain the placement order of VMs;
  each VM is placed First Fit onto the first feasible server in a list sorted in advance
  by energy efficiency (P_max / CPU, i.e. watts per MIPS at full load).
- Every decoded solution satisfies (4.1)–(4.3) and (4.9) by construction. This is the
  second approach discussed in section 4.6 of the paper (a decoding procedure that
  generates only feasible solutions).
- Heuristic initialisation: one particle receives keys proportional to CPU demand, so the
  initial swarm contains a solution equivalent to the FFD order.
- If no initial particle can be decoded (the instance has no feasible placement), a
  `ValueError` with a clear message is raised.

**Variant B — penalty function (`pso_penalty`):**
- A particle position is a vector of n real values in [0, m); the integer part is the
  server index for each VM.
- Fitness follows (4.14): f(X) = F(X) + λ·g(X) with λ = 10 × Σ P_max, so any capacity
  violation dominates the energy term.
- One particle is initialised with the FFD placement, so the initial best solution is
  always feasible.

## 5. Experiments (`run_experiments.py`) — chapter 6

- Three scenarios × (5 deterministic heuristics + 2 PSO variants × 10 runs with seeds
  100–109).
- Outputs in `results/`: `results.csv` (one row per run), `energy_comparison.png`,
  `active_servers.png`, `convergence.png`.
- Findings: PSO-decoding is best on all three scenarios (672 W vs. 755 W for the best
  heuristic; 2336 vs. 3308; 5580 vs. 6940, i.e. savings of 11 %, 29 % and 20 %).
  PSO-penalty beats the heuristics on the small scenario (best run 666 W) but degenerates
  to its FFD warm start on the medium and large ones, consistent with the literature and
  the reason decoding-based representations are preferred for larger instances.
- Contribution of the PSO search on top of the decoder (comparison with the decoded FFD
  order without any search, i.e. an energy-aware FFD): a further 12–15 %.

## 6. Demo (`demo.py`)

- Arbitrary input via `--vm`, `--pm`, `--seed`, `--umax`, `--iter`. Without arguments it
  reproduces the instance shown in the paper (60 VM / 30 PM, seed 7).
- Output: a table (power, active servers, utilisation, kWh per year, saving in %) and a
  data-center figure `results/demo_placement.png` (First Fit on top, PSO below; servers as
  columns, VMs stacked, idle servers grey, red line = u_max).
