import csv
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vmp.heuristics import HEURISTICS
from vmp.labels import L
from vmp.model import solution_metrics
from vmp.pso import PSOParams, pso_decoding, pso_penalty
from vmp.scenarios import default_scenarios

RESULTS = Path(__file__).parent / "results"
PSO_RUNS = 10
PARAMS = PSOParams(swarm_size=30, iterations=200)


def run_all():
    RESULTS.mkdir(exist_ok=True)
    rows = []
    convergence = {}

    for instance in default_scenarios():
        print(f"\n=== Scenario: {instance.name} ===")

        for name, heur in HEURISTICS.items():
            t0 = time.perf_counter()
            assignment = heur(instance)
            elapsed = time.perf_counter() - t0
            if assignment is None:
                print(f"{name:12s} found no feasible placement")
                continue
            met = solution_metrics(instance, assignment)
            rows.append({"scenario": instance.name, "algorithm": name, "run": 0,
                         "time_s": elapsed, **met})
            print(f"{name:12s} power={met['energy']:9.1f} W  "
                  f"active={met['active_servers']:3d}  t={elapsed:.3f}s")

        for name, algo in [(L["pso_decoding"], pso_decoding), (L["pso_penalty"], pso_penalty)]:
            energies, histories = [], []
            for run in range(PSO_RUNS):
                t0 = time.perf_counter()
                res = algo(instance, PARAMS, seed=100 + run)
                elapsed = time.perf_counter() - t0
                met = solution_metrics(instance, res.assignment)
                met["feasible"] = res.feasible and met["feasible"]
                rows.append({"scenario": instance.name, "algorithm": name,
                             "run": run, "time_s": elapsed, **met})
                energies.append(met["energy"])
                histories.append(res.history)
            convergence[(instance.name, name)] = histories
            print(f"{name:12s} power={np.mean(energies):9.1f} W "
                  f"(min={np.min(energies):.1f}, max={np.max(energies):.1f}, "
                  f"std={np.std(energies):.1f})  t={elapsed:.2f}s/run")

    with open(RESULTS / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return rows, convergence


def plot_energy(rows):
    scenarios = list(dict.fromkeys(r["scenario"] for r in rows))
    algorithms = list(dict.fromkeys(r["algorithm"] for r in rows))
    fig, axes = plt.subplots(1, len(scenarios), figsize=(5 * len(scenarios), 4.2))
    for ax, sc in zip(np.atleast_1d(axes), scenarios):
        means, stds = [], []
        for alg in algorithms:
            vals = [r["energy"] for r in rows
                    if r["scenario"] == sc and r["algorithm"] == alg and r["feasible"]]
            means.append(np.mean(vals) if vals else 0.0)
            stds.append(np.std(vals) if vals else 0.0)
        ax.bar(algorithms, means, yerr=stds, capsize=4, color="#4878a8")
        ax.set_title(sc)
        ax.set_ylabel(L["power"])
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "energy_comparison.png", dpi=200)
    plt.close(fig)


def plot_active_servers(rows):
    scenarios = list(dict.fromkeys(r["scenario"] for r in rows))
    algorithms = list(dict.fromkeys(r["algorithm"] for r in rows))
    fig, axes = plt.subplots(1, len(scenarios), figsize=(5 * len(scenarios), 4.2))
    for ax, sc in zip(np.atleast_1d(axes), scenarios):
        means = []
        for alg in algorithms:
            vals = [r["active_servers"] for r in rows
                    if r["scenario"] == sc and r["algorithm"] == alg and r["feasible"]]
            means.append(np.mean(vals) if vals else 0.0)
        ax.bar(algorithms, means, color="#5a9a68")
        ax.set_title(sc)
        ax.set_ylabel(L["active_servers"])
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "active_servers.png", dpi=200)
    plt.close(fig)


def plot_convergence(convergence):
    scenarios = list(dict.fromkeys(sc for sc, _ in convergence))
    fig, axes = plt.subplots(1, len(scenarios), figsize=(5 * len(scenarios), 4.2))
    colors = {L["pso_decoding"]: "#4878a8", L["pso_penalty"]: "#c05a5a"}
    for ax, sc in zip(np.atleast_1d(axes), scenarios):
        for (s, alg), histories in convergence.items():
            if s != sc:
                continue
            h = np.array([np.minimum.accumulate(x) for x in histories])
            finite = np.isfinite(h).all(axis=1)
            h = h[finite]
            if not len(h):
                continue
            mean = h.mean(axis=0)
            ax.plot(mean, label=alg, color=colors.get(alg))
            ax.fill_between(range(h.shape[1]), h.min(axis=0), h.max(axis=0),
                            alpha=0.15, color=colors.get(alg))
        ax.set_title(sc)
        ax.set_xlabel(L["iteration"])
        ax.set_ylabel(L["best_objective"])
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "convergence.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    rows, convergence = run_all()
    plot_energy(rows)
    plot_active_servers(rows)
    plot_convergence(convergence)
    print(f"\nResults saved to: {RESULTS}")
