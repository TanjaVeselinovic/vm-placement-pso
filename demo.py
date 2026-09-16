import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vmp.heuristics import first_fit, first_fit_decreasing
from vmp.labels import L
from vmp.model import solution_metrics
from vmp.pso import PSOParams, pso_decoding
from vmp.scenarios import PM_PROFILES, generate_instance

RESULTS = Path(__file__).parent / "results"

VM_COLORS = ["#4878a8", "#5a9a68", "#c9a24b", "#8a6fae"]


def run_demo(n_vms, n_pms, seed, u_max, iterations):
    instance = generate_instance(
        f"demo ({n_vms} VM / {n_pms} PM, seed={seed})", n_vms, n_pms, seed, u_max
    )
    results = {}
    for name, algo in [("FF", first_fit), ("FFD", first_fit_decreasing)]:
        a = algo(instance)
        if a is not None:
            results[name] = a
    pso = pso_decoding(instance, PSOParams(swarm_size=30, iterations=iterations), seed=seed)
    results["PSO"] = pso.assignment
    return instance, results


def print_table(instance, results):
    base = None
    print(f"\nScenario: {instance.name}, umax={instance.u_max}")
    print(f"{'Algorithm':10s} {'Power [W]':>13s} {'Active PM':>11s} "
          f"{'Utilis.':>10s} {'kWh/year':>10s} {'Saving':>8s}")
    print("-" * 68)
    for name, assignment in results.items():
        met = solution_metrics(instance, assignment)
        kwh = met["energy"] * 8760 / 1000
        if base is None:
            base = met["energy"]
        saving = (1 - met["energy"] / base) * 100
        print(f"{name:10s} {met['energy']:13.1f} {met['active_servers']:11d} "
              f"{met['mean_utilization']*100:9.1f}% {kwh:10.0f} {saving:7.1f}%")


def plot_datacenter(instance, results, out_path):
    vm_cpu, _, pm_cpu, _, _, _ = instance.arrays()
    shown = {k: results[k] for k in ("FF", "PSO") if k in results}
    fig, axes = plt.subplots(len(shown), 1, figsize=(max(10, instance.m * 0.35), 4 * len(shown)))
    profile_cpu = [p.cpu for p in PM_PROFILES]

    for ax, (name, assignment) in zip(np.atleast_1d(axes), shown.items()):
        met = solution_metrics(instance, assignment)
        for j in range(instance.m):
            vms_here = np.flatnonzero(assignment == j)
            if vms_here.size == 0:
                ax.bar(j, 1.0, width=0.75, color="#e8e8e8", edgecolor="#bbbbbb")
                continue
            bottom = 0.0
            for i in vms_here:
                share = vm_cpu[i] / pm_cpu[j]
                color = VM_COLORS[profile_share_color(vm_cpu[i])]
                ax.bar(j, share, width=0.75, bottom=bottom,
                       color=color, edgecolor="white", linewidth=0.5)
                bottom += share
        sizes = ["S", "M", "L"]
        labels = [sizes[profile_cpu.index(instance.pms[j].cpu)] for j in range(instance.m)]
        ax.set_xticks(range(instance.m))
        ax.set_xticklabels(labels, fontsize=7)
        ax.axhline(instance.u_max, color="#c05a5a", linestyle="--", linewidth=1)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(L["cpu_utilisation"])
        ax.set_title(L["placement_title"].format(
            name=name, energy=met["energy"], active=met["active_servers"], total=instance.m))
    axes_flat = np.atleast_1d(axes)
    axes_flat[-1].set_xlabel(L["servers_axis"])
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def profile_share_color(cpu):
    thresholds = [750, 1500, 2250]
    for idx, t in enumerate(thresholds):
        if cpu <= t:
            return idx
    return 3


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VM placement demo: heuristics vs. PSO")
    parser.add_argument("--vm", type=int, default=60, help="number of virtual machines")
    parser.add_argument("--pm", type=int, default=30, help="number of physical servers")
    parser.add_argument("--seed", type=int, default=7, help="instance generator seed (7 = the instance shown in the paper)")
    parser.add_argument("--umax", type=float, default=1.0, help="CPU utilisation cap u_max (0, 1]")
    parser.add_argument("--iter", type=int, default=200, help="number of PSO iterations")
    args = parser.parse_args()

    RESULTS.mkdir(exist_ok=True)
    instance, results = run_demo(args.vm, args.pm, args.seed, args.umax, args.iter)
    print_table(instance, results)
    out = RESULTS / "demo_placement.png"
    plot_datacenter(instance, results, out)
    print(f"\nPlacement figure: {out}")
