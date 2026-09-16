from dataclasses import dataclass, field

import numpy as np

from .heuristics import first_fit_decreasing
from .model import capacity_violation, energy


@dataclass
class PSOParams:
    swarm_size: int = 30
    iterations: int = 200
    w_start: float = 0.9
    w_end: float = 0.4
    c1: float = 1.49445
    c2: float = 1.49445


@dataclass
class PSOResult:
    assignment: np.ndarray
    energy: float
    feasible: bool
    history: list = field(default_factory=list)


def _efficiency_order(instance):
    _, _, pm_cpu, _, _, p_max = instance.arrays()
    return np.argsort(p_max / pm_cpu, kind="stable")


def decode_random_keys(instance, keys, server_order):
    vm_cpu, vm_ram, pm_cpu, pm_ram, _, _ = instance.arrays()
    cpu_free = (instance.u_max * pm_cpu)[server_order]
    ram_free = pm_ram[server_order].copy()
    assignment = np.full(instance.n, -1, dtype=int)
    vm_order = np.argsort(-keys, kind="stable")
    for i in vm_order:
        fits = (cpu_free >= vm_cpu[i] - 1e-9) & (ram_free >= vm_ram[i] - 1e-9)
        pos = np.flatnonzero(fits)
        if pos.size == 0:
            return None
        j = pos[0]
        assignment[i] = server_order[j]
        cpu_free[j] -= vm_cpu[i]
        ram_free[j] -= vm_ram[i]
    return assignment


def pso_decoding(instance, params=PSOParams(), seed=0):
    rng = np.random.default_rng(seed)
    n, s = instance.n, params.swarm_size
    server_order = _efficiency_order(instance)
    vm_cpu, _, _, _, _, _ = instance.arrays()

    x = rng.random((s, n))
    x[0] = vm_cpu / vm_cpu.max()
    v = np.zeros((s, n))

    def fitness(keys):
        assignment = decode_random_keys(instance, keys, server_order)
        if assignment is None:
            return np.inf, None
        return energy(instance, assignment), assignment

    fits, assigns = zip(*(fitness(x[k]) for k in range(s)))
    fits = np.array(fits)
    if not np.isfinite(fits).any():
        raise ValueError(
            f"Instance '{instance.name}' has no feasible placement: none of the {s} "
            "initial particles could be decoded without exceeding server capacity."
        )
    pbest_x, pbest_f = x.copy(), fits.copy()
    g = int(np.argmin(fits))
    gbest_x, gbest_f, gbest_a = x[g].copy(), fits[g], assigns[g]
    history = [gbest_f]

    for t in range(params.iterations):
        w = params.w_start - (params.w_start - params.w_end) * t / params.iterations
        r1, r2 = rng.random((s, n)), rng.random((s, n))
        v = w * v + params.c1 * r1 * (pbest_x - x) + params.c2 * r2 * (gbest_x - x)
        v = np.clip(v, -0.2, 0.2)
        x = np.clip(x + v, 0.0, 1.0)
        for k in range(s):
            f, a = fitness(x[k])
            if f < pbest_f[k]:
                pbest_f[k], pbest_x[k] = f, x[k].copy()
            if f < gbest_f:
                gbest_f, gbest_x, gbest_a = f, x[k].copy(), a
        history.append(gbest_f)

    return PSOResult(gbest_a, gbest_f, True, history)


def pso_penalty(instance, params=PSOParams(), seed=0, penalty=None):
    rng = np.random.default_rng(seed)
    n, m, s = instance.n, instance.m, params.swarm_size
    _, _, _, _, _, p_max = instance.arrays()
    lam = penalty if penalty is not None else 10.0 * float(np.sum(p_max))

    x = rng.random((s, n)) * m
    warm = first_fit_decreasing(instance)
    if warm is not None:
        x[0] = warm + 0.5
    v = np.zeros((s, n))

    def to_assignment(pos):
        return np.clip(pos.astype(int), 0, m - 1)

    def fitness(pos):
        a = to_assignment(pos)
        return energy(instance, a) + lam * capacity_violation(instance, a), a

    fits, assigns = zip(*(fitness(x[k]) for k in range(s)))
    fits = np.array(fits)
    pbest_x, pbest_f = x.copy(), fits.copy()
    g = int(np.argmin(fits))
    gbest_x, gbest_f, gbest_a = x[g].copy(), fits[g], assigns[g]
    history = [gbest_f]

    for t in range(params.iterations):
        w = params.w_start - (params.w_start - params.w_end) * t / params.iterations
        r1, r2 = rng.random((s, n)), rng.random((s, n))
        v = w * v + params.c1 * r1 * (pbest_x - x) + params.c2 * r2 * (gbest_x - x)
        v = np.clip(v, -0.2 * m, 0.2 * m)
        x = np.clip(x + v, 0.0, m - 1e-9)
        for k in range(s):
            f, a = fitness(x[k])
            if f < pbest_f[k]:
                pbest_f[k], pbest_x[k] = f, x[k].copy()
            if f < gbest_f:
                gbest_f, gbest_x, gbest_a = f, x[k].copy(), a
        history.append(gbest_f)

    feasible = capacity_violation(instance, gbest_a) == 0.0
    return PSOResult(gbest_a, energy(instance, gbest_a), feasible, history)
