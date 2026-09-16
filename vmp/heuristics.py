import numpy as np


def _place_in_order(instance, vm_order, choose):
    vm_cpu, vm_ram, pm_cpu, pm_ram, _, _ = instance.arrays()
    cpu_free = instance.u_max * pm_cpu.copy()
    ram_free = pm_ram.copy()
    assignment = np.full(instance.n, -1, dtype=int)
    for i in vm_order:
        fits = (cpu_free >= vm_cpu[i] - 1e-9) & (ram_free >= vm_ram[i] - 1e-9)
        candidates = np.flatnonzero(fits)
        if candidates.size == 0:
            return None
        j = choose(candidates, cpu_free, ram_free, vm_cpu[i], vm_ram[i])
        assignment[i] = j
        cpu_free[j] -= vm_cpu[i]
        ram_free[j] -= vm_ram[i]
    return assignment


def _first(candidates, cpu_free, ram_free, cpu_req, ram_req):
    return candidates[0]


def _best(candidates, cpu_free, ram_free, cpu_req, ram_req):
    leftover = cpu_free[candidates] - cpu_req
    return candidates[np.argmin(leftover)]


def _worst(candidates, cpu_free, ram_free, cpu_req, ram_req):
    leftover = cpu_free[candidates] - cpu_req
    return candidates[np.argmax(leftover)]


def first_fit(instance):
    return _place_in_order(instance, range(instance.n), _first)


def best_fit(instance):
    return _place_in_order(instance, range(instance.n), _best)


def worst_fit(instance):
    return _place_in_order(instance, range(instance.n), _worst)


def _decreasing_order(instance):
    vm_cpu, _, _, _, _, _ = instance.arrays()
    return np.argsort(-vm_cpu, kind="stable")


def first_fit_decreasing(instance):
    return _place_in_order(instance, _decreasing_order(instance), _first)


def best_fit_decreasing(instance):
    return _place_in_order(instance, _decreasing_order(instance), _best)


HEURISTICS = {
    "FF": first_fit,
    "BF": best_fit,
    "WF": worst_fit,
    "FFD": first_fit_decreasing,
    "BFD": best_fit_decreasing,
}
