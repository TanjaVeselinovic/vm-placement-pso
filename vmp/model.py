from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VirtualMachine:
    cpu: float
    ram: float


@dataclass(frozen=True)
class PhysicalMachine:
    cpu: float
    ram: float
    p_idle: float
    p_max: float


@dataclass(frozen=True)
class Instance:
    name: str
    vms: tuple
    pms: tuple
    u_max: float = 1.0

    @property
    def n(self):
        return len(self.vms)

    @property
    def m(self):
        return len(self.pms)

    def arrays(self):
        vm_cpu = np.array([v.cpu for v in self.vms])
        vm_ram = np.array([v.ram for v in self.vms])
        pm_cpu = np.array([p.cpu for p in self.pms])
        pm_ram = np.array([p.ram for p in self.pms])
        p_idle = np.array([p.p_idle for p in self.pms])
        p_max = np.array([p.p_max for p in self.pms])
        return vm_cpu, vm_ram, pm_cpu, pm_ram, p_idle, p_max


def server_loads(instance, assignment):
    vm_cpu, vm_ram, _, _, _, _ = instance.arrays()
    cpu_load = np.bincount(assignment, weights=vm_cpu, minlength=instance.m)
    ram_load = np.bincount(assignment, weights=vm_ram, minlength=instance.m)
    return cpu_load, ram_load


def is_feasible(instance, assignment):
    _, _, pm_cpu, pm_ram, _, _ = instance.arrays()
    cpu_load, ram_load = server_loads(instance, assignment)
    eps = 1e-9
    return bool(
        np.all(cpu_load <= instance.u_max * pm_cpu + eps)
        and np.all(ram_load <= pm_ram + eps)
    )


def capacity_violation(instance, assignment):
    _, _, pm_cpu, pm_ram, _, _ = instance.arrays()
    cpu_load, ram_load = server_loads(instance, assignment)
    cpu_over = np.maximum(0.0, cpu_load - instance.u_max * pm_cpu) / pm_cpu
    ram_over = np.maximum(0.0, ram_load - pm_ram) / pm_ram
    return float(np.sum(cpu_over) + np.sum(ram_over))


def energy(instance, assignment):
    _, _, pm_cpu, _, p_idle, p_max = instance.arrays()
    cpu_load, _ = server_loads(instance, assignment)
    active = cpu_load > 0
    u = np.where(active, cpu_load / pm_cpu, 0.0)
    return float(np.sum(np.where(active, p_idle + (p_max - p_idle) * u, 0.0)))


def solution_metrics(instance, assignment):
    _, _, pm_cpu, _, _, _ = instance.arrays()
    cpu_load, _ = server_loads(instance, assignment)
    active = cpu_load > 0
    u = cpu_load[active] / pm_cpu[active]
    return {
        "energy": energy(instance, assignment),
        "active_servers": int(np.sum(active)),
        "mean_utilization": float(np.mean(u)) if u.size else 0.0,
        "load_stddev": float(np.std(u)) if u.size else 0.0,
        "feasible": is_feasible(instance, assignment),
    }
