import numpy as np

from .labels import L
from .model import Instance, PhysicalMachine, VirtualMachine

PM_PROFILES = [
    PhysicalMachine(cpu=3720.0, ram=4096.0, p_idle=86.0, p_max=117.0),
    PhysicalMachine(cpu=5320.0, ram=4096.0, p_idle=93.7, p_max=135.0),
    PhysicalMachine(cpu=10640.0, ram=8192.0, p_idle=110.0, p_max=190.0),
]

VM_TYPES = [
    VirtualMachine(cpu=2500.0, ram=870.0),
    VirtualMachine(cpu=2000.0, ram=1740.0),
    VirtualMachine(cpu=1000.0, ram=1740.0),
    VirtualMachine(cpu=500.0, ram=613.0),
]


def generate_instance(name, n_vms, n_pms, seed, u_max=1.0):
    rng = np.random.default_rng(seed)
    pm_idx = rng.integers(0, len(PM_PROFILES), size=n_pms)
    vm_idx = rng.integers(0, len(VM_TYPES), size=n_vms)
    pms = tuple(PM_PROFILES[i] for i in pm_idx)
    vms = tuple(VM_TYPES[i] for i in vm_idx)
    return Instance(name=name, vms=vms, pms=pms, u_max=u_max)


def default_scenarios(seed=2026, u_max=1.0):
    return [
        generate_instance(L["small"], 20, 10, seed, u_max),
        generate_instance(L["medium"], 80, 40, seed + 1, u_max),
        generate_instance(L["large"], 200, 100, seed + 2, u_max),
    ]
