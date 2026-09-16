import os

LANG = os.environ.get("VMP_LANG", "en").lower()

_LABELS = {
    "en": {
        "small": "small (20 VM / 10 PM)",
        "medium": "medium (80 VM / 40 PM)",
        "large": "large (200 VM / 100 PM)",
        "pso_decoding": "PSO-decoding",
        "pso_penalty": "PSO-penalty",
        "power": "Power consumption [W]",
        "active_servers": "Number of active servers",
        "iteration": "Iteration",
        "best_objective": "Best objective value",
        "cpu_utilisation": "CPU utilisation",
        "servers_axis": "Physical servers (S/M/L = server type; grey = switched off)",
        "placement_title": "{name}: {energy:.0f} W, {active} active of {total} servers",
    },
    "sr": {
        "small": "mali (20 VM / 10 PM)",
        "medium": "srednji (80 VM / 40 PM)",
        "large": "veliki (200 VM / 100 PM)",
        "pso_decoding": "PSO-dekod",
        "pso_penalty": "PSO-kazna",
        "power": "Potrošnja energije [W]",
        "active_servers": "Broj aktivnih servera",
        "iteration": "Iteracija",
        "best_objective": "Najbolja vrijednost funkcije cilja",
        "cpu_utilisation": "Iskorištenost CPU",
        "servers_axis": "Fizički serveri (S/M/L = tip servera; sivo = ugašen)",
        "placement_title": "{name}: {energy:.0f} W, {active} aktivnih od {total} servera",
    },
}

L = _LABELS.get(LANG, _LABELS["en"])
