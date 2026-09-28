"""Manifest of the standard CWRU 12kHz Drive-End fault dataset subset used
in this project.

Subset: 12 kHz Drive End bearing fault data, fault diameters 0.007" /
0.014" / 0.021", centroid fault location OR@6:00, motor loads 0-3 HP
(approx. 1797/1772/1750/1730 RPM), plus normal baseline. This mirrors the
most commonly used CWRU benchmark subset in the fault-diagnosis literature.

File numbers below correspond to the official CWRU Bearing Data Center
file numbering (https://engineering.case.edu/bearingdatacenter).
Each entry maps: file_number -> (label, load_hp).
"""
from typing import Dict, Tuple

# label -> {load_hp: file_number}
CWRU_12K_DE_FILES: Dict[str, Dict[int, int]] = {
    "normal": {0: 97, 1: 98, 2: 99, 3: 100},
    "ball_007": {0: 118, 1: 119, 2: 120, 3: 121},
    "ball_014": {0: 185, 1: 186, 2: 187, 3: 188},
    "ball_021": {0: 222, 1: 223, 2: 225, 3: 226},
    "inner_race_007": {0: 105, 1: 106, 2: 107, 3: 108},
    "inner_race_014": {0: 169, 1: 170, 2: 171, 3: 172},
    "inner_race_021": {0: 209, 1: 210, 2: 211, 3: 212},
    "outer_race_007": {0: 130, 1: 131, 2: 132, 3: 133},
    "outer_race_014": {0: 197, 1: 198, 2: 199, 3: 200},
    "outer_race_021": {0: 234, 1: 235, 2: 236, 3: 237},
}


def build_index() -> Dict[str, dict]:
    """Return {filename: {"label": str, "load_hp": int}} for downloading."""
    index = {}
    for label, loads in CWRU_12K_DE_FILES.items():
        for load_hp, file_number in loads.items():
            filename = f"{file_number}.mat"
            index[filename] = {"label": label, "load_hp": load_hp, "file_number": file_number}
    return index


def file_number_to_url(file_number: int) -> str:
    """Construct the CWRU Bearing Data Center download URL for a file number.

    The Bearing Data Center serves files at a stable per-file URL of the
    form https://engineering.case.edu/sites/default/files/<id>.mat
    """
    return f"https://engineering.case.edu/sites/default/files/{file_number}.mat"
