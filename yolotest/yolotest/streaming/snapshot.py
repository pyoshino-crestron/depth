from dataclasses import dataclass

@dataclass
class Snapshot:
    fov: float
    predicted: float | None
    actual_distance: float
    corrected_distance: float | None

    