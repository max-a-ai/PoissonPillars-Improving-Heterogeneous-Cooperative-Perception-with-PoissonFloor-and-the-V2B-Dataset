"""The four v2b classes and how every dataset's labels map onto them.

``car`` = cars, vans, "regular vehicles"; ``bicycle`` = every
two-wheeler and its rider (bicycle, motorcycle, scooter, cyclist);
``pedestrian`` = every pedestrian subtype; everything else (trucks,
buses, trailers, cones, signs, animals) is ``other``.
"""

from __future__ import annotations

CLASSES: tuple[str, ...] = ("car", "bicycle", "pedestrian", "other")

CLASS_MAP: dict[str, str] = {
    # car-like
    "car": "car",
    "van": "car",
    "vehicle": "car",
    "vehicle.car": "car",
    "regular_vehicle": "car",
    "type_vehicle": "car",
    "policecar": "car",
    "longvehicle": "car",
    # bicycles and other two-wheelers
    "bicycle": "bicycle",
    "cyclist": "bicycle",
    "bicyclist": "bicycle",
    "motorcycle": "bicycle",
    "motorcyclist": "bicycle",
    "scooter": "bicycle",
    "e-scooter": "bicycle",
    "wheeled_rider": "bicycle",
    "vehicle.bicycle": "bicycle",
    "vehicle.motorcycle": "bicycle",
    "type_cyclist": "bicycle",
    "bicycle_rider": "bicycle",
    "bicyclerider": "bicycle",
    # V2X-Real spells its riders this way ("motorcylerider" is the
    # dataset's own typo); they are two-wheelers, not pedestrians
    "scooterrider": "bicycle",
    "motorcyclerider": "bicycle",
    "motorcylerider": "bicycle",
    # pedestrians
    "pedestrian": "pedestrian",
    "otherpedestrians": "pedestrian",
    "type_pedestrian": "pedestrian",
    "official_signaler": "pedestrian",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    "human.pedestrian.stroller": "pedestrian",
    "human.pedestrian.wheelchair": "pedestrian",
    "human.pedestrian.personal_mobility": "pedestrian",
    "wheelchair": "pedestrian",
    "child": "pedestrian",
    "roadworker": "pedestrian",
    "stroller": "pedestrian",
}


def canon(label: str) -> str:
    """The v2b class of a dataset label (``other`` when unknown)."""
    return CLASS_MAP.get(str(label).lower().strip(), "other")
