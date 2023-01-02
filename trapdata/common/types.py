from dataclasses import dataclass


@dataclass
class CoordinateDMS:
    degrees: int
    minutes: int
    seconds: float


class Location:
    latitude: CoordinateDMS
    longitude: CoordinateDMS
