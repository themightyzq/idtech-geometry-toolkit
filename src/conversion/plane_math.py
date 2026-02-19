"""
Plane geometry for idTech brush faces.

Primary representation: normal vector + distance from origin.
Can convert to/from three non-collinear points (idTech 1 format) and
normal+distance (idTech 4 / brushDef3 format).
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Tuple

Vec3 = Tuple[float, float, float]

EPSILON = 1e-6


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _length(v: Vec3) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _normalize(v: Vec3) -> Vec3:
    ln = _length(v)
    if ln < EPSILON:
        return (0.0, 0.0, 1.0)
    return (v[0] / ln, v[1] / ln, v[2] / ln)


def _scale(v: Vec3, s: float) -> Vec3:
    return (v[0] * s, v[1] * s, v[2] * s)


def _add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


@dataclass
class PlaneGeometry:
    """Represents a brush face plane with texture information.

    Internally stores normal + distance.  Can round-trip to/from three-point
    representation used by idTech 1 MAP format.
    """

    normal: Vec3 = (0.0, 0.0, 1.0)
    dist: float = 0.0
    texture: str = "GROUND1_6"

    # Texture parameters (idTech 1 style)
    offset_x: float = 0.0
    offset_y: float = 0.0
    rotation: float = 0.0
    scale_x: float = 1.0
    scale_y: float = 1.0

    # Optional three-point cache (set by from_three_points)
    _p1: Vec3 = field(default=None, repr=False)
    _p2: Vec3 = field(default=None, repr=False)
    _p3: Vec3 = field(default=None, repr=False)

    # ---------------------------------------------------------------
    # Constructors
    # ---------------------------------------------------------------

    @classmethod
    def from_three_points(cls, p1: Vec3, p2: Vec3, p3: Vec3, texture: str = "GROUND1_6",
                          offset_x: float = 0.0, offset_y: float = 0.0,
                          rotation: float = 0.0, scale_x: float = 1.0,
                          scale_y: float = 1.0) -> "PlaneGeometry":
        """Compute plane from three non-collinear points (winding order matters)."""
        v1 = _sub(p2, p1)
        v2 = _sub(p3, p1)
        normal = _normalize(_cross(v1, v2))
        dist = _dot(normal, p1)
        pg = cls(
            normal=normal,
            dist=dist,
            texture=texture,
            offset_x=offset_x,
            offset_y=offset_y,
            rotation=rotation,
            scale_x=scale_x,
            scale_y=scale_y,
        )
        pg._p1 = p1
        pg._p2 = p2
        pg._p3 = p3
        return pg

    @classmethod
    def from_normal_distance(cls, normal: Vec3, dist: float, texture: str = "GROUND1_6",
                             **tex_kwargs) -> "PlaneGeometry":
        normal = _normalize(normal)
        return cls(normal=normal, dist=dist, texture=texture, **tex_kwargs)

    # ---------------------------------------------------------------
    # Conversions
    # ---------------------------------------------------------------

    def to_three_points(self) -> Tuple[Vec3, Vec3, Vec3]:
        """Generate three non-collinear points lying on this plane.

        If original three-point data is cached, return that directly to
        preserve integer coordinates from box-brush construction.
        """
        if self._p1 is not None:
            return (self._p1, self._p2, self._p3)

        # Find a reference vector not parallel to normal
        n = self.normal
        if abs(n[2]) < 0.9:
            ref = (0.0, 0.0, 1.0)
        else:
            ref = (1.0, 0.0, 0.0)

        u = _normalize(_cross(n, ref))
        v = _cross(n, u)

        origin = _scale(n, self.dist)
        p1 = origin
        p2 = _add(origin, _scale(u, 64.0))
        p3 = _add(origin, _scale(v, 64.0))
        return (p1, p2, p3)

    def to_normal_distance(self) -> Tuple[Vec3, float]:
        return (self.normal, self.dist)
