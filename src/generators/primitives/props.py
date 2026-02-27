"""
Decorative prop primitives for environment dressing.

These are OPEN ELEMENTS — standalone furniture/objects placed inside rooms via
Module Mode. They do NOT need to be sealed and have no portals or footprints.

Following the Structural pattern (Pillar, Arch, Buttress, Battlement).

See CLAUDE.md §2.1 (Geometry-Only Philosophy) — brush geometry only, no entities.
"""

from __future__ import annotations
import math
import random
from typing import Any, Dict, List

from quake_levelgenerator.src.conversion.map_writer import Brush, Plane
from .base import GeometricPrimitive, Vec3


# ---------------------------------------------------------------------------
# Shared radial-segment helpers
# ---------------------------------------------------------------------------
# These build true faceted geometry from _radial_segment() pie-slices.
# Unlike _generate_polygonal_solid() (overlapping rotated boxes that still
# look cubic), radial segments give visible N-sided facets.

# Minimum outer radius for radial segments.  Below this, individual
# pie-slice brush BBoxes fall under the 1-unit GEOM-004 minimum for
# props.  At small radii the object is too tiny to visually
# distinguish facets anyway, so we fall back to
# _generate_polygonal_solid (overlapping-box approach) which always
# passes GEOM-004.
_MIN_RADIAL_RADIUS = 4


def _radial_ring(prim: GeometricPrimitive, cx: float, cy: float,
                 z1: float, z2: float, inner_r: float, outer_r: float,
                 sides: int) -> List[Brush]:
    """Ring of radial pie-slice segments (hollow center).

    Falls back to _generate_polygonal_solid for small radii to stay
    GEOM-004 compliant (1u minimum for props).
    """
    if outer_r < _MIN_RADIAL_RADIUS:
        # Small radius — use overlapping-box approach (passes GEOM-004 1u min)
        return prim._generate_polygonal_solid(
            cx, cy, z1, z2, outer_r, sides,
            texture=prim.texture_structural,
        )
    brushes: List[Brush] = []
    seg = 2 * math.pi / sides
    for i in range(sides):
        brushes.append(prim._radial_segment(
            cx, cy, z1, z2, inner_r, outer_r,
            i * seg, (i + 1) * seg,
            texture=prim.texture_structural,
        ))
    return brushes


def _radial_disk(prim: GeometricPrimitive, cx: float, cy: float,
                 z1: float, z2: float, radius: float,
                 sides: int) -> List[Brush]:
    """Near-solid disk from pie-slice segments.

    Uses a small inner_r (never 0) to avoid degenerate planes where
    inner corners collapse to a single point.  Falls back to
    overlapping boxes for small radii (GEOM-004 compliance, 1u min for props).
    """
    inner_r = max(4, int(radius * 0.3))
    return _radial_ring(prim, cx, cy, z1, z2, inner_r, radius, sides)


# ---------------------------------------------------------------------------
# Tier 1 — Simple props (1-3 brushes, _box only)
# ---------------------------------------------------------------------------

class Crate(GeometricPrimitive):
    """A wooden storage crate with independent width/length."""

    crate_width: float = 32.0
    crate_length: float = 32.0
    crate_height: float = 32.0
    has_lid: bool = False
    lid_thickness: float = 8.0
    has_bands: bool = True
    band_overhang: float = 2.0
    band_height: float = 8.0
    lid_overhang: float = 4.0
    random_seed: int = 0

    @classmethod
    def get_display_name(cls) -> str:
        return "Crate"

    @classmethod
    def get_category(cls) -> str:
        return "Structural"

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "crate_width": {
                "type": "float", "default": 32.0, "min": 16, "max": 64,
                "label": "Width", "description": "Width of the crate (X axis)"
            },
            "crate_length": {
                "type": "float", "default": 32.0, "min": 16, "max": 64,
                "label": "Length", "description": "Length of the crate (Y axis)"
            },
            "crate_height": {
                "type": "float", "default": 32.0, "min": 16, "max": 64,
                "label": "Height", "description": "Height of the main crate body"
            },
            "has_lid": {
                "type": "bool", "default": False, "label": "Lid Overhang",
                "description": "Add a slightly wider lid on top"
            },
            "lid_thickness": {
                "type": "float", "default": 8.0, "min": 8, "max": 16,
                "label": "Lid Thickness", "description": "Thickness of the lid"
            },
            "has_bands": {
                "type": "bool", "default": True, "label": "Iron Bands",
                "description": "Add horizontal iron banding strips"
            },
            "band_overhang": {
                "type": "float", "default": 2.0, "min": 1, "max": 8,
                "label": "Band Overhang", "description": "How far bands protrude from the crate body"
            },
            "band_height": {
                "type": "float", "default": 8.0, "min": 1, "max": 16,
                "label": "Band Height", "description": "Vertical thickness of iron bands"
            },
            "lid_overhang": {
                "type": "float", "default": 4.0, "min": 1, "max": 8,
                "label": "Lid Overhang", "description": "How far the lid extends beyond the crate body"
            },
            "random_seed": {
                "type": "int", "default": 0, "min": 0, "max": 999999,
                "label": "Random Seed", "description": "Seed for variation"
            },
        }

    def generate(self) -> List[Brush]:
        self._reset_tags()
        ox, oy, oz = self.params.origin
        brushes: List[Brush] = []
        hw = self.crate_width / 2
        hl = self.crate_length / 2

        # Main crate body
        brushes.append(self._structural_box(
            ox - hw, oy - hl, oz,
            ox + hw, oy + hl, oz + self.crate_height,
        ))

        top_z = oz + self.crate_height

        # Iron bands at 1/3 and 2/3 height
        if self.has_bands:
            band_extra = self.band_overhang
            for frac in [1 / 3, 2 / 3]:
                bz = oz + self.crate_height * frac - self.band_height / 2
                brushes.append(self._structural_box(
                    ox - hw - band_extra, oy - hl - band_extra, bz,
                    ox + hw + band_extra, oy + hl + band_extra, bz + self.band_height,
                ))

        # Lid overhang
        if self.has_lid:
            lid_extra = self.lid_overhang
            brushes.append(self._structural_box(
                ox - hw - lid_extra, oy - hl - lid_extra, top_z,
                ox + hw + lid_extra, oy + hl + lid_extra, top_z + self.lid_thickness,
            ))

        return brushes


class Table(GeometricPrimitive):
    """A table — rectangular or polygonal slab on legs or pedestal."""

    table_width: float = 64.0
    table_length: float = 96.0
    table_height: float = 32.0
    top_thickness: float = 8.0
    leg_width: float = 8.0
    has_legs: bool = True
    has_apron: bool = True
    apron_height: float = 8.0
    shape_type: str = "rectangular"
    shape_sides: int = 8
    pedestal_base: bool = False

    @classmethod
    def get_display_name(cls) -> str:
        return "Table"

    @classmethod
    def get_category(cls) -> str:
        return "Structural"

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "table_width": {
                "type": "float", "default": 64.0, "min": 32, "max": 128,
                "label": "Width", "description": "Width of the table top"
            },
            "table_length": {
                "type": "float", "default": 96.0, "min": 48, "max": 192,
                "label": "Length", "description": "Length of the table top"
            },
            "table_height": {
                "type": "float", "default": 32.0, "min": 24, "max": 48,
                "label": "Height", "description": "Height from floor to top surface"
            },
            "top_thickness": {
                "type": "float", "default": 8.0, "min": 4, "max": 16,
                "label": "Top Thickness", "description": "Thickness of the table top slab"
            },
            "leg_width": {
                "type": "float", "default": 8.0, "min": 8, "max": 16,
                "label": "Leg Width", "description": "Width of each leg"
            },
            "has_legs": {
                "type": "bool", "default": True, "label": "Has Legs",
                "description": "If false, generates a solid block instead of legs"
            },
            "shape_type": {
                "type": "choice", "default": "rectangular",
                "choices": ["rectangular", "round"],
                "label": "Shape", "description": "rectangular=box, round=cylindrical top"
            },
            "shape_sides": {
                "type": "int", "default": 8, "min": 3, "max": 12,
                "label": "Sides",
                "description": "Facets for round mode (4=square, 8=octagonal, 12=round)"
            },
            "has_apron": {
                "type": "bool", "default": True, "label": "Apron Rails",
                "description": "Add horizontal rails connecting leg tops"
            },
            "apron_height": {
                "type": "float", "default": 8.0, "min": 4, "max": 16,
                "label": "Apron Height", "description": "Height of the apron rail boards"
            },
            "pedestal_base": {
                "type": "bool", "default": False, "label": "Pedestal Base",
                "description": "Single center pedestal instead of 4 legs"
            },
        }

    def generate(self) -> List[Brush]:
        self._reset_tags()
        ox, oy, oz = self.params.origin
        brushes: List[Brush] = []
        hw = self.table_width / 2
        hl = self.table_length / 2
        top_z = oz + self.table_height - self.top_thickness
        leg_h = self.table_height - self.top_thickness

        if self.shape_type == "round":
            radius = min(hw, hl)

            # Round table top (radial disk)
            brushes.extend(_radial_disk(
                self, ox, oy, top_z, oz + self.table_height,
                radius, self.shape_sides,
            ))

            if not self.has_legs:
                # Solid round base
                brushes.extend(_radial_disk(
                    self, ox, oy, oz, top_z,
                    radius, self.shape_sides,
                ))
            elif self.pedestal_base:
                # Single center pedestal column
                ped_r = max(1.0, self._snap_coord(radius * 0.3))
                brushes.extend(_radial_disk(
                    self, ox, oy, oz, top_z,
                    ped_r, self.shape_sides,
                ))
            else:
                # Legs at cardinal points around perimeter
                lw = self.leg_width
                leg_inset = radius * 0.7
                num_legs = min(4, self.shape_sides)
                for i in range(num_legs):
                    angle = i * (2 * math.pi / num_legs)
                    lx = self._snap_coord(ox + leg_inset * math.cos(angle))
                    ly = self._snap_coord(oy + leg_inset * math.sin(angle))
                    brushes.append(self._structural_box(
                        lx - lw / 2, ly - lw / 2, oz,
                        lx + lw / 2, ly + lw / 2, oz + leg_h,
                    ))
        else:
            # Rectangular table
            if not self.has_legs:
                brushes.append(self._structural_box(
                    ox - hw, oy - hl, oz,
                    ox + hw, oy + hl, oz + self.table_height,
                ))
            else:
                # Table top slab
                brushes.append(self._structural_box(
                    ox - hw, oy - hl, top_z,
                    ox + hw, oy + hl, oz + self.table_height,
                ))

                if self.pedestal_base:
                    # Single center pedestal
                    ped_w = max(1.0, self._snap_coord(min(hw, hl) * 0.5))
                    brushes.append(self._structural_box(
                        ox - ped_w, oy - ped_w, oz,
                        ox + ped_w, oy + ped_w, oz + leg_h,
                    ))
                else:
                    # Four corner legs
                    lw = self.leg_width
                    inset = lw / 2
                    for sx, sy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
                        lx = ox + sx * (hw - inset)
                        ly = oy + sy * (hl - inset)
                        brushes.append(self._structural_box(
                            lx - lw / 2, ly - lw / 2, oz,
                            lx + lw / 2, ly + lw / 2, oz + leg_h,
                        ))

                    # Apron rails connecting leg tops
                    if self.has_apron:
                        apron_h = self.apron_height
                        apron_z = top_z - apron_h
                        apron_inset = self.leg_width / 2
                        # Front rail
                        brushes.append(self._structural_box(
                            ox - hw + apron_inset, oy - hl + apron_inset, apron_z,
                            ox + hw - apron_inset, oy - hl + apron_inset + apron_h, top_z,
                        ))
                        # Back rail
                        brushes.append(self._structural_box(
                            ox - hw + apron_inset, oy + hl - apron_inset - apron_h, apron_z,
                            ox + hw - apron_inset, oy + hl - apron_inset, top_z,
                        ))
                        # Left rail
                        brushes.append(self._structural_box(
                            ox - hw + apron_inset, oy - hl + apron_inset, apron_z,
                            ox - hw + apron_inset + apron_h, oy + hl - apron_inset, top_z,
                        ))
                        # Right rail
                        brushes.append(self._structural_box(
                            ox + hw - apron_inset - apron_h, oy - hl + apron_inset, apron_z,
                            ox + hw - apron_inset, oy + hl - apron_inset, top_z,
                        ))

        return brushes


class Sarcophagus(GeometricPrimitive):
    """A stone coffin with configurable body shape and lid."""

    body_width: float = 32.0
    body_length: float = 80.0
    body_height: float = 24.0
    body_shape: str = "rectangular"
    body_sides: int = 6
    lid_style: str = "flat"
    has_plinth: bool = False
    plinth_height: float = 8.0
    plinth_overhang: float = 8.0
    has_rim: bool = True
    rim_height: float = 8.0
    rim_overhang: float = 2.0
    lid_thickness: float = 8.0
    lid_overhang: float = 4.0

    @classmethod
    def get_display_name(cls) -> str:
        return "Sarcophagus"

    @classmethod
    def get_category(cls) -> str:
        return "Structural"

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "body_width": {
                "type": "float", "default": 32.0, "min": 24, "max": 48,
                "label": "Width", "description": "Width of the coffin body"
            },
            "body_length": {
                "type": "float", "default": 80.0, "min": 64, "max": 112,
                "label": "Length", "description": "Length of the coffin body"
            },
            "body_height": {
                "type": "float", "default": 24.0, "min": 16, "max": 40,
                "label": "Height", "description": "Height of the coffin body"
            },
            "body_shape": {
                "type": "choice", "default": "rectangular",
                "choices": ["rectangular", "coffin", "round"],
                "label": "Body Shape",
                "description": "rectangular=box, coffin=tapered hexagon, round=cylindrical"
            },
            "body_sides": {
                "type": "int", "default": 6, "min": 3, "max": 12,
                "label": "Body Sides",
                "description": "Facets for round mode (4=square, 8=octagonal, 12=round)"
            },
            "lid_style": {
                "type": "choice", "default": "flat",
                "choices": ["flat", "peaked", "absent"],
                "label": "Lid Style", "description": "flat=slab lid, peaked=angled lid, absent=open top"
            },
            "has_plinth": {
                "type": "bool", "default": False, "label": "Plinth Base",
                "description": "Add a wider base platform"
            },
            "plinth_height": {
                "type": "float", "default": 8.0, "min": 2, "max": 16,
                "label": "Plinth Height", "description": "Height of the base platform"
            },
            "plinth_overhang": {
                "type": "float", "default": 8.0, "min": 4, "max": 16,
                "label": "Plinth Overhang", "description": "How far the plinth extends beyond the body"
            },
            "has_rim": {
                "type": "bool", "default": True, "label": "Rim/Lip",
                "description": "Add a rim between body and lid for visual definition"
            },
            "rim_height": {
                "type": "float", "default": 8.0, "min": 2, "max": 16,
                "label": "Rim Height", "description": "Vertical thickness of the rim"
            },
            "rim_overhang": {
                "type": "float", "default": 2.0, "min": 1, "max": 8,
                "label": "Rim Overhang", "description": "How far the rim extends beyond the body"
            },
            "lid_thickness": {
                "type": "float", "default": 8.0, "min": 4, "max": 16,
                "label": "Lid Thickness", "description": "Thickness of the flat lid slab"
            },
            "lid_overhang": {
                "type": "float", "default": 4.0, "min": 1, "max": 8,
                "label": "Lid Overhang", "description": "How far the lid extends beyond the body"
            },
        }

    def generate(self) -> List[Brush]:
        self._reset_tags()
        ox, oy, oz = self.params.origin
        brushes: List[Brush] = []
        hw = self.body_width / 2
        hl = self.body_length / 2
        tex = self.params.texture

        base_z = oz

        # Optional plinth (wider base)
        if self.has_plinth:
            plinth_extra = self.plinth_overhang
            brushes.append(self._structural_box(
                ox - hw - plinth_extra, oy - hl - plinth_extra, base_z,
                ox + hw + plinth_extra, oy + hl + plinth_extra, base_z + self.plinth_height,
            ))
            base_z += self.plinth_height

        top_body = self._snap_coord(base_z + self.body_height)

        # Main coffin body
        if self.body_shape == "coffin":
            brushes.append(self._coffin_body_brush(
                ox, oy, base_z, top_body, hw, hl, tex,
            ))
        elif self.body_shape == "round":
            r = min(hw, hl)
            brushes.extend(_radial_disk(
                self, ox, oy, base_z, top_body,
                r, self.body_sides,
            ))
        else:
            # Rectangular (default)
            brushes.append(self._structural_box(
                ox - hw, oy - hl, base_z,
                ox + hw, oy + hl, top_body,
            ))

        top_z = top_body

        # Rim / lip around body top (works with any lid style including absent)
        if self.has_rim:
            rim_h = self.rim_height
            rim_extra = self.rim_overhang
            if self.body_shape == "coffin":
                brushes.append(self._coffin_body_brush(
                    ox, oy, top_z, top_z + rim_h, hw, hl, tex,
                    overhang=rim_extra,
                ))
            elif self.body_shape == "round":
                r = min(hw, hl)
                brushes.extend(_radial_disk(
                    self, ox, oy, top_z, top_z + rim_h,
                    r + rim_extra, self.body_sides,
                ))
            else:
                brushes.append(self._structural_box(
                    ox - hw - rim_extra, oy - hl - rim_extra, top_z,
                    ox + hw + rim_extra, oy + hl + rim_extra, top_z + rim_h,
                ))
            top_z += rim_h

        # Lid
        if self.lid_style == "flat":
            lid_h = self.lid_thickness
            lid_extra = self.lid_overhang
            if self.body_shape == "coffin":
                brushes.append(self._coffin_body_brush(
                    ox, oy, top_z, top_z + lid_h, hw, hl, tex,
                    overhang=lid_extra,
                ))
            elif self.body_shape == "round":
                r = min(hw, hl)
                brushes.extend(_radial_disk(
                    self, ox, oy, top_z, top_z + lid_h,
                    r + lid_extra, self.body_sides,
                ))
            else:
                brushes.append(self._structural_box(
                    ox - hw - lid_extra, oy - hl - lid_extra, top_z,
                    ox + hw + lid_extra, oy + hl + lid_extra, top_z + lid_h,
                ))
        elif self.lid_style == "peaked":
            lid_extra = self.lid_overhang
            peak_h = max(1.0, self._snap_coord(self.lid_thickness))
            if self.body_shape == "coffin":
                brushes.append(self._coffin_peaked_lid_brush(
                    ox, oy, top_z, hw, hl, peak_h, lid_extra, tex,
                ))
            elif self.body_shape == "round":
                r = min(hw, hl)
                brushes.append(self._peaked_lid_brush(
                    ox, oy, top_z, r + lid_extra, r + lid_extra, peak_h, tex,
                ))
            else:
                brushes.append(self._peaked_lid_brush(
                    ox, oy, top_z, hw + lid_extra, hl + lid_extra, peak_h, tex,
                ))
        # "absent" = no lid

        return brushes

    def _coffin_body_brush(self, ox: float, oy: float,
                           z_bot: float, z_top: float,
                           hw: float, hl: float, tex: str,
                           overhang: float = 0.0) -> Brush:
        """Build a hexagonal coffin body: wide at shoulders, tapered at head/foot.

        Uses a single 8-plane convex brush (idTech 1 inward-facing normals).
        overhang adds uniform extra width to all edges (for rim/lid).
        """
        oh = overhang
        # Coffin proportions: shoulder at 30% from center toward head
        shoulder_y = self._snap_coord(oy + hl * 0.3)
        head_hw = self._snap_coord(hw * 0.7 + oh)
        foot_hw = self._snap_coord(hw * 0.5 + oh)
        hw = self._snap_coord(hw + oh)
        hl_ext = self._snap_coord(hl + oh)
        z_bot = self._snap_coord(z_bot)
        z_top = self._snap_coord(z_top)

        planes = [
            # Bottom (normal +Z inward)
            Plane((ox - hw, oy - hl_ext, z_bot),
                  (ox + hw, oy - hl_ext, z_bot),
                  (ox - hw, oy + hl_ext, z_bot), tex),
            # Top (normal -Z inward)
            Plane((ox - hw, oy - hl_ext, z_top),
                  (ox - hw, oy + hl_ext, z_top),
                  (ox + hw, oy - hl_ext, z_top), tex),
            # Head end +Y (normal -Y inward)
            Plane((ox - head_hw, oy + hl_ext, z_bot),
                  (ox + head_hw, oy + hl_ext, z_bot),
                  (ox - head_hw, oy + hl_ext, z_top), tex),
            # Foot end -Y (normal +Y inward)
            Plane((ox - foot_hw, oy - hl_ext, z_bot),
                  (ox - foot_hw, oy - hl_ext, z_top),
                  (ox + foot_hw, oy - hl_ext, z_bot), tex),
            # Left: head to shoulder (normal inward +X)
            Plane((ox - head_hw, oy + hl_ext, z_bot),
                  (ox - head_hw, oy + hl_ext, z_top),
                  (ox - hw, shoulder_y, z_bot), tex),
            # Left: shoulder to foot (normal inward +X)
            Plane((ox - hw, shoulder_y, z_bot),
                  (ox - hw, shoulder_y, z_top),
                  (ox - foot_hw, oy - hl_ext, z_bot), tex),
            # Right: head to shoulder (normal inward -X)
            Plane((ox + head_hw, oy + hl_ext, z_bot),
                  (ox + hw, shoulder_y, z_bot),
                  (ox + head_hw, oy + hl_ext, z_top), tex),
            # Right: shoulder to foot (normal inward -X)
            Plane((ox + hw, shoulder_y, z_bot),
                  (ox + foot_hw, oy - hl_ext, z_bot),
                  (ox + hw, shoulder_y, z_top), tex),
        ]
        return Brush(planes=planes, brush_id=self._next_id())

    def _coffin_peaked_lid_brush(self, ox: float, oy: float,
                                 z_base: float, hw: float, hl: float,
                                 peak_h: float, overhang: float,
                                 tex: str) -> Brush:
        """Build a peaked lid following the coffin hexagonal outline.

        Uses the same shoulder/head/foot proportions as the coffin body.
        The ridge runs along Y at x=ox, peak height = peak_h.
        3 sections (head, shoulder-to-foot) merged into one brush via
        the rectangular peaked prism at the shoulder (widest) width —
        the hexagonal body planes clip the overhang naturally.

        In practice we build this as two brushes would be needed for a
        perfect hex peaked lid. Instead, use the shoulder (max) width
        for the peaked prism — the overhang past head/foot looks natural
        like a lid that extends beyond the narrower body sections.
        """
        oh = overhang
        lid_hw = self._snap_coord(hw + oh)
        lid_hl = self._snap_coord(hl + oh)
        return self._peaked_lid_brush(ox, oy, z_base, lid_hw, lid_hl, peak_h, tex)

    def _peaked_lid_brush(self, ox: float, oy: float, z_base: float,
                          lid_hw: float, lid_hl: float,
                          peak_h: float, tex: str) -> Brush:
        """Build a peaked lid as a single pentahedral prism.

        Ridge runs along Y (length). Cross-section in XZ is an isoceles
        triangle: base at z_base spanning [-lid_hw, +lid_hw], apex at z_base+peak_h.
        Uses idTech 1 inward-facing normals (5 planes).
        """
        lid_hw = self._snap_coord(lid_hw)
        lid_hl = self._snap_coord(lid_hl)
        z_base = self._snap_coord(z_base)
        peak_h = self._snap_coord(peak_h)
        z_peak = z_base + peak_h

        planes = [
            # Bottom (normal +Z inward)
            Plane((ox - lid_hw, oy - lid_hl, z_base),
                  (ox + lid_hw, oy - lid_hl, z_base),
                  (ox - lid_hw, oy + lid_hl, z_base), tex),
            # Front -Y (normal +Y inward)
            Plane((ox - lid_hw, oy - lid_hl, z_base),
                  (ox, oy - lid_hl, z_peak),
                  (ox + lid_hw, oy - lid_hl, z_base), tex),
            # Back +Y (normal -Y inward)
            Plane((ox - lid_hw, oy + lid_hl, z_base),
                  (ox + lid_hw, oy + lid_hl, z_base),
                  (ox, oy + lid_hl, z_peak), tex),
            # Left slope (normal inward: +X, -Z)
            Plane((ox - lid_hw, oy - lid_hl, z_base),
                  (ox - lid_hw, oy + lid_hl, z_base),
                  (ox, oy - lid_hl, z_peak), tex),
            # Right slope (normal inward: -X, -Z)
            Plane((ox + lid_hw, oy - lid_hl, z_base),
                  (ox, oy - lid_hl, z_peak),
                  (ox + lid_hw, oy + lid_hl, z_base), tex),
        ]
        return Brush(planes=planes, brush_id=self._next_id())


class Barrel(GeometricPrimitive):
    """A solid barrel built from radial segments with metal band hoops.

    Body is a bulging cylinder made of pie-slice ring sections, always
    capped solid at top and bottom.  Bands are evenly spaced metal hoops
    that protrude from the body — top and bottom bands double as rims.
    """

    barrel_radius: float = 16.0
    barrel_height: float = 40.0
    sides: int = 8
    bulge_ratio: float = 1.15
    band_count: int = 4
    band_height: float = 4.0
    band_width: float = 4.0

    @classmethod
    def get_display_name(cls) -> str:
        return "Barrel"

    @classmethod
    def get_category(cls) -> str:
        return "Structural"

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "barrel_radius": {
                "type": "float", "default": 16.0, "min": 12, "max": 32,
                "label": "Radius", "description": "Barrel radius at widest point"
            },
            "barrel_height": {
                "type": "float", "default": 40.0, "min": 24, "max": 64,
                "label": "Height", "description": "Total barrel height"
            },
            "sides": {
                "type": "int", "default": 8, "min": 4, "max": 12,
                "label": "Sides", "description": "Number of facets (4=square, 8=octagonal)"
            },
            "bulge_ratio": {
                "type": "float", "default": 1.15, "min": 1.0, "max": 1.3,
                "label": "Bulge", "description": "How pronounced the barrel bulge is (1.0=straight)"
            },
            "band_count": {
                "type": "int", "default": 4, "min": 0, "max": 6,
                "label": "Bands", "description": "Number of metal hoop bands (includes rim bands)"
            },
            "band_height": {
                "type": "float", "default": 4.0, "min": 1, "max": 16,
                "label": "Band Height", "description": "Vertical thickness of each band"
            },
            "band_width": {
                "type": "float", "default": 4.0, "min": 1, "max": 16,
                "label": "Band Width", "description": "How far each band protrudes from the body"
            },
        }

    def generate(self) -> List[Brush]:
        self._reset_tags()
        ox, oy, oz = self.params.origin
        brushes: List[Brush] = []
        r = self.barrel_radius
        h = self.barrel_height
        bh = self.band_height
        bw = self.band_width

        base_z = oz
        top_z = oz + h

        # Wall thickness — inner radius is the hollow interior
        inner_r = max(4, self._snap_coord(r * 0.4))

        # ---- Body: 5 ring sections with sinusoidal bulge ----
        mid_r = r
        end_r = max(inner_r + 4, self._snap_coord(r * (2.0 - self.bulge_ratio)))
        num_sections = 5
        section_h = self._snap_coord(h / num_sections)

        for i in range(num_sections):
            sz1 = base_z + i * section_h
            sz2 = base_z + (i + 1) * section_h if i < num_sections - 1 else top_z
            sec_r = self._snap_coord(
                end_r + (mid_r - end_r) * math.sin(math.pi * (i + 0.5) / num_sections)
            )
            sec_r = max(inner_r + 4, sec_r)
            brushes.extend(_radial_ring(
                self, ox, oy, sz1, sz2, inner_r, sec_r, self.sides,
            ))

        # ---- Bands: evenly spaced hoops from bottom edge to top edge ----
        if self.band_count > 0:
            half_bh = self._snap_coord(bh / 2)
            for bi in range(self.band_count):
                # Fraction along body: 0.0 = bottom edge, 1.0 = top edge
                if self.band_count == 1:
                    frac = 0.5
                else:
                    frac = bi / (self.band_count - 1)

                # Band center Z, clamped so band stays within body
                band_cz = self._snap_coord(base_z + h * frac)
                bz1 = max(base_z, band_cz - half_bh)
                bz2 = min(top_z, bz1 + bh)

                # Body radius at this height + radial overhang
                t_norm = frac  # 0..1 through the body
                body_r_here = self._snap_coord(
                    end_r + (mid_r - end_r) * math.sin(math.pi * t_norm)
                )
                body_r_here = max(inner_r + 4, body_r_here)
                band_r = body_r_here + bw

                brushes.extend(_radial_ring(
                    self, ox, oy, bz1, bz2, inner_r, band_r, self.sides,
                ))

        # ---- Caps: solid polygonal disks at top and bottom ----
        # Use full barrel radius so the cap completely fills the interior;
        # any excess overlaps harmlessly into the wall rings.
        cap_r = r
        cap_h = max(1.0, bh)

        brushes.extend(self._generate_polygonal_solid(
            ox, oy, base_z, base_z + cap_h,
            cap_r, self.sides,
            texture=self.texture_structural,
        ))
        brushes.extend(self._generate_polygonal_solid(
            ox, oy, top_z - cap_h, top_z,
            cap_r, self.sides,
            texture=self.texture_structural,
        ))

        return brushes


# ---------------------------------------------------------------------------
# Tier 2 — Moderate complexity (3-10 brushes, mixed helpers)
# ---------------------------------------------------------------------------

class Pedestal(GeometricPrimitive):
    """Display platform: base + column shaft + top platform."""

    column_radius: float = 12.0
    total_height: float = 48.0
    platform_radius: float = 16.0
    sides: int = 8
    has_base: bool = True
    base_radius: float = 20.0
    base_height: float = 8.0
    platform_height: float = 8.0
    transition_height: float = 8.0
    shape_type: str = "round"

    @classmethod
    def get_display_name(cls) -> str:
        return "Pedestal"

    @classmethod
    def get_category(cls) -> str:
        return "Structural"

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "column_radius": {
                "type": "float", "default": 12.0, "min": 8, "max": 24,
                "label": "Column Radius", "description": "Radius of the central shaft"
            },
            "total_height": {
                "type": "float", "default": 48.0, "min": 32, "max": 80,
                "label": "Height", "description": "Total height of the pedestal"
            },
            "platform_radius": {
                "type": "float", "default": 16.0, "min": 12, "max": 32,
                "label": "Platform Radius", "description": "Radius of the top display platform"
            },
            "sides": {
                "type": "int", "default": 8, "min": 3, "max": 12,
                "label": "Sides",
                "description": "Facets for round mode (4=square, 8=octagonal, 12=round)"
            },
            "has_base": {
                "type": "bool", "default": True, "label": "Has Base",
                "description": "Add a wider base at the bottom"
            },
            "base_radius": {
                "type": "float", "default": 20.0, "min": 16, "max": 32,
                "label": "Base Radius", "description": "Radius of the bottom base"
            },
            "base_height": {
                "type": "float", "default": 8.0, "min": 4, "max": 16,
                "label": "Base Height", "description": "Height of the base slab"
            },
            "platform_height": {
                "type": "float", "default": 8.0, "min": 4, "max": 16,
                "label": "Platform Height", "description": "Height of the top display platform"
            },
            "transition_height": {
                "type": "float", "default": 8.0, "min": 4, "max": 16,
                "label": "Transition Height", "description": "Height of transition rings between sections"
            },
            "shape_type": {
                "type": "choice", "default": "round",
                "choices": ["square", "round"],
                "label": "Shape", "description": "square=box sections, round=cylindrical"
            },
        }

    def generate(self) -> List[Brush]:
        self._reset_tags()
        ox, oy, oz = self.params.origin
        brushes: List[Brush] = []
        base_h = self.base_height
        plat_h = self.platform_height
        trans_h = self.transition_height

        cur_z = oz

        if self.shape_type == "square":
            # Square box sections
            if self.has_base:
                br = self.base_radius
                brushes.append(self._structural_box(
                    ox - br, oy - br, cur_z,
                    ox + br, oy + br, cur_z + base_h,
                ))
                cur_z += base_h

                # Transition ring: base to shaft
                trans_r = self._snap_coord((self.base_radius + self.column_radius) / 2)
                brushes.append(self._structural_box(
                    ox - trans_r, oy - trans_r, cur_z,
                    ox + trans_r, oy + trans_r, cur_z + trans_h,
                ))
                cur_z += trans_h

            shaft_top = oz + self.total_height - plat_h
            if shaft_top <= cur_z:
                shaft_top = cur_z + 8.0

            cr = self.column_radius
            brushes.append(self._structural_box(
                ox - cr, oy - cr, cur_z,
                ox + cr, oy + cr, shaft_top,
            ))

            # Transition ring: shaft to platform
            trans_r2 = self._snap_coord((self.column_radius + self.platform_radius) / 2)
            brushes.append(self._structural_box(
                ox - trans_r2, oy - trans_r2, shaft_top - trans_h,
                ox + trans_r2, oy + trans_r2, shaft_top,
            ))

            pr = self.platform_radius
            brushes.append(self._structural_box(
                ox - pr, oy - pr, shaft_top,
                ox + pr, oy + pr, shaft_top + plat_h,
            ))
        else:
            # Round sections (default) — true faceted cylinders
            if self.has_base:
                brushes.extend(_radial_disk(
                    self, ox, oy, cur_z, cur_z + base_h,
                    self.base_radius, self.sides,
                ))
                cur_z += base_h

                # Transition ring: base to shaft
                trans_r = self._snap_coord((self.base_radius + self.column_radius) / 2)
                brushes.extend(_radial_disk(
                    self, ox, oy, cur_z, cur_z + trans_h,
                    trans_r, self.sides,
                ))
                cur_z += trans_h

            shaft_top = oz + self.total_height - plat_h
            if shaft_top <= cur_z:
                shaft_top = cur_z + 8.0
            brushes.extend(_radial_disk(
                self, ox, oy, cur_z, shaft_top,
                self.column_radius, self.sides,
            ))

            # Transition ring: shaft to platform
            trans_r2 = self._snap_coord((self.column_radius + self.platform_radius) / 2)
            brushes.extend(_radial_disk(
                self, ox, oy, shaft_top - trans_h, shaft_top,
                trans_r2, self.sides,
            ))

            brushes.extend(_radial_disk(
                self, ox, oy, shaft_top, shaft_top + plat_h,
                self.platform_radius, self.sides,
            ))

        return brushes


class Brazier(GeometricPrimitive):
    """Fire bowl with configurable stand style.

    Uses _radial_segment (pie-slice brushes) for round mode so that 'sides'
    directly controls visible facets.  Square mode uses _structural_box.
    """

    bowl_radius: float = 20.0
    bowl_height: float = 16.0
    stem_radius: float = 8.0
    stem_height: float = 32.0
    base_radius: float = 16.0
    sides: int = 8
    style: str = "pedestal"
    shape_type: str = "round"
    base_height: float = 8.0
    collar_overhang: float = 4.0
    collar_height: float = 8.0
    leg_width: float = 8.0
    chain_width: float = 8.0
    link_height: float = 12.0
    link_gap: float = 4.0

    @classmethod
    def get_display_name(cls) -> str:
        return "Brazier"

    @classmethod
    def get_category(cls) -> str:
        return "Structural"

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "bowl_radius": {
                "type": "float", "default": 20.0, "min": 12, "max": 32,
                "label": "Bowl Radius", "description": "Radius of the fire bowl"
            },
            "bowl_height": {
                "type": "float", "default": 16.0, "min": 8, "max": 24,
                "label": "Bowl Height", "description": "Height of the fire bowl"
            },
            "stem_radius": {
                "type": "float", "default": 8.0, "min": 8, "max": 16,
                "label": "Stem Radius", "description": "Radius of the stem"
            },
            "stem_height": {
                "type": "float", "default": 32.0, "min": 16, "max": 48,
                "label": "Stem Height", "description": "Height of the stem"
            },
            "base_radius": {
                "type": "float", "default": 16.0, "min": 12, "max": 24,
                "label": "Base Radius", "description": "Radius of the base plate"
            },
            "sides": {
                "type": "int", "default": 8, "min": 4, "max": 12,
                "label": "Sides",
                "description": "Facets for round mode (4=square, 8=octagonal, 12=round)"
            },
            "style": {
                "type": "choice", "default": "pedestal",
                "choices": ["pedestal", "tripod", "hanging"],
                "label": "Style",
                "description": "pedestal=base+stem, tripod=3 legs, hanging=suspended bowl"
            },
            "shape_type": {
                "type": "choice", "default": "round",
                "choices": ["round", "square"],
                "label": "Shape",
                "description": "round=cylindrical (uses sides), square=boxy"
            },
            "base_height": {
                "type": "float", "default": 8.0, "min": 4, "max": 16,
                "label": "Base Height", "description": "Height of the base plate"
            },
            "collar_overhang": {
                "type": "float", "default": 4.0, "min": 2, "max": 8,
                "label": "Collar Overhang", "description": "How far the collar extends beyond the stem"
            },
            "collar_height": {
                "type": "float", "default": 8.0, "min": 4, "max": 16,
                "label": "Collar Height", "description": "Vertical thickness of the collar ring"
            },
            "leg_width": {
                "type": "float", "default": 8.0, "min": 4, "max": 12,
                "label": "Leg Width", "description": "Width of tripod legs"
            },
            "chain_width": {
                "type": "float", "default": 8.0, "min": 4, "max": 12,
                "label": "Chain Width", "description": "Width of hanging chain links"
            },
            "link_height": {
                "type": "float", "default": 12.0, "min": 8, "max": 16,
                "label": "Link Height", "description": "Height of each chain link"
            },
            "link_gap": {
                "type": "float", "default": 4.0, "min": 2, "max": 8,
                "label": "Link Gap", "description": "Vertical gap between chain links"
            },
        }

    # -- helpers for radial-segment rings (same pattern as Barrel) ----------

    def _round_section(self, cx: float, cy: float, z1: float, z2: float,
                       radius: float) -> List[Brush]:
        """Cylindrical section using shared radial helpers."""
        inner_r = max(4, self._snap_coord(radius * 0.35))
        return _radial_ring(self, cx, cy, z1, z2, inner_r, radius, self.sides)

    def _square_section(self, cx: float, cy: float, z1: float, z2: float,
                        radius: float) -> List[Brush]:
        """Square box section centered on (cx, cy)."""
        r = self._snap_coord(radius)
        return [self._structural_box(cx - r, cy - r, z1, cx + r, cy + r, z2)]

    def _section(self, cx: float, cy: float, z1: float, z2: float,
                 radius: float) -> List[Brush]:
        """Dispatch to round or square based on shape_type."""
        if self.shape_type == "round":
            return self._round_section(cx, cy, z1, z2, radius)
        return self._square_section(cx, cy, z1, z2, radius)

    # -- main generate ------------------------------------------------------

    def generate(self) -> List[Brush]:
        self._reset_tags()
        ox, oy, oz = self.params.origin
        brushes: List[Brush] = []
        base_h = self.base_height

        if self.style == "pedestal":
            # Base plate (wider, flat)
            brushes.extend(self._section(
                ox, oy, oz, oz + base_h, self.base_radius,
            ))
            # Stem (narrow column)
            stem_bottom = oz + base_h
            stem_top = stem_bottom + self.stem_height
            brushes.extend(self._section(
                ox, oy, stem_bottom, stem_top, self.stem_radius,
            ))
            # Collar ring at top of stem (just below bowl)
            collar_r = self.stem_radius + self.collar_overhang
            collar_h = self.collar_height
            brushes.extend(self._section(
                ox, oy, stem_top - collar_h, stem_top, collar_r,
            ))
            # Hollow bowl (ring instead of solid disk)
            if self.shape_type == "round":
                bowl_inner = max(4, self._snap_coord(self.bowl_radius * 0.6))
                brushes.extend(_radial_ring(
                    self, ox, oy, stem_top, stem_top + self.bowl_height,
                    bowl_inner, self.bowl_radius, self.sides,
                ))
            else:
                # Square hollow bowl approximation: outer box minus would be CSG, so just use ring
                brushes.extend(self._section(
                    ox, oy, stem_top, stem_top + self.bowl_height,
                    self.bowl_radius,
                ))

        elif self.style == "tripod":
            # 3 individual leg boxes — each a thin tilted column from foot to
            # bowl base.  We create each leg as a box aligned to the foot-to-
            # center vector rather than a bounding-box rectangle.
            leg_w = int(self.leg_width)
            hlw = leg_w // 2
            bowl_z = oz + self.stem_height
            leg_spread = self._snap_coord(self.base_radius * 0.8)

            for i in range(3):
                angle = i * (2 * math.pi / 3)
                foot_x = self._snap_coord(ox + leg_spread * math.cos(angle))
                foot_y = self._snap_coord(oy + leg_spread * math.sin(angle))
                # Each leg is a narrow vertical box at the foot position
                # extending up to bowl height — produces 3 clearly separate
                # stilts rather than one oversized bounding rectangle.
                brushes.append(self._structural_box(
                    foot_x - hlw, foot_y - hlw, oz,
                    foot_x + hlw, foot_y + hlw, bowl_z,
                ))

            # Collar ring below bowl
            collar_r = self.stem_radius + self.collar_overhang
            collar_h = self.collar_height
            brushes.extend(self._section(
                ox, oy, bowl_z - collar_h, bowl_z, collar_r,
            ))
            # Hollow bowl on top
            if self.shape_type == "round":
                bowl_inner = max(4, self._snap_coord(self.bowl_radius * 0.6))
                brushes.extend(_radial_ring(
                    self, ox, oy, bowl_z, bowl_z + self.bowl_height,
                    bowl_inner, self.bowl_radius, self.sides,
                ))
            else:
                brushes.extend(self._section(
                    ox, oy, bowl_z, bowl_z + self.bowl_height,
                    self.bowl_radius,
                ))

        elif self.style == "hanging":
            # Bowl at bottom
            brushes.extend(self._section(
                ox, oy, oz, oz + self.bowl_height,
                self.bowl_radius,
            ))
            # Chain links above bowl (3 small square links — chains are always
            # boxy regardless of shape_type)
            chain_w = int(self.chain_width)
            hcw = chain_w // 2
            link_h = int(self.link_height)
            link_gap = int(self.link_gap)
            chain_z = oz + self.bowl_height
            for _ in range(3):
                brushes.append(self._structural_box(
                    ox - hcw, oy - hcw, chain_z,
                    ox + hcw, oy + hcw, chain_z + link_h,
                ))
                chain_z += link_h + link_gap
            # Horizontal crossbar at top
            bar_w = self._snap_coord(self.bowl_radius * 1.5)
            brushes.append(self._structural_box(
                ox - bar_w, oy - hcw, chain_z,
                ox + bar_w, oy + hcw, chain_z + chain_w,
            ))

        return brushes


class Throne(GeometricPrimitive):
    """A seat with backrest and optional armrests/platform.

    Styles:
      blocky    — flat box backrest + flat crown (original)
      ornate    — 3-section gradual taper (100% -> 80% -> 60%) + flat crown
      gothic    — full-width backrest + peaked pointed cap + optional finials
      high_back — 3-section taper + peaked cap on top
    """

    seat_width: float = 48.0
    seat_depth: float = 32.0
    seat_height: float = 24.0
    back_height: float = 64.0
    has_armrests: bool = False
    has_platform: bool = False
    has_visible_legs: bool = True
    style: str = "blocky"
    back_thickness: float = 8.0
    arm_width: float = 8.0
    platform_height: float = 8.0
    platform_overhang: float = 16.0
    platform_steps: int = 1
    leg_width: float = 8.0
    crown_overhang: float = 4.0
    crown_height: float = 8.0
    has_finials: bool = False
    finial_width: float = 8.0
    finial_height: float = 16.0

    @classmethod
    def get_display_name(cls) -> str:
        return "Throne"

    @classmethod
    def get_category(cls) -> str:
        return "Structural"

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "seat_width": {
                "type": "float", "default": 48.0, "min": 32, "max": 64,
                "label": "Seat Width", "description": "Width of the seating area"
            },
            "seat_depth": {
                "type": "float", "default": 32.0, "min": 24, "max": 48,
                "label": "Seat Depth", "description": "Depth of the seating area"
            },
            "seat_height": {
                "type": "float", "default": 24.0, "min": 16, "max": 32,
                "label": "Seat Height", "description": "Height of the seat from floor/platform"
            },
            "back_height": {
                "type": "float", "default": 64.0, "min": 48, "max": 96,
                "label": "Back Height", "description": "Total height of the backrest"
            },
            "has_armrests": {
                "type": "bool", "default": False, "label": "Armrests",
                "description": "Add armrests on both sides"
            },
            "has_platform": {
                "type": "bool", "default": False, "label": "Platform",
                "description": "Add a raised platform/dais beneath"
            },
            "has_visible_legs": {
                "type": "bool", "default": True, "label": "Visible Legs",
                "description": "Split seat into legs + thin slab (shows space under seat)"
            },
            "style": {
                "type": "choice", "default": "blocky",
                "choices": ["blocky", "ornate", "gothic", "high_back"],
                "label": "Style",
                "description": "blocky=simple, ornate=tapered, gothic=peaked, high_back=taper+peak"
            },
            "back_thickness": {
                "type": "float", "default": 8.0, "min": 4, "max": 16,
                "label": "Back Thickness", "description": "Thickness of the backrest slab"
            },
            "arm_width": {
                "type": "float", "default": 8.0, "min": 4, "max": 12,
                "label": "Arm Width", "description": "Width of each armrest"
            },
            "platform_height": {
                "type": "float", "default": 8.0, "min": 4, "max": 16,
                "label": "Platform Height", "description": "Total height of the platform/dais"
            },
            "platform_overhang": {
                "type": "float", "default": 16.0, "min": 8, "max": 24,
                "label": "Platform Overhang", "description": "How far the platform extends beyond the seat"
            },
            "platform_steps": {
                "type": "int", "default": 1, "min": 1, "max": 3,
                "label": "Platform Steps", "description": "Number of concentric platform steps"
            },
            "leg_width": {
                "type": "float", "default": 8.0, "min": 4, "max": 12,
                "label": "Leg Width", "description": "Width of each leg"
            },
            "crown_overhang": {
                "type": "float", "default": 4.0, "min": 2, "max": 8,
                "label": "Crown Overhang", "description": "How far the crown extends beyond the backrest"
            },
            "crown_height": {
                "type": "float", "default": 8.0, "min": 4, "max": 16,
                "label": "Crown Height", "description": "Height of the backrest crown piece"
            },
            "has_finials": {
                "type": "bool", "default": False, "label": "Finials",
                "description": "Add peaked finials at backrest top corners"
            },
            "finial_width": {
                "type": "float", "default": 8.0, "min": 4, "max": 12,
                "label": "Finial Width", "description": "Width of each finial base"
            },
            "finial_height": {
                "type": "float", "default": 16.0, "min": 8, "max": 24,
                "label": "Finial Height", "description": "Height of the finial peak"
            },
        }

    # ------------------------------------------------------------------
    # Helper: pointed backrest cap (pentahedral prism, ridge along X)
    # ------------------------------------------------------------------
    def _pointed_backrest_cap(self, ox: float, back_y1: float,
                              back_y2: float, z_base: float,
                              hw: float, peak_h: float,
                              tex: str) -> Brush:
        """Pentahedral prism with ridge along X, triangular cross-section in YZ.

        Used for Gothic/high_back peaked backrest tops. The ridge runs
        left-right (X axis) at the midpoint of the backrest thickness.
        """
        hw = self._snap_coord(hw)
        y1 = self._snap_coord(back_y1)
        y2 = self._snap_coord(back_y2)
        z_base = self._snap_coord(z_base)
        peak_h = self._snap_coord(peak_h)
        z_peak = z_base + peak_h
        y_mid = self._snap_coord((y1 + y2) / 2)

        planes = [
            # Bottom (normal +Z inward — faces down)
            Plane((ox - hw, y1, z_base),
                  (ox + hw, y1, z_base),
                  (ox - hw, y2, z_base), tex),
            # Left -X (normal +X inward)
            Plane((ox - hw, y1, z_base),
                  (ox - hw, y2, z_base),
                  (ox - hw, y1, z_peak), tex),
            # Right +X (normal -X inward)
            Plane((ox + hw, y1, z_base),
                  (ox + hw, y1, z_peak),
                  (ox + hw, y2, z_base), tex),
            # Front slope (normal inward: -Y, -Z)
            Plane((ox - hw, y1, z_base),
                  (ox - hw, y_mid, z_peak),
                  (ox + hw, y1, z_base), tex),
            # Back slope (normal inward: +Y, -Z)
            Plane((ox - hw, y2, z_base),
                  (ox + hw, y2, z_base),
                  (ox - hw, y_mid, z_peak), tex),
        ]
        return Brush(planes=planes, brush_id=self._next_id())

    # ------------------------------------------------------------------
    # Helper: pyramid (square base, 4 triangular faces converging to apex)
    # ------------------------------------------------------------------
    def _pyramid(self, cx: float, cy: float, z_base: float,
                 half_w: float, half_d: float, peak_h: float,
                 tex: str) -> Brush:
        """5-plane pyramid: square base + 4 slope faces to apex."""
        cx = self._snap_coord(cx)
        cy = self._snap_coord(cy)
        z_base = self._snap_coord(z_base)
        half_w = self._snap_coord(half_w)
        half_d = self._snap_coord(half_d)
        peak_h = self._snap_coord(peak_h)
        z_peak = z_base + peak_h

        planes = [
            # Bottom (normal +Z inward — faces down)
            Plane((cx - half_w, cy - half_d, z_base),
                  (cx + half_w, cy - half_d, z_base),
                  (cx - half_w, cy + half_d, z_base), tex),
            # Front -Y slope (normal inward: -Y, -Z)
            Plane((cx - half_w, cy - half_d, z_base),
                  (cx, cy, z_peak),
                  (cx + half_w, cy - half_d, z_base), tex),
            # Back +Y slope (normal inward: +Y, -Z)
            Plane((cx + half_w, cy + half_d, z_base),
                  (cx, cy, z_peak),
                  (cx - half_w, cy + half_d, z_base), tex),
            # Left -X slope (normal inward: +X, -Z)
            Plane((cx - half_w, cy + half_d, z_base),
                  (cx, cy, z_peak),
                  (cx - half_w, cy - half_d, z_base), tex),
            # Right +X slope (normal inward: -X, -Z)
            Plane((cx + half_w, cy - half_d, z_base),
                  (cx, cy, z_peak),
                  (cx + half_w, cy + half_d, z_base), tex),
        ]
        return Brush(planes=planes, brush_id=self._next_id())

    def generate(self) -> List[Brush]:
        self._reset_tags()
        ox, oy, oz = self.params.origin
        brushes: List[Brush] = []
        hw = self.seat_width / 2
        hd = self.seat_depth / 2
        back_thick = self.back_thickness
        arm_w = self.arm_width
        tex = self.params.texture

        base_z = oz

        # ==============================================
        # Platform/dais — stepped concentric rings
        # ==============================================
        if self.has_platform:
            plat_h = self.platform_height
            plat_extra = self.platform_overhang
            steps = max(1, min(3, self.platform_steps))
            step_h = self._snap_coord(plat_h / steps)
            for i in range(steps):
                # Step 0 = bottom/widest, step N-1 = top/narrowest
                frac = (steps - i) / steps  # 1.0 -> 1/steps
                overhang = self._snap_coord(plat_extra * frac)
                brushes.append(self._structural_box(
                    ox - hw - overhang, oy - hd - overhang, base_z,
                    ox + hw + overhang, oy + hd + overhang, base_z + step_h,
                ))
                base_z += step_h

        # ==============================================
        # Seat block (or legs + thin slab)
        # ==============================================
        if self.has_visible_legs:
            leg_size = self.leg_width
            leg_top = base_z + self.seat_height - leg_size
            for sx, sy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
                lx = ox + sx * (hw - leg_size / 2)
                ly = oy + sy * (hd - leg_size / 2)
                brushes.append(self._structural_box(
                    lx - leg_size / 2, ly - leg_size / 2, base_z,
                    lx + leg_size / 2, ly + leg_size / 2, leg_top,
                ))
            brushes.append(self._structural_box(
                ox - hw, oy - hd, leg_top,
                ox + hw, oy + hd, base_z + self.seat_height,
            ))
        else:
            brushes.append(self._structural_box(
                ox - hw, oy - hd, base_z,
                ox + hw, oy + hd, base_z + self.seat_height,
            ))

        # ==============================================
        # Backrest — style-specific
        # ==============================================
        back_base = base_z
        backrest_top = back_base + self.back_height
        # effective_hw tracks the width of the topmost backrest section
        # (needed for crown/cap and finial positioning)
        effective_hw = hw

        if self.style == "ornate" or self.style == "high_back":
            # 3-section gradual taper: 100% -> 80% -> 60%
            section_h = self._snap_coord(self.back_height / 3)
            taper_hw_mid = self._snap_coord(hw * 0.8)
            taper_hw_top = self._snap_coord(hw * 0.6)
            # Lower third — full width
            brushes.append(self._structural_box(
                ox - hw, oy + hd, back_base,
                ox + hw, oy + hd + back_thick, back_base + section_h,
            ))
            # Middle third — 80% width
            brushes.append(self._structural_box(
                ox - taper_hw_mid, oy + hd, back_base + section_h,
                ox + taper_hw_mid, oy + hd + back_thick, back_base + 2 * section_h,
            ))
            # Upper third — 60% width
            brushes.append(self._structural_box(
                ox - taper_hw_top, oy + hd, back_base + 2 * section_h,
                ox + taper_hw_top, oy + hd + back_thick, backrest_top,
            ))
            effective_hw = taper_hw_top
        elif self.style == "gothic":
            # Gothic: full-width single slab
            brushes.append(self._structural_box(
                ox - hw, oy + hd, back_base,
                ox + hw, oy + hd + back_thick, backrest_top,
            ))
            effective_hw = hw
        else:
            # Blocky: single slab
            brushes.append(self._structural_box(
                ox - hw, oy + hd, back_base,
                ox + hw, oy + hd + back_thick, backrest_top,
            ))
            effective_hw = hw

        # ==============================================
        # Backrest crown / cap
        # ==============================================
        crown_extra = self.crown_overhang
        crown_h = self.crown_height

        if self.style in ("gothic", "high_back"):
            # Peaked pointed cap (pentahedral prism) instead of flat crown
            brushes.append(self._pointed_backrest_cap(
                ox,
                oy + hd,
                oy + hd + back_thick,
                backrest_top,
                effective_hw + crown_extra,
                crown_h,
                tex,
            ))
            crown_top_z = backrest_top + crown_h
        else:
            # Flat crown box (blocky / ornate)
            crown_hw = effective_hw + crown_extra
            brushes.append(self._structural_box(
                ox - crown_hw, oy + hd, backrest_top,
                ox + crown_hw, oy + hd + back_thick, backrest_top + crown_h,
            ))
            crown_top_z = backrest_top + crown_h

        # ==============================================
        # Armrests — style-specific
        # ==============================================
        if self.has_armrests:
            arm_top_z = base_z + self.seat_height + 16.0  # slightly above seat
            for sx in [-1, 1]:
                arm_x = ox + sx * (hw - arm_w / 2)

                if self.style in ("gothic", "high_back"):
                    # Angled armrests: box post + wedge ramp on top
                    # Post from base up to arm_front_z (lower front height)
                    arm_front_z = base_z + self.seat_height + 8.0
                    arm_back_z = arm_top_z
                    brushes.append(self._structural_box(
                        arm_x - arm_w / 2, oy - hd, base_z,
                        arm_x + arm_w / 2, oy + hd, arm_front_z,
                    ))
                    # Wedge slopes from front (low) to back (high)
                    brushes.append(self._wedge(
                        arm_x - arm_w / 2, oy - hd, arm_front_z,
                        arm_x + arm_w / 2, oy + hd, arm_back_z,
                        ramp_axis="y",
                    ))
                else:
                    # Blocky/ornate: box post + flat cap
                    brushes.append(self._structural_box(
                        arm_x - arm_w / 2, oy - hd, base_z,
                        arm_x + arm_w / 2, oy + hd, base_z + arm_top_z - base_z,
                    ))
                    cap_extra = self.crown_overhang
                    cap_h = self.crown_height
                    brushes.append(self._structural_box(
                        arm_x - arm_w / 2 - cap_extra, oy - hd, arm_top_z,
                        arm_x + arm_w / 2 + cap_extra, oy + hd, arm_top_z + cap_h,
                    ))

        # ==============================================
        # Finials — peaked pyramids at backrest top corners
        # ==============================================
        if self.has_finials:
            fin_hw = self._snap_coord(self.finial_width / 2)
            fin_hd = fin_hw  # square base
            fin_h = self.finial_height
            # Post height: small box pedestal under each pyramid
            post_h = self._snap_coord(fin_hw)
            for sx in [-1, 1]:
                fin_cx = ox + sx * effective_hw
                fin_cy = self._snap_coord(oy + hd + back_thick / 2)
                # Small post at backrest top
                brushes.append(self._structural_box(
                    fin_cx - fin_hw, fin_cy - fin_hd, backrest_top,
                    fin_cx + fin_hw, fin_cy + fin_hd, backrest_top + post_h,
                ))
                # Pyramid cap
                brushes.append(self._pyramid(
                    fin_cx, fin_cy,
                    backrest_top + post_h,
                    fin_hw, fin_hd, fin_h, tex,
                ))

        return brushes


class Altar(GeometricPrimitive):
    """Altar with configurable support style and optional polygonal shape."""

    altar_width: float = 64.0
    altar_depth: float = 32.0
    altar_height: float = 36.0
    pedestal_style: str = "solid"
    pedestal_sides: int = 4
    has_reredos: bool = False
    reredos_height: float = 64.0
    has_step: bool = False
    has_pinnacles: bool = False
    shape_type: str = "rectangular"
    shape_sides: int = 6
    top_thickness: float = 8.0
    step_height: float = 8.0
    step_depth: float = 16.0
    top_overhang: float = 4.0
    reredos_thickness: float = 8.0
    pinnacle_width: float = 8.0
    pinnacle_height: float = 16.0

    @classmethod
    def get_display_name(cls) -> str:
        return "Altar"

    @classmethod
    def get_category(cls) -> str:
        return "Structural"

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "altar_width": {
                "type": "float", "default": 64.0, "min": 48, "max": 128,
                "label": "Width", "description": "Width of the altar top slab"
            },
            "altar_depth": {
                "type": "float", "default": 32.0, "min": 24, "max": 48,
                "label": "Depth", "description": "Depth of the altar top slab"
            },
            "altar_height": {
                "type": "float", "default": 36.0, "min": 24, "max": 48,
                "label": "Height", "description": "Height from floor to top surface"
            },
            "pedestal_style": {
                "type": "choice", "default": "solid",
                "choices": ["solid", "twin_pillars", "slab"],
                "label": "Pedestal Style",
                "description": "solid=box, twin_pillars=two columns, slab=thin slab base"
            },
            "pedestal_sides": {
                "type": "int", "default": 4, "min": 3, "max": 12,
                "label": "Pedestal Sides",
                "description": "Facets for twin_pillars (4=square, 8=octagonal, 12=round)"
            },
            "has_reredos": {
                "type": "bool", "default": False, "label": "Reredos (Back Screen)",
                "description": "Add a tall decorative screen behind the altar"
            },
            "reredos_height": {
                "type": "float", "default": 64.0, "min": 48, "max": 96,
                "label": "Reredos Height", "description": "Height of the back screen"
            },
            "has_step": {
                "type": "bool", "default": False, "label": "Step",
                "description": "Add a step/platform in front of the altar"
            },
            "has_pinnacles": {
                "type": "bool", "default": False, "label": "Reredos Pinnacles",
                "description": "Add small vertical pinnacles at top corners of reredos"
            },
            "shape_type": {
                "type": "choice", "default": "rectangular",
                "choices": ["rectangular", "round"],
                "label": "Shape", "description": "rectangular=box, round=cylindrical"
            },
            "shape_sides": {
                "type": "int", "default": 6, "min": 3, "max": 12,
                "label": "Shape Sides",
                "description": "Facets for round mode (4=square, 8=octagonal, 12=round)"
            },
            "top_thickness": {
                "type": "float", "default": 8.0, "min": 4, "max": 16,
                "label": "Top Thickness", "description": "Thickness of the altar top slab"
            },
            "step_height": {
                "type": "float", "default": 8.0, "min": 4, "max": 16,
                "label": "Step Height", "description": "Height of the front step"
            },
            "step_depth": {
                "type": "float", "default": 16.0, "min": 8, "max": 24,
                "label": "Step Depth", "description": "Depth of the front step"
            },
            "top_overhang": {
                "type": "float", "default": 4.0, "min": 2, "max": 8,
                "label": "Top Overhang", "description": "How far the top slab extends beyond the pedestal"
            },
            "reredos_thickness": {
                "type": "float", "default": 8.0, "min": 4, "max": 16,
                "label": "Reredos Thickness", "description": "Thickness of the back screen"
            },
            "pinnacle_width": {
                "type": "float", "default": 8.0, "min": 4, "max": 12,
                "label": "Pinnacle Width", "description": "Width of reredos pinnacles"
            },
            "pinnacle_height": {
                "type": "float", "default": 16.0, "min": 8, "max": 24,
                "label": "Pinnacle Height", "description": "Height of reredos pinnacles"
            },
        }

    def generate(self) -> List[Brush]:
        self._reset_tags()
        ox, oy, oz = self.params.origin
        brushes: List[Brush] = []
        hw = self.altar_width / 2
        hd = self.altar_depth / 2
        top_h = self.top_thickness  # Top slab thickness
        base_h = self.altar_height - top_h

        # Step in front
        if self.has_step:
            step_h = self.step_height
            step_depth = self.step_depth
            brushes.append(self._structural_box(
                ox - hw - step_h, oy - hd - step_depth, oz,
                ox + hw + step_h, oy - hd, oz + step_h,
            ))

        # Pedestal/support
        if self.shape_type == "round":
            radius = min(hw, hd)

            if self.pedestal_style == "solid":
                brushes.extend(_radial_disk(
                    self, ox, oy, oz, oz + base_h,
                    radius, self.shape_sides,
                ))
            elif self.pedestal_style == "twin_pillars":
                pillar_r = max(1.0, min(hd * 0.6, hw * 0.3))
                pillar_spacing = hw * 0.6
                for sx in [-1, 1]:
                    px = ox + sx * pillar_spacing
                    brushes.extend(_radial_disk(
                        self, px, oy, oz, oz + base_h,
                        pillar_r, self.pedestal_sides,
                    ))
            elif self.pedestal_style == "slab":
                slab_h = max(1.0, self._snap_coord(base_h * 0.3))
                brushes.extend(_radial_disk(
                    self, ox, oy, oz, oz + slab_h,
                    radius, self.shape_sides,
                ))

            # Top slab (round)
            brushes.extend(_radial_disk(
                self, ox, oy, oz + base_h, oz + self.altar_height,
                radius, self.shape_sides,
            ))
        else:
            # Rectangular (default)
            if self.pedestal_style == "solid":
                brushes.append(self._structural_box(
                    ox - hw, oy - hd, oz,
                    ox + hw, oy + hd, oz + base_h,
                ))
            elif self.pedestal_style == "twin_pillars":
                pillar_r = max(1.0, min(hd * 0.6, hw * 0.3))
                pillar_spacing = hw * 0.6
                for sx in [-1, 1]:
                    px = ox + sx * pillar_spacing
                    brushes.extend(_radial_disk(
                        self, px, oy, oz, oz + base_h,
                        pillar_r, self.pedestal_sides,
                    ))
            elif self.pedestal_style == "slab":
                slab_h = max(1.0, self._snap_coord(base_h * 0.3))
                brushes.append(self._structural_box(
                    ox - hw, oy - hd, oz,
                    ox + hw, oy + hd, oz + slab_h,
                ))

            # Top slab (rectangular) — wider/deeper than pedestal for overhang
            top_extra = self.top_overhang
            brushes.append(self._structural_box(
                ox - hw - top_extra, oy - hd - top_extra, oz + base_h,
                ox + hw + top_extra, oy + hd + top_extra, oz + self.altar_height,
            ))

        # Reredos (back screen — stepped: 2 stacked boxes decreasing in width)
        if self.has_reredos:
            screen_thick = self.reredos_thickness
            mid_h = self._snap_coord(self.reredos_height * 0.6)
            upper_h = self.reredos_height - mid_h
            # Lower section — full width
            brushes.append(self._structural_box(
                ox - hw, oy + hd, oz,
                ox + hw, oy + hd + screen_thick, oz + mid_h,
            ))
            # Upper section — narrower
            taper_hw = self._snap_coord(hw * 0.7)
            brushes.append(self._structural_box(
                ox - taper_hw, oy + hd, oz + mid_h,
                ox + taper_hw, oy + hd + screen_thick, oz + self.reredos_height,
            ))

            # Pinnacles at top corners of reredos
            if self.has_pinnacles:
                pin_w = self.pinnacle_width
                pin_h = self.pinnacle_height
                reredos_top = oz + self.reredos_height
                for sx in [-1, 1]:
                    px = ox + sx * (hw - pin_w / 2)
                    brushes.append(self._structural_box(
                        px - pin_w / 2, oy + hd, reredos_top,
                        px + pin_w / 2, oy + hd + screen_thick, reredos_top + pin_h,
                    ))

        return brushes


class Bookshelf(GeometricPrimitive):
    """Back panel + side panels + shelves + optional individual books."""

    shelf_width: float = 64.0
    shelf_height: float = 96.0
    shelf_depth: float = 16.0
    shelf_count: int = 3
    has_books: bool = False
    has_molding: bool = True
    panel_thickness: float = 8.0
    shelf_thickness: float = 8.0
    book_width: float = 8.0
    molding_overhang: float = 4.0
    molding_height: float = 8.0
    random_seed: int = 0

    @classmethod
    def get_display_name(cls) -> str:
        return "Bookshelf"

    @classmethod
    def get_category(cls) -> str:
        return "Structural"

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "shelf_width": {
                "type": "float", "default": 64.0, "min": 32, "max": 96,
                "label": "Width", "description": "Total width of the bookshelf"
            },
            "shelf_height": {
                "type": "float", "default": 96.0, "min": 64, "max": 128,
                "label": "Height", "description": "Total height of the bookshelf"
            },
            "shelf_depth": {
                "type": "float", "default": 16.0, "min": 12, "max": 24,
                "label": "Depth", "description": "Front-to-back depth"
            },
            "shelf_count": {
                "type": "int", "default": 3, "min": 2, "max": 5,
                "label": "Shelves", "description": "Number of horizontal shelf boards"
            },
            "has_books": {
                "type": "bool", "default": False, "label": "Books",
                "description": "Add individual book blocks on shelves"
            },
            "has_molding": {
                "type": "bool", "default": True, "label": "Molding Trim",
                "description": "Add top cornice and base plinth trim"
            },
            "panel_thickness": {
                "type": "float", "default": 8.0, "min": 1, "max": 12,
                "label": "Panel Thickness", "description": "Thickness of side/back panels"
            },
            "shelf_thickness": {
                "type": "float", "default": 8.0, "min": 1, "max": 12,
                "label": "Shelf Thickness", "description": "Thickness of shelf boards"
            },
            "book_width": {
                "type": "float", "default": 8.0, "min": 4, "max": 12,
                "label": "Book Width", "description": "Width of each individual book"
            },
            "molding_overhang": {
                "type": "float", "default": 4.0, "min": 2, "max": 8,
                "label": "Molding Overhang", "description": "How far molding extends beyond the shelf body"
            },
            "molding_height": {
                "type": "float", "default": 8.0, "min": 4, "max": 16,
                "label": "Molding Height", "description": "Height of the cornice and plinth trim"
            },
            "random_seed": {
                "type": "int", "default": 0, "min": 0, "max": 999999,
                "label": "Random Seed", "description": "Seed for book variation"
            },
        }

    def generate(self) -> List[Brush]:
        self._reset_tags()
        ox, oy, oz = self.params.origin
        brushes: List[Brush] = []
        hw = self.shelf_width / 2
        pt = self.panel_thickness
        st = self.shelf_thickness

        # Back panel
        brushes.append(self._structural_box(
            ox - hw, oy + self.shelf_depth - pt, oz,
            ox + hw, oy + self.shelf_depth, oz + self.shelf_height,
        ))

        # Left side panel
        brushes.append(self._structural_box(
            ox - hw, oy, oz,
            ox - hw + pt, oy + self.shelf_depth, oz + self.shelf_height,
        ))

        # Right side panel
        brushes.append(self._structural_box(
            ox + hw - pt, oy, oz,
            ox + hw, oy + self.shelf_depth, oz + self.shelf_height,
        ))

        # Horizontal shelves (including top and bottom)
        shelf_spacing = self.shelf_height / (self.shelf_count + 1)
        inner_left = ox - hw + pt
        inner_right = ox + hw - pt

        for i in range(self.shelf_count + 2):  # bottom, N shelves, top
            if i == 0:
                sz = oz
            elif i == self.shelf_count + 1:
                sz = oz + self.shelf_height - st
            else:
                sz = oz + i * shelf_spacing - st / 2

            brushes.append(self._structural_box(
                inner_left, oy, sz,
                inner_right, oy + self.shelf_depth - pt, sz + st,
            ))

        # Individual books on shelves
        if self.has_books:
            rng = random.Random(self.random_seed)
            inner_w = self.shelf_width - 2 * pt
            book_w = self.book_width
            book_d = max(1.0, self.shelf_depth - pt - 4)

            for i in range(1, self.shelf_count + 1):
                shelf_top_z = oz + i * shelf_spacing - st / 2 + st
                next_shelf_z = oz + (i + 1) * shelf_spacing - st / 2
                if i == self.shelf_count:
                    next_shelf_z = oz + self.shelf_height - st
                shelf_gap = next_shelf_z - shelf_top_z

                # Generate 3-5 individual books per shelf
                num_books = rng.randint(3, 5)
                # Place books side by side with small gaps
                book_x = inner_left
                for _ in range(num_books):
                    if book_x + book_w > inner_right:
                        break
                    book_h = max(1.0, self._snap_coord(
                        rng.uniform(0.6, 0.9) * shelf_gap
                    ))
                    brushes.append(self._structural_box(
                        book_x, oy, shelf_top_z,
                        book_x + book_w, oy + book_d, shelf_top_z + book_h,
                    ))
                    book_x += book_w + self._snap_coord(rng.uniform(0, 4))

        # Molding trim (top cornice and base plinth)
        if self.has_molding:
            mold_extra = self.molding_overhang
            mold_h = self.molding_height
            # Top cornice
            brushes.append(self._structural_box(
                ox - hw - mold_extra, oy - mold_extra, oz + self.shelf_height,
                ox + hw + mold_extra, oy + self.shelf_depth + mold_extra, oz + self.shelf_height + mold_h,
            ))
            # Base plinth
            brushes.append(self._structural_box(
                ox - hw - mold_extra, oy - mold_extra, oz - mold_h,
                ox + hw + mold_extra, oy + self.shelf_depth + mold_extra, oz,
            ))

        return brushes


class WeaponRack(GeometricPrimitive):
    """Back panel with horizontal pegs and medieval weapon shapes."""

    rack_width: float = 64.0
    rack_height: float = 80.0
    peg_count: int = 3
    peg_depth: float = 16.0
    has_base: bool = False
    has_weapons: bool = False
    weapon_type: str = "mixed"
    panel_thickness: float = 8.0
    peg_width: float = 8.0
    peg_height: float = 8.0
    rail_height: float = 8.0
    base_height: float = 8.0
    random_seed: int = 0

    @classmethod
    def get_display_name(cls) -> str:
        return "Weapon Rack"

    @classmethod
    def get_category(cls) -> str:
        return "Structural"

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "rack_width": {
                "type": "float", "default": 64.0, "min": 32, "max": 96,
                "label": "Width", "description": "Width of the rack panel"
            },
            "rack_height": {
                "type": "float", "default": 80.0, "min": 48, "max": 112,
                "label": "Height", "description": "Height of the rack panel"
            },
            "peg_count": {
                "type": "int", "default": 3, "min": 2, "max": 4,
                "label": "Pegs", "description": "Number of horizontal peg rows"
            },
            "peg_depth": {
                "type": "float", "default": 16.0, "min": 8, "max": 24,
                "label": "Peg Depth", "description": "How far pegs extend from the panel"
            },
            "has_base": {
                "type": "bool", "default": False, "label": "Base",
                "description": "Add a floor base for freestanding placement"
            },
            "has_weapons": {
                "type": "bool", "default": False, "label": "Weapons",
                "description": "Add decorative weapon shapes on pegs"
            },
            "weapon_type": {
                "type": "choice", "default": "mixed",
                "choices": ["mixed", "swords", "axes", "spears"],
                "label": "Weapon Type",
                "description": "Type of weapons displayed on pegs"
            },
            "panel_thickness": {
                "type": "float", "default": 8.0, "min": 4, "max": 12,
                "label": "Panel Thickness", "description": "Thickness of the back panel"
            },
            "peg_width": {
                "type": "float", "default": 8.0, "min": 4, "max": 12,
                "label": "Peg Width", "description": "Width of each peg"
            },
            "peg_height": {
                "type": "float", "default": 8.0, "min": 4, "max": 12,
                "label": "Peg Height", "description": "Height of each peg"
            },
            "rail_height": {
                "type": "float", "default": 8.0, "min": 4, "max": 12,
                "label": "Rail Height", "description": "Height of the top rail"
            },
            "base_height": {
                "type": "float", "default": 8.0, "min": 4, "max": 12,
                "label": "Base Height", "description": "Height of the floor base"
            },
            "random_seed": {
                "type": "int", "default": 0, "min": 0, "max": 999999,
                "label": "Random Seed", "description": "Seed for weapon variation"
            },
        }

    def generate(self) -> List[Brush]:
        self._reset_tags()
        ox, oy, oz = self.params.origin
        brushes: List[Brush] = []
        hw = self.rack_width / 2
        panel_t = self.panel_thickness
        peg_w = self.peg_width
        peg_h = self.peg_height

        base_z = oz

        # Optional floor base
        if self.has_base:
            base_h = self.base_height
            brushes.append(self._structural_box(
                ox - hw - 4, oy, base_z,
                ox + hw + 4, oy + self.peg_depth + panel_t, base_z + base_h,
            ))
            base_z += base_h

        # Back panel
        brushes.append(self._structural_box(
            ox - hw, oy + self.peg_depth, base_z,
            ox + hw, oy + self.peg_depth + panel_t, base_z + self.rack_height,
        ))

        # Top rail
        rail_h = self.rail_height
        brushes.append(self._structural_box(
            ox - hw, oy, base_z + self.rack_height,
            ox + hw, oy + self.peg_depth + panel_t, base_z + self.rack_height + rail_h,
        ))

        # Pegs (two paired pegs per row at 1/3 and 2/3 width)
        peg_spacing = self.rack_height / (self.peg_count + 1)
        for i in range(1, self.peg_count + 1):
            peg_z = base_z + i * peg_spacing - peg_h / 2
            for peg_x_frac in [1 / 3, 2 / 3]:
                px = ox - hw + self.rack_width * peg_x_frac
                brushes.append(self._structural_box(
                    px - peg_w / 2, oy, peg_z,
                    px + peg_w / 2, oy + self.peg_depth, peg_z + peg_h,
                ))

        # Weapon shapes on pegs
        if self.has_weapons:
            rng = random.Random(self.random_seed)
            for i in range(1, self.peg_count + 1):
                peg_z = base_z + i * peg_spacing - peg_h / 2
                peg_top = peg_z + peg_h
                peg_cy = oy + self.peg_depth / 2

                if self.weapon_type == "mixed":
                    wtype = rng.choice(["sword", "axe", "spear"])
                else:
                    wtype = self.weapon_type[:-1] if self.weapon_type.endswith("s") else self.weapon_type

                wx = self._snap_coord(ox + rng.uniform(-hw * 0.2, hw * 0.2))

                if wtype == "sword":
                    # Blade: tall thin vertical box
                    blade_h = self._snap_coord(rng.uniform(32, 48))
                    brushes.append(self._structural_box(
                        wx - 4, peg_cy - 4, peg_top,
                        wx + 4, peg_cy + 4, peg_top + blade_h,
                    ))
                    # Crossguard: horizontal box at base of blade
                    brushes.append(self._structural_box(
                        wx - 12, peg_cy - 4, peg_top,
                        wx + 12, peg_cy + 4, peg_top + 8,
                    ))
                    # Handle: thin box below crossguard
                    brushes.append(self._structural_box(
                        wx - 4, peg_cy - 4, peg_z - 12,
                        wx + 4, peg_cy + 4, peg_z,
                    ))
                elif wtype == "axe":
                    # Handle: thin tall vertical box
                    handle_h = self._snap_coord(rng.uniform(32, 48))
                    brushes.append(self._structural_box(
                        wx - 4, peg_cy - 4, peg_top,
                        wx + 4, peg_cy + 4, peg_top + handle_h,
                    ))
                    # Axe head: wedge at top of handle
                    head_base = peg_top + handle_h - 16
                    brushes.append(self._wedge(
                        wx - 12, peg_cy - 4, head_base,
                        wx, peg_cy + 4, head_base + 16,
                        ramp_axis="x",
                    ))
                elif wtype == "spear":
                    # Shaft: very thin long vertical box
                    shaft_h = self._snap_coord(rng.uniform(48, 64))
                    brushes.append(self._structural_box(
                        wx - 4, peg_cy - 4, peg_z - 8,
                        wx + 4, peg_cy + 4, peg_z + shaft_h,
                    ))
                    # Spear tip: wedge at top
                    brushes.append(self._wedge(
                        wx - 4, peg_cy - 4, peg_z + shaft_h,
                        wx + 4, peg_cy + 4, peg_z + shaft_h + 12,
                        ramp_axis="y",
                    ))

        return brushes


class TorchSconce(GeometricPrimitive):
    """Torch holder with multiple style options."""

    bracket_width: float = 8.0
    arm_depth: float = 16.0
    mount_height: float = 64.0
    base_sides: int = 6
    has_torch: bool = False
    style: str = "bracket"
    cup_overhang: float = 4.0
    cup_height: float = 8.0
    base_height: float = 8.0

    @classmethod
    def get_display_name(cls) -> str:
        return "Torch Sconce"

    @classmethod
    def get_category(cls) -> str:
        return "Structural"

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "bracket_width": {
                "type": "float", "default": 8.0, "min": 8, "max": 16,
                "label": "Width", "description": "Width of the bracket/pole"
            },
            "arm_depth": {
                "type": "float", "default": 16.0, "min": 8, "max": 24,
                "label": "Arm Depth", "description": "How far the arm extends from wall/pole"
            },
            "mount_height": {
                "type": "float", "default": 64.0, "min": 48, "max": 96,
                "label": "Height", "description": "Height of the mount/total height"
            },
            "base_sides": {
                "type": "int", "default": 6, "min": 3, "max": 12,
                "label": "Base Sides",
                "description": "Facets for standing base/cup (4=square, 8=octagonal, 12=round)"
            },
            "has_torch": {
                "type": "bool", "default": False, "label": "Torch Stick",
                "description": "Add a visible torch stick rising from cup"
            },
            "style": {
                "type": "choice", "default": "bracket",
                "choices": ["bracket", "candelabra", "standing"],
                "label": "Style",
                "description": "bracket=wall-mount, candelabra=multi-arm, standing=floor stand"
            },
            "cup_overhang": {
                "type": "float", "default": 4.0, "min": 2, "max": 8,
                "label": "Cup Overhang", "description": "How far the cup extends beyond the bracket"
            },
            "cup_height": {
                "type": "float", "default": 8.0, "min": 4, "max": 12,
                "label": "Cup Height", "description": "Height of the torch cup"
            },
            "base_height": {
                "type": "float", "default": 8.0, "min": 4, "max": 12,
                "label": "Base Height", "description": "Height of the standing base"
            },
        }

    def generate(self) -> List[Brush]:
        self._reset_tags()
        ox, oy, oz = self.params.origin
        brushes: List[Brush] = []
        bw = self.bracket_width
        hbw = bw / 2

        if self.style == "bracket":
            tex = self.params.texture
            plate_depth = self._snap_coord(max(4, bw / 2))
            arm_y_start = self._snap_coord(oy + plate_depth)
            arm_y_end = self._snap_coord(oy + plate_depth + self.arm_depth)

            # Arm rises upward from wall to tip
            arm_rise = self._snap_coord(self.arm_depth / 2)
            arm_thick = self._snap_coord(max(4, bw / 2))
            z_wall = self._snap_coord(oz + self.mount_height)
            z_tip = self._snap_coord(z_wall + arm_rise)

            # 1. Small wall plate
            plate_hw = self._snap_coord(hbw)
            plate_h = self._snap_coord(max(12, bw * 1.5))
            plate_z_bot = self._snap_coord(z_wall - plate_h / 2)
            plate_z_top = self._snap_coord(z_wall + plate_h / 2)
            brushes.append(self._structural_box(
                ox - plate_hw, oy, plate_z_bot,
                ox + plate_hw, oy + plate_depth, plate_z_top,
            ))

            # 2. Arm — angled upward from plate to tip
            ahw = self._snap_coord(max(2, hbw / 2))
            zwb = self._snap_coord(z_wall - arm_thick / 2)
            zwt = self._snap_coord(z_wall + arm_thick / 2)
            ztb = self._snap_coord(z_tip - arm_thick / 2)
            ztt = self._snap_coord(z_tip + arm_thick / 2)
            ay1, ay2 = arm_y_start, arm_y_end
            ax1 = self._snap_coord(ox - ahw)
            ax2 = self._snap_coord(ox + ahw)
            brushes.append(Brush(planes=[
                Plane((ax1, ay1, zwb), (ax1, ay1, zwt),
                      (ax1, ay2, zwb), tex),
                Plane((ax2, ay1, zwb), (ax2, ay2, zwb),
                      (ax2, ay1, zwt), tex),
                Plane((ax1, ay1, zwb), (ax2, ay1, zwb),
                      (ax2, ay1, zwt), tex),
                Plane((ax2, ay2, ztb), (ax1, ay2, ztb),
                      (ax1, ay2, ztt), tex),
                Plane((ax1, ay1, zwb), (ax1, ay2, ztb),
                      (ax2, ay1, zwb), tex),
                Plane((ax1, ay1, zwt), (ax2, ay1, zwt),
                      (ax1, ay2, ztt), tex),
            ], brush_id=self._next_id()))

            # 3. Torch head — wider piece at arm tip
            head_hw = self._snap_coord(max(4, hbw))
            head_h = self._snap_coord(self.cup_height)
            brushes.append(self._structural_box(
                ox - head_hw, arm_y_end - head_hw, z_tip,
                ox + head_hw, arm_y_end + head_hw, z_tip + head_h,
            ))

            # 4. Torch stick rising from head
            if self.has_torch:
                torch_hw = self._snap_coord(max(2, ahw))
                torch_top = self._snap_coord(z_tip + head_h + 16)
                brushes.append(self._structural_box(
                    ox - torch_hw, arm_y_end - torch_hw, z_tip + head_h,
                    ox + torch_hw, arm_y_end + torch_hw, torch_top,
                ))

        elif self.style == "candelabra":
            # Central pole
            brushes.append(self._structural_box(
                ox - hbw, oy - hbw, oz,
                ox + hbw, oy + hbw, oz + self.mount_height,
            ))
            # 3 horizontal cross-arms at top
            arm_z = oz + self.mount_height - bw
            arm_len = self.arm_depth
            for i in range(3):
                angle = i * (2 * math.pi / 3)
                end_x = self._snap_coord(ox + arm_len * math.cos(angle))
                end_y = self._snap_coord(oy + arm_len * math.sin(angle))
                brushes.append(self._structural_box(
                    min(ox, end_x) - hbw, min(oy, end_y) - hbw, arm_z,
                    max(ox, end_x) + hbw, max(oy, end_y) + hbw, arm_z + bw,
                ))
                # Cup at each arm end
                cup_hw = (bw + self.cup_overhang) / 2
                brushes.append(self._structural_box(
                    end_x - cup_hw, end_y - cup_hw, arm_z + bw,
                    end_x + cup_hw, end_y + cup_hw, arm_z + bw + self.cup_height,
                ))

        elif self.style == "standing":
            # Wide round base
            base_h = self.base_height
            base_r = max(12.0, self.arm_depth)
            brushes.extend(_radial_disk(
                self, ox, oy, oz, oz + base_h,
                base_r, self.base_sides,
            ))
            # Thin tall pole
            brushes.append(self._structural_box(
                ox - hbw, oy - hbw, oz + base_h,
                ox + hbw, oy + hbw, oz + self.mount_height,
            ))
            # Cup at top
            cup_r = max(1.0, bw + self.cup_overhang)
            cup_h = self.cup_height
            brushes.extend(_radial_disk(
                self, ox, oy, oz + self.mount_height, oz + self.mount_height + cup_h,
                cup_r, self.base_sides,
            ))

        return brushes


class Fountain(GeometricPrimitive):
    """Circular basin with optional central column."""

    basin_radius: float = 32.0
    basin_height: float = 24.0
    wall_thickness: float = 8.0
    segments: int = 8
    has_column: bool = True
    has_basin_floor: bool = True
    has_column_cap: bool = True
    column_radius: float = 8.0
    column_height: float = 48.0
    column_sides: int = 8
    column_shape: str = "round"
    base_overhang: float = 4.0
    base_height: float = 8.0
    floor_thickness: float = 8.0
    cap_overhang: float = 4.0
    cap_height: float = 8.0

    @classmethod
    def get_display_name(cls) -> str:
        return "Fountain"

    @classmethod
    def get_category(cls) -> str:
        return "Structural"

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "basin_radius": {
                "type": "float", "default": 32.0, "min": 24, "max": 48,
                "label": "Basin Radius", "description": "Outer radius of the basin"
            },
            "basin_height": {
                "type": "float", "default": 24.0, "min": 16, "max": 32,
                "label": "Basin Height", "description": "Height of the basin wall"
            },
            "wall_thickness": {
                "type": "float", "default": 8.0, "min": 8, "max": 16,
                "label": "Wall Thickness", "description": "Thickness of the basin wall"
            },
            "segments": {
                "type": "int", "default": 8, "min": 4, "max": 12,
                "label": "Segments", "description": "Number of radial segments for the basin ring"
            },
            "has_column": {
                "type": "bool", "default": True, "label": "Central Column",
                "description": "Add a column rising from center of the basin"
            },
            "column_radius": {
                "type": "float", "default": 8.0, "min": 8, "max": 16,
                "label": "Column Radius", "description": "Radius of the central column"
            },
            "column_height": {
                "type": "float", "default": 48.0, "min": 32, "max": 64,
                "label": "Column Height", "description": "Height of the central column"
            },
            "column_sides": {
                "type": "int", "default": 8, "min": 4, "max": 12,
                "label": "Column Sides",
                "description": "Facets for round column (4=square, 8=octagonal, 12=round)"
            },
            "column_shape": {
                "type": "choice", "default": "round",
                "choices": ["square", "round"],
                "label": "Column Shape",
                "description": "square=box column, round=cylindrical column"
            },
            "has_basin_floor": {
                "type": "bool", "default": True, "label": "Basin Floor",
                "description": "Add solid floor inside the basin"
            },
            "has_column_cap": {
                "type": "bool", "default": True, "label": "Column Cap",
                "description": "Add wider cap at top of center column"
            },
            "base_overhang": {
                "type": "float", "default": 4.0, "min": 2, "max": 8,
                "label": "Base Overhang", "description": "How far the base step extends beyond the basin"
            },
            "base_height": {
                "type": "float", "default": 8.0, "min": 4, "max": 16,
                "label": "Base Height", "description": "Height of the base step"
            },
            "floor_thickness": {
                "type": "float", "default": 8.0, "min": 4, "max": 12,
                "label": "Floor Thickness", "description": "Thickness of the basin floor"
            },
            "cap_overhang": {
                "type": "float", "default": 4.0, "min": 2, "max": 8,
                "label": "Cap Overhang", "description": "How far the column cap extends beyond the column"
            },
            "cap_height": {
                "type": "float", "default": 8.0, "min": 4, "max": 12,
                "label": "Cap Height", "description": "Height of the column cap"
            },
        }

    def generate(self) -> List[Brush]:
        self._reset_tags()
        ox, oy, oz = self.params.origin
        brushes: List[Brush] = []
        inner_r = self.basin_radius - self.wall_thickness
        outer_r = self.basin_radius

        # Base step (wider ring at bottom of basin)
        base_step_r = outer_r + self.base_overhang
        base_step_h = self.base_height
        brushes.extend(_radial_ring(
            self, ox, oy, oz - base_step_h, oz,
            inner_r, base_step_r, self.segments,
        ))

        # Basin ring from radial segments
        seg_angle = 2 * math.pi / self.segments
        for i in range(self.segments):
            a1 = i * seg_angle
            a2 = (i + 1) * seg_angle
            brushes.append(self._radial_segment(
                ox, oy, oz, oz + self.basin_height,
                inner_r, outer_r, a1, a2,
                texture=self.texture_structural,
            ))

        # Basin floor (solid disk inside the basin)
        if self.has_basin_floor:
            floor_h = self.floor_thickness
            brushes.extend(_radial_disk(
                self, ox, oy, oz, oz + floor_h,
                inner_r, self.segments,
            ))

        # Central column
        if self.has_column:
            if self.column_shape == "square":
                cr = self.column_radius
                brushes.append(self._structural_box(
                    ox - cr, oy - cr, oz,
                    ox + cr, oy + cr, oz + self.column_height,
                ))
            else:
                brushes.extend(_radial_disk(
                    self, ox, oy, oz, oz + self.column_height,
                    self.column_radius, self.column_sides,
                ))

            # Column cap (wider cap at top of column)
            if self.has_column_cap:
                cap_r = self.column_radius + self.cap_overhang
                cap_h = self.cap_height
                col_top = oz + self.column_height
                if self.column_shape == "square":
                    brushes.append(self._structural_box(
                        ox - cap_r, oy - cap_r, col_top,
                        ox + cap_r, oy + cap_r, col_top + cap_h,
                    ))
                else:
                    brushes.extend(_radial_disk(
                        self, ox, oy, col_top, col_top + cap_h,
                        cap_r, self.column_sides,
                    ))

        return brushes


# ---------------------------------------------------------------------------
# Tier 3 — Complex (8+ brushes, distinctive silhouettes)
# ---------------------------------------------------------------------------

class Well(GeometricPrimitive):
    """Ring of radial segments with optional A-frame roof."""

    well_radius: float = 32.0
    wall_height: float = 32.0
    wall_thickness: float = 8.0
    segments: int = 8
    has_roof: bool = False
    has_rim: bool = True
    roof_height: float = 48.0
    rim_height: float = 8.0
    rim_overhang: float = 4.0
    post_width: float = 8.0
    roof_overhang: float = 8.0

    @classmethod
    def get_display_name(cls) -> str:
        return "Well"

    @classmethod
    def get_category(cls) -> str:
        return "Structural"

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "well_radius": {
                "type": "float", "default": 32.0, "min": 24, "max": 48,
                "label": "Radius", "description": "Outer radius of the well wall"
            },
            "wall_height": {
                "type": "float", "default": 32.0, "min": 24, "max": 48,
                "label": "Wall Height", "description": "Height of the stone wall ring"
            },
            "wall_thickness": {
                "type": "float", "default": 8.0, "min": 8, "max": 16,
                "label": "Wall Thickness", "description": "Thickness of the ring wall"
            },
            "segments": {
                "type": "int", "default": 8, "min": 4, "max": 12,
                "label": "Segments", "description": "Number of radial segments"
            },
            "has_roof": {
                "type": "bool", "default": False, "label": "Roof",
                "description": "Add an A-frame roof above the well"
            },
            "has_rim": {
                "type": "bool", "default": True, "label": "Rim Cap",
                "description": "Add a wider capstone rim at top of wall"
            },
            "roof_height": {
                "type": "float", "default": 48.0, "min": 32, "max": 64,
                "label": "Roof Height", "description": "Height of the roof peak above the wall"
            },
            "rim_height": {
                "type": "float", "default": 8.0, "min": 4, "max": 16,
                "label": "Rim Height", "description": "Vertical thickness of the capstone rim"
            },
            "rim_overhang": {
                "type": "float", "default": 4.0, "min": 2, "max": 8,
                "label": "Rim Overhang", "description": "How far the rim extends beyond the wall"
            },
            "post_width": {
                "type": "float", "default": 8.0, "min": 4, "max": 12,
                "label": "Post Width", "description": "Width of the roof support posts"
            },
            "roof_overhang": {
                "type": "float", "default": 8.0, "min": 4, "max": 16,
                "label": "Roof Overhang", "description": "How far the roof extends beyond the well"
            },
        }

    def generate(self) -> List[Brush]:
        self._reset_tags()
        ox, oy, oz = self.params.origin
        brushes: List[Brush] = []
        inner_r = self.well_radius - self.wall_thickness
        outer_r = self.well_radius

        # Wall ring
        seg_angle = 2 * math.pi / self.segments
        for i in range(self.segments):
            a1 = i * seg_angle
            a2 = (i + 1) * seg_angle
            brushes.append(self._radial_segment(
                ox, oy, oz, oz + self.wall_height,
                inner_r, outer_r, a1, a2,
                texture=self.texture_structural,
            ))

        # Rim cap (wider capstone at top of wall)
        if self.has_rim:
            rim_h = self.rim_height
            # BUG FIX: Use max(1, ...) to prevent undersized inner radius
            # that creates visible holes (same class as barrel cap bug)
            rim_inner = max(1, inner_r - self.rim_overhang)
            rim_outer = outer_r + self.rim_overhang
            wall_top_rim = oz + self.wall_height
            for i in range(self.segments):
                a1 = i * seg_angle
                a2 = (i + 1) * seg_angle
                brushes.append(self._radial_segment(
                    ox, oy, wall_top_rim, wall_top_rim + rim_h,
                    rim_inner, rim_outer, a1, a2,
                    texture=self.texture_structural,
                ))

        # A-frame roof
        if self.has_roof:
            post_w = self.post_width
            hp = post_w / 2
            wall_top = oz + self.wall_height
            roof_top = wall_top + self.roof_height
            r = outer_r

            # Two vertical posts on opposite sides with post bases
            brushes.append(self._structural_box(
                ox - hp, oy - r - hp, wall_top,
                ox + hp, oy - r + hp, roof_top,
            ))
            # Post base 1
            brushes.append(self._structural_box(
                ox - 8, oy - r - 8, wall_top,
                ox + 8, oy - r + 8, wall_top + 8,
            ))
            brushes.append(self._structural_box(
                ox - hp, oy + r - hp, wall_top,
                ox + hp, oy + r + hp, roof_top,
            ))
            # Post base 2
            brushes.append(self._structural_box(
                ox - 8, oy + r - 8, wall_top,
                ox + 8, oy + r + 8, wall_top + 8,
            ))

            # Ridge beam connecting posts at top
            brushes.append(self._structural_box(
                ox - hp, oy - r, roof_top - post_w,
                ox + hp, oy + r, roof_top,
            ))

            # Two roof slopes as wedges (x1 < x2 always for _wedge)
            roof_span = r + self.roof_overhang  # Overhang

            # Left slope: ramps UP from left eave to center ridge
            brushes.append(self._wedge(
                ox - roof_span, oy - r - hp, wall_top,
                ox, oy + r + hp, roof_top,
                ramp_axis="x",
            ))
            # Right slope: ramps DOWN from center ridge to right eave
            brushes.append(self._wedge(
                ox, oy - r - hp, roof_top,
                ox + roof_span, oy + r + hp, wall_top,
                ramp_axis="x",
            ))

        return brushes


class Cage(GeometricPrimitive):
    """Vertical bars between top/bottom frames."""

    cage_width: float = 48.0
    cage_height: float = 72.0
    bar_count: int = 8
    bar_width: float = 8.0
    style: str = "standing"
    frame_thickness: float = 8.0
    has_top_cross: bool = True
    bar_shape: str = "square"
    bar_sides: int = 6
    chain_width: float = 8.0
    chain_height: float = 32.0
    ring_size: float = 16.0

    @classmethod
    def get_display_name(cls) -> str:
        return "Cage"

    @classmethod
    def get_category(cls) -> str:
        return "Structural"

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "cage_width": {
                "type": "float", "default": 48.0, "min": 32, "max": 64,
                "label": "Width", "description": "Width and depth of the cage"
            },
            "cage_height": {
                "type": "float", "default": 72.0, "min": 48, "max": 96,
                "label": "Height", "description": "Interior height between frames"
            },
            "bar_count": {
                "type": "int", "default": 8, "min": 4, "max": 12,
                "label": "Bars", "description": "Total number of vertical bars"
            },
            "bar_width": {
                "type": "float", "default": 8.0, "min": 2, "max": 12,
                "label": "Bar Width", "description": "Width of each bar"
            },
            "style": {
                "type": "choice", "default": "standing",
                "choices": ["standing", "hanging"],
                "label": "Style",
                "description": "standing=on floor, hanging=suspended (adds chain column above)"
            },
            "frame_thickness": {
                "type": "float", "default": 8.0, "min": 2, "max": 16,
                "label": "Frame Thickness", "description": "Thickness of the top/bottom frames"
            },
            "has_top_cross": {
                "type": "bool", "default": True, "label": "Top Crossbars",
                "description": "Add perpendicular crossbars on top frame"
            },
            "bar_shape": {
                "type": "choice", "default": "square",
                "choices": ["square", "round"],
                "label": "Bar Shape",
                "description": "square=box bars, round=cylindrical bars"
            },
            "bar_sides": {
                "type": "int", "default": 6, "min": 3, "max": 12,
                "label": "Bar Sides",
                "description": "Facets for round bars (4=square, 8=octagonal, 12=round)"
            },
            "chain_width": {
                "type": "float", "default": 8.0, "min": 4, "max": 12,
                "label": "Chain Width", "description": "Width of the hanging chain"
            },
            "chain_height": {
                "type": "float", "default": 32.0, "min": 16, "max": 48,
                "label": "Chain Height", "description": "Height of the hanging chain"
            },
            "ring_size": {
                "type": "float", "default": 16.0, "min": 8, "max": 24,
                "label": "Ring Size", "description": "Size of the chain attachment ring"
            },
        }

    def generate(self) -> List[Brush]:
        self._reset_tags()
        ox, oy, oz = self.params.origin
        brushes: List[Brush] = []
        hw = self.cage_width / 2
        ft = self.frame_thickness

        # Bottom frame (4 pieces forming a square ring)
        brushes.extend(self._make_frame_ring(ox, oy, oz, hw, ft, ft))

        # Top frame
        top_z = oz + ft + self.cage_height
        brushes.extend(self._make_frame_ring(ox, oy, top_z, hw, ft, ft))

        # Vertical bars distributed around perimeter
        bar_z_bot = oz + ft
        bar_z_top = top_z
        bars_per_side = max(1, self.bar_count // 4)

        for side in range(4):
            for i in range(bars_per_side):
                t = (i + 0.5) / bars_per_side
                bx, by = self._bar_position(ox, oy, hw, ft, side, t)
                if self.bar_shape == "round":
                    bar_r = max(4, self._snap_coord(self.bar_width / 2))
                    brushes.extend(_radial_disk(
                        self, bx, by, bar_z_bot, bar_z_top,
                        bar_r, self.bar_sides,
                    ))
                else:
                    hbw = self.bar_width / 2
                    brushes.append(self._structural_box(
                        bx - hbw, by - hbw, bar_z_bot,
                        bx + hbw, by + hbw, bar_z_top,
                    ))

        # Top crossbars
        if self.has_top_cross:
            bar_w = self.frame_thickness
            bar_top = top_z + ft
            # X-axis crossbar
            brushes.append(self._structural_box(
                ox - hw, oy - bar_w / 2, top_z,
                ox + hw, oy + bar_w / 2, bar_top,
            ))
            # Y-axis crossbar
            brushes.append(self._structural_box(
                ox - bar_w / 2, oy - hw, top_z,
                ox + bar_w / 2, oy + hw, bar_top,
            ))

        # Hanging style: small frame ring + vertical chain
        if self.style == "hanging":
            chain_w = self.chain_width
            chain_h = self.chain_height
            hcw = chain_w / 2
            ring_size = self.ring_size
            chain_z = top_z + ft
            # Small frame ring (4 boxes)
            for dx, dy, is_x in [(0, -ring_size / 2, True), (0, ring_size / 2, True),
                                  (-ring_size / 2, 0, False), (ring_size / 2, 0, False)]:
                if is_x:
                    brushes.append(self._structural_box(
                        ox - ring_size / 2, oy + dy - hcw, chain_z,
                        ox + ring_size / 2, oy + dy + hcw, chain_z + chain_w,
                    ))
                else:
                    brushes.append(self._structural_box(
                        ox + dx - hcw, oy - ring_size / 2, chain_z,
                        ox + dx + hcw, oy + ring_size / 2, chain_z + chain_w,
                    ))
            # Vertical chain above ring
            brushes.append(self._structural_box(
                ox - hcw, oy - hcw, chain_z + chain_w,
                ox + hcw, oy + hcw, chain_z + chain_w + chain_h,
            ))

        return brushes

    def _make_frame_ring(self, ox: float, oy: float, z: float,
                         hw: float, ft: float, height: float) -> List[Brush]:
        """Create a square frame ring from 4 box brushes (no hollow brush in idTech)."""
        brushes = []
        # Front
        brushes.append(self._structural_box(
            ox - hw, oy - hw, z,
            ox + hw, oy - hw + ft, z + height,
        ))
        # Back
        brushes.append(self._structural_box(
            ox - hw, oy + hw - ft, z,
            ox + hw, oy + hw, z + height,
        ))
        # Left (between front and back)
        brushes.append(self._structural_box(
            ox - hw, oy - hw + ft, z,
            ox - hw + ft, oy + hw - ft, z + height,
        ))
        # Right (between front and back)
        brushes.append(self._structural_box(
            ox + hw - ft, oy - hw + ft, z,
            ox + hw, oy + hw - ft, z + height,
        ))
        return brushes

    def _bar_position(self, ox: float, oy: float, hw: float,
                      ft: float, side: int, t: float):
        """Get bar center position for a given side and parametric position.

        Bars are centered on the frame pieces (between inner and outer frame edges).
        The interpolation range spans between adjacent frame inner edges.
        """
        inner = hw - ft  # Inner edge distance from center
        span = 2 * inner
        pos = -inner + t * span
        if side == 0:    # Front (Y = -hw + ft/2)
            return ox + pos, oy - hw + ft / 2
        elif side == 1:  # Back (Y = +hw - ft/2)
            return ox + pos, oy + hw - ft / 2
        elif side == 2:  # Left (X = -hw + ft/2)
            return ox - hw + ft / 2, oy + pos
        else:            # Right (X = +hw - ft/2)
            return ox + hw - ft / 2, oy + pos
