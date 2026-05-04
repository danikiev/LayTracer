"""Generate the LayTracer logo assets.

Run from the repository root:

    python branding/logo/generate_logos.py

Use ``--check`` to verify that committed logo files are up to date.
"""

from __future__ import annotations

import argparse
from io import BytesIO
import sys
from dataclasses import dataclass
from math import sqrt
from pathlib import Path

from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
from matplotlib.patches import Circle as MplCircle
from matplotlib.patches import PathPatch
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D


LIGHT_NAVY = "#255699"
ACCENT_ORANGE = "#FF6A1A"
LIGHT_GRAY_BLUE = "#9AA7B2"
WHITE = "#FFFFFF"

ROOT = Path(__file__).resolve().parents[2]
FONT_PATH = Path(__file__).resolve().parent / "fonts" / "Poppins-SemiBold.ttf"
STATIC_DIR = ROOT / "docs" / "source" / "_static"

FULL_LOGO = STATIC_DIR / "laytracer-logo-full.svg"
FULL_LOGO_PDF = STATIC_DIR / "laytracer-logo-full.pdf"
MEDIUM_LOGO = STATIC_DIR / "laytracer-logo-medium.svg"
ICON_LOGO = STATIC_DIR / "laytracer-icon.svg"
ICON_CIRCLE_LOGO = STATIC_DIR / "laytracer-icon-circle.svg"

TAGLINE_FAST = "FAST TWO-POINT"
TAGLINE_REST = "SEISMIC RAY TRACING IN LAYERED MEDIA"


@dataclass(frozen=True)
class TextBox:
    left: float
    right: float
    top: float
    bottom: float


@dataclass(frozen=True)
class YMotif:
    letter: str
    box: TextBox
    rays: list[str]
    dots: list[str]


def _fmt(value: float) -> str:
    if abs(value) < 0.0005:
        value = 0.0
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _svg_point(x: float, y: float, tx: float, baseline_y: float) -> str:
    return f"{_fmt(tx + x)} {_fmt(baseline_y - y)}"


def _path_to_svg(path: MplPath, tx: float, baseline_y: float) -> str:
    commands: list[str] = []
    for vertices, code in path.iter_segments(curves=True, simplify=False):
        points = vertices.reshape((-1, 2))
        if code == MplPath.MOVETO:
            commands.append(f"M {_svg_point(points[0, 0], points[0, 1], tx, baseline_y)}")
        elif code == MplPath.LINETO:
            commands.append(f"L {_svg_point(points[0, 0], points[0, 1], tx, baseline_y)}")
        elif code == MplPath.CURVE3:
            commands.append(
                "Q "
                + " ".join(_svg_point(x, y, tx, baseline_y) for x, y in points)
            )
        elif code == MplPath.CURVE4:
            commands.append(
                "C "
                + " ".join(_svg_point(x, y, tx, baseline_y) for x, y in points)
            )
        elif code == MplPath.CLOSEPOLY:
            commands.append("Z")
    return " ".join(commands)


def _text_path(text: str, size: float, font: FontProperties) -> TextPath:
    return TextPath((0, 0), text, size=size, prop=font)


def _text_element(
    text: str,
    *,
    left: float,
    baseline_y: float,
    size: float,
    color: str,
    font: FontProperties,
) -> tuple[str, TextBox]:
    path = _text_path(text, size, font)
    bbox = path.get_extents()
    tx = left - bbox.x0
    d = _path_to_svg(path, tx, baseline_y)
    element = f'<path fill="{color}" d="{d}"/>'
    box = TextBox(
        left=left,
        right=left + bbox.width,
        top=baseline_y - bbox.y1,
        bottom=baseline_y - bbox.y0,
    )
    return element, box


def _pdf_canvas(width: float, height: float) -> tuple[Figure, object]:
    fig = Figure(figsize=(width / 72.0, height / 72.0), dpi=72)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_axis_off()
    return fig, ax


def _pdf_text_element(
    ax,
    text: str,
    *,
    left: float,
    baseline_y: float,
    size: float,
    color: str,
    font: FontProperties,
) -> TextBox:
    path = _text_path(text, size, font)
    bbox = path.get_extents()
    tx = left - bbox.x0
    transform = Affine2D().scale(1.0, -1.0).translate(tx, baseline_y)
    patch = PathPatch(
        path,
        facecolor=color,
        edgecolor="none",
        linewidth=0,
        transform=transform + ax.transData,
    )
    ax.add_patch(patch)
    return TextBox(
        left=left,
        right=left + bbox.width,
        top=baseline_y - bbox.y1,
        bottom=baseline_y - bbox.y0,
    )


def _pdf_tracked_text_elements(
    ax,
    text: str,
    *,
    left: float,
    baseline_y: float,
    size: float,
    color: str,
    font: FontProperties,
    tracking: float,
) -> TextBox:
    boxes: list[TextBox] = []
    cursor = left
    for letter in text:
        box = _pdf_text_element(
            ax,
            letter,
            left=cursor,
            baseline_y=baseline_y,
            size=size,
            color=color,
            font=font,
        )
        boxes.append(box)
        cursor = box.right + tracking

    return TextBox(
        left=min(box.left for box in boxes),
        right=max(box.right for box in boxes),
        top=min(box.top for box in boxes),
        bottom=max(box.bottom for box in boxes),
    )


def _line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    color: str,
    width: float,
    opacity: float | None = None,
    dasharray: str | None = None,
) -> str:
    opacity_attr = "" if opacity is None else f' opacity="{_fmt(opacity)}"'
    dash_attr = "" if dasharray is None else f' stroke-dasharray="{dasharray}"'
    return (
        f'<line x1="{_fmt(x1)}" y1="{_fmt(y1)}" x2="{_fmt(x2)}" y2="{_fmt(y2)}" '
        f'stroke="{color}" stroke-width="{_fmt(width)}" '
        f'stroke-linecap="round"{opacity_attr}{dash_attr}/>'
    )


def _circle(cx: float, cy: float, radius: float, color: str) -> str:
    return (
        f'<circle cx="{_fmt(cx)}" cy="{_fmt(cy)}" r="{_fmt(radius)}" '
        f'fill="{color}"/>'
    )


def _pdf_line(
    ax,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    color: str,
    width: float,
    opacity: float | None = None,
    dasharray: str | None = None,
) -> None:
    line = Line2D(
        [x1, x2],
        [y1, y2],
        color=color,
        linewidth=width,
        alpha=opacity,
        solid_capstyle="round",
        dash_capstyle="round",
    )
    if dasharray:
        line.set_dashes([float(value) for value in dasharray.split()])
    ax.add_line(line)


def _pdf_circle(ax, cx: float, cy: float, radius: float, color: str) -> None:
    ax.add_patch(MplCircle((cx, cy), radius, facecolor=color, edgecolor="none"))


def _svg(width: float, height: float, elements: list[str]) -> str:
    body = "\n  ".join(elements)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{_fmt(width)}" '
        f'height="{_fmt(height)}" viewBox="0 0 {_fmt(width)} {_fmt(height)}" '
        'role="img" aria-labelledby="title desc">\n'
        "  <title id=\"title\">LayTracer logo</title>\n"
        "  <desc id=\"desc\">LayTracer wordmark with a layer boundary, dashed reflected rays, and colored y endpoint dots.</desc>\n"
        f"  {body}\n"
        "</svg>\n"
    )


def _tracked_text_elements(
    text: str,
    *,
    left: float,
    baseline_y: float,
    size: float,
    color: str,
    font: FontProperties,
    tracking: float,
) -> tuple[list[str], TextBox]:
    elements: list[str] = []
    boxes: list[TextBox] = []
    cursor = left
    for letter in text:
        element, box = _text_element(
            letter,
            left=cursor,
            baseline_y=baseline_y,
            size=size,
            color=color,
            font=font,
        )
        elements.append(element)
        boxes.append(box)
        cursor = box.right + tracking

    return elements, TextBox(
        left=min(box.left for box in boxes),
        right=max(box.right for box in boxes),
        top=min(box.top for box in boxes),
        bottom=max(box.bottom for box in boxes),
    )


def _scanline_intervals(polygon, y: float) -> list[tuple[float, float]]:
    crossings: list[float] = []
    points = list(polygon)
    for start, end in zip(points, points[1:] + points[:1]):
        x1, y1 = float(start[0]), float(start[1])
        x2, y2 = float(end[0]), float(end[1])
        if (y1 <= y < y2) or (y2 <= y < y1):
            crossings.append(x1 + (y - y1) * (x2 - x1) / (y2 - y1))

    crossings.sort()
    return [
        (crossings[index], crossings[index + 1])
        for index in range(0, len(crossings) - 1, 2)
    ]


def _fit_x_over_y(points: list[tuple[float, float]]) -> tuple[float, float]:
    mean_y = sum(y for y, _ in points) / len(points)
    mean_x = sum(x for _, x in points) / len(points)
    variance_y = sum((y - mean_y) ** 2 for y, _ in points)
    if variance_y == 0:
        return 0.0, mean_x
    slope = sum((y - mean_y) * (x - mean_x) for y, x in points) / variance_y
    return slope, mean_x - slope * mean_y


def _y_ray_geometry(
    *,
    left: float,
    baseline_y: float,
    size: float,
    font: FontProperties,
    boundary_y: float,
    radius: float,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    # Measure the filled y outline so the ray angle follows the actual font.
    path = _text_path("y", size, font)
    bbox = path.get_extents()
    polygon = path.to_polygons()[0]
    tx = left - bbox.x0

    left_points: list[tuple[float, float]] = []
    right_points: list[tuple[float, float]] = []
    span = bbox.y1 - bbox.y0
    for index in range(40):
        y = bbox.y1 - 2.0 - index * (span - 4.0) / 39.0
        intervals = _scanline_intervals(polygon, y)
        if len(intervals) < 2:
            continue
        left_interval = intervals[0]
        right_interval = intervals[-1]
        left_points.append((y, 0.5 * (left_interval[0] + left_interval[1])))
        right_points.append((y, 0.5 * (right_interval[0] + right_interval[1])))

    if len(left_points) < 2 or len(right_points) < 2:
        raise RuntimeError("Could not measure the two upper branches of the y glyph.")

    left_slope_local, _ = _fit_x_over_y(left_points)
    right_slope_local, _ = _fit_x_over_y(right_points)
    left_slope_svg = -left_slope_local
    right_slope_svg = -right_slope_local
    ray_slope = 0.5 * (abs(left_slope_svg) + abs(right_slope_svg))

    boundary_local_y = baseline_y - boundary_y
    boundary_intervals = _scanline_intervals(polygon, boundary_local_y)
    if boundary_intervals:
        interval = boundary_intervals[len(boundary_intervals) // 2]
        reflection_x = tx + 0.5 * (interval[0] + interval[1])
    else:
        reflection_x = left + 0.5 * bbox.width
    reflection_x += 0.015 * size

    dot_y = baseline_y - bbox.y1 - radius - 3
    reflection_point = (reflection_x, boundary_y)
    ray_dx = ray_slope * (boundary_y - dot_y)
    navy_dot = (reflection_x - ray_dx, dot_y)
    orange_dot = (reflection_x + ray_dx, dot_y)
    return navy_dot, orange_dot, reflection_point


def _y_center_at_svg_y(
    *,
    left: float,
    baseline_y: float,
    size: float,
    font: FontProperties,
    svg_y: float,
) -> float:
    path = _text_path("y", size, font)
    bbox = path.get_extents()
    polygon = path.to_polygons()[0]
    local_y = baseline_y - svg_y
    intervals = _scanline_intervals(polygon, local_y)
    if not intervals:
        return left + 0.5 * bbox.width
    interval = intervals[len(intervals) // 2]
    return left - bbox.x0 + 0.5 * (interval[0] + interval[1])


def _branch_rays(
    navy_dot: tuple[float, float],
    orange_dot: tuple[float, float],
    reflection_point: tuple[float, float],
    *,
    width: float = 4.0,
) -> list[str]:
    return [
        _line(
            navy_dot[0],
            navy_dot[1],
            reflection_point[0],
            reflection_point[1],
            color=LIGHT_GRAY_BLUE,
            width=width,
            opacity=0.95,
            dasharray="9 7",
        ),
        _line(
            reflection_point[0],
            reflection_point[1],
            orange_dot[0],
            orange_dot[1],
            color=LIGHT_GRAY_BLUE,
            width=width,
            opacity=0.95,
            dasharray="9 7",
        ),
    ]


def _y_motif(
    *,
    left: float,
    baseline_y: float,
    size: float,
    font: FontProperties,
    boundary_y: float,
    dot_radius: float = 9.0,
    ray_width: float = 4.0,
) -> YMotif:
    letter, box = _text_element(
        "y",
        left=left,
        baseline_y=baseline_y,
        size=size,
        color=LIGHT_NAVY,
        font=font,
    )
    navy_dot, orange_dot, reflection_point = _y_ray_geometry(
        left=left,
        baseline_y=baseline_y,
        size=size,
        font=font,
        boundary_y=boundary_y,
        radius=dot_radius,
    )
    return YMotif(
        letter=letter,
        box=box,
        rays=_branch_rays(
            navy_dot,
            orange_dot,
            reflection_point,
            width=ray_width,
        ),
        dots=[
            _circle(navy_dot[0], navy_dot[1], dot_radius, LIGHT_NAVY),
            _circle(orange_dot[0], orange_dot[1], dot_radius, ACCENT_ORANGE),
        ],
    )


def _pdf_y_letter(
    ax,
    *,
    left: float,
    baseline_y: float,
    size: float,
    font: FontProperties,
    boundary_y: float,
    dot_radius: float = 9.0,
) -> tuple[TextBox, tuple[float, float], tuple[float, float], tuple[float, float]]:
    box = _pdf_text_element(
        ax,
        "y",
        left=left,
        baseline_y=baseline_y,
        size=size,
        color=LIGHT_NAVY,
        font=font,
    )
    navy_dot, orange_dot, reflection_point = _y_ray_geometry(
        left=left,
        baseline_y=baseline_y,
        size=size,
        font=font,
        boundary_y=boundary_y,
        radius=dot_radius,
    )
    return box, navy_dot, orange_dot, reflection_point


def _pdf_y_overlays(
    ax,
    navy_dot: tuple[float, float],
    orange_dot: tuple[float, float],
    reflection_point: tuple[float, float],
    *,
    dot_radius: float = 9.0,
    ray_width: float = 4.0,
) -> None:
    _pdf_line(
        ax,
        navy_dot[0],
        navy_dot[1],
        reflection_point[0],
        reflection_point[1],
        color=LIGHT_GRAY_BLUE,
        width=ray_width,
        opacity=0.95,
        dasharray="9 7",
    )
    _pdf_line(
        ax,
        reflection_point[0],
        reflection_point[1],
        orange_dot[0],
        orange_dot[1],
        color=LIGHT_GRAY_BLUE,
        width=ray_width,
        opacity=0.95,
        dasharray="9 7",
    )
    _pdf_circle(ax, navy_dot[0], navy_dot[1], dot_radius, LIGHT_NAVY)
    _pdf_circle(ax, orange_dot[0], orange_dot[1], dot_radius, ACCENT_ORANGE)


def _chord_x_range(
    *,
    center_x: float,
    center_y: float,
    radius: float,
    y: float,
    stroke_width: float,
) -> tuple[float, float]:
    inner_radius = radius - 0.5 * stroke_width
    dy = y - center_y
    if abs(dy) >= inner_radius:
        raise ValueError("Chord y-position is outside the circle.")
    half_width = sqrt(inner_radius**2 - dy**2)
    return center_x - half_width, center_x + half_width


def _wordmark(font: FontProperties, *, include_tagline: bool) -> str:
    width = 800.0
    height = 180.0 if include_tagline else 168.0
    left = 56.0
    baseline_y = 104.0
    y_shift = 14.0
    boundary_y = 120.0
    font_size = 128.0
    tracking = 7.0
    y_gap = 24.0

    elements: list[str] = []
    la_elements, la_box = _tracked_text_elements(
        "La",
        left=left,
        baseline_y=baseline_y,
        size=font_size,
        color=LIGHT_NAVY,
        font=font,
        tracking=tracking,
    )
    y_left = la_box.right + y_gap
    y_motif = _y_motif(
        left=y_left,
        baseline_y=baseline_y + y_shift,
        size=font_size,
        font=font,
        boundary_y=boundary_y,
    )
    y_box = y_motif.box
    tracer_elements, tracer_box = _tracked_text_elements(
        "Tracer",
        left=y_box.right + y_gap,
        baseline_y=baseline_y,
        size=font_size,
        color=ACCENT_ORANGE,
        font=font,
        tracking=tracking,
    )

    elements.append(
        _line(
            la_box.left - 14.0,
            boundary_y,
            tracer_box.right + 16.0,
            boundary_y,
            color=LIGHT_GRAY_BLUE,
            width=10.0,
        )
    )
    elements.extend(la_elements)
    elements.append(y_motif.letter)
    elements.extend(tracer_elements)
    elements.extend(y_motif.rays)
    elements.extend(y_motif.dots)

    if include_tagline:
        tagline_size = 17.0
        tagline_baseline_y = boundary_y + 27.0
        descender_x = _y_center_at_svg_y(
            left=y_left,
            baseline_y=baseline_y + y_shift,
            size=font_size,
            font=font,
            svg_y=tagline_baseline_y - 0.5 * tagline_size,
        )
        text_gap = 24.0
        fast, fast_box = _text_element(
            TAGLINE_FAST,
            left=0.0,
            baseline_y=tagline_baseline_y,
            size=tagline_size,
            color=ACCENT_ORANGE,
            font=font,
        )
        fast_width = fast_box.right - fast_box.left
        fast, _ = _text_element(
            TAGLINE_FAST,
            left=descender_x - text_gap - fast_width,
            baseline_y=tagline_baseline_y,
            size=tagline_size,
            color=ACCENT_ORANGE,
            font=font,
        )
        rest, _ = _text_element(
            TAGLINE_REST,
            left=descender_x + text_gap,
            baseline_y=tagline_baseline_y,
            size=tagline_size,
            color=LIGHT_NAVY,
            font=font,
        )
        elements.extend([fast, rest])

    return _svg(width, height, elements)


def _wordmark_pdf(font: FontProperties) -> bytes:
    width = 800.0
    height = 180.0
    left = 56.0
    baseline_y = 104.0
    y_shift = 14.0
    boundary_y = 120.0
    font_size = 128.0
    tracking = 7.0
    y_gap = 24.0

    fig, ax = _pdf_canvas(width, height)
    la_box = _pdf_tracked_text_elements(
        ax,
        "La",
        left=left,
        baseline_y=baseline_y,
        size=font_size,
        color=LIGHT_NAVY,
        font=font,
        tracking=tracking,
    )
    y_left = la_box.right + y_gap
    y_box, navy_dot, orange_dot, reflection_point = _pdf_y_letter(
        ax,
        left=y_left,
        baseline_y=baseline_y + y_shift,
        size=font_size,
        font=font,
        boundary_y=boundary_y,
    )
    tracer_box = _pdf_tracked_text_elements(
        ax,
        "Tracer",
        left=y_box.right + y_gap,
        baseline_y=baseline_y,
        size=font_size,
        color=ACCENT_ORANGE,
        font=font,
        tracking=tracking,
    )

    _pdf_line(
        ax,
        la_box.left - 14.0,
        boundary_y,
        tracer_box.right + 16.0,
        boundary_y,
        color=LIGHT_GRAY_BLUE,
        width=10.0,
    )
    # Move the boundary behind the wordmark while preserving the exact geometry.
    ax.lines[-1].set_zorder(-1)
    _pdf_y_overlays(ax, navy_dot, orange_dot, reflection_point)

    tagline_size = 17.0
    tagline_baseline_y = boundary_y + 27.0
    descender_x = _y_center_at_svg_y(
        left=y_left,
        baseline_y=baseline_y + y_shift,
        size=font_size,
        font=font,
        svg_y=tagline_baseline_y - 0.5 * tagline_size,
    )
    text_gap = 24.0
    _, fast_box = _text_element(
        TAGLINE_FAST,
        left=0.0,
        baseline_y=tagline_baseline_y,
        size=tagline_size,
        color=ACCENT_ORANGE,
        font=font,
    )
    fast_width = fast_box.right - fast_box.left
    _pdf_text_element(
        ax,
        TAGLINE_FAST,
        left=descender_x - text_gap - fast_width,
        baseline_y=tagline_baseline_y,
        size=tagline_size,
        color=ACCENT_ORANGE,
        font=font,
    )
    _pdf_text_element(
        ax,
        TAGLINE_REST,
        left=descender_x + text_gap,
        baseline_y=tagline_baseline_y,
        size=tagline_size,
        color=LIGHT_NAVY,
        font=font,
    )

    output = BytesIO()
    fig.savefig(
        output,
        format="pdf",
        transparent=True,
        metadata={"Creator": "LayTracer logo generator", "CreationDate": None},
    )
    return output.getvalue()


def _icon(font: FontProperties, *, include_circle: bool = False) -> str:
    width = 180.0
    height = 180.0
    center_x = 90.0
    center_y = 90.0
    circle_radius = 84.0
    boundary_y = 120.0
    boundary_width = 10.0
    font_size = 128.0
    baseline_y = 118.0

    y_path = _text_path("y", font_size, font)
    y_bbox = y_path.get_extents()
    y_left = 0.5 * (width - y_bbox.width)

    boundary_x1, boundary_x2 = _chord_x_range(
        center_x=center_x,
        center_y=center_y,
        radius=circle_radius,
        y=boundary_y,
        stroke_width=boundary_width,
    )
    y_motif = _y_motif(
        left=y_left,
        baseline_y=baseline_y,
        size=font_size,
        font=font,
        boundary_y=boundary_y,
    )

    elements: list[str] = []
    if include_circle:
        elements.append(_circle(center_x, center_y, circle_radius, WHITE))

    elements.extend([
        _line(
            boundary_x1,
            boundary_y,
            boundary_x2,
            boundary_y,
            color=LIGHT_GRAY_BLUE,
            width=boundary_width,
        ),
        y_motif.letter,
        *y_motif.rays,
        *y_motif.dots,
    ]
    )
    return _svg(width, height, elements)


def _icon_plain(font: FontProperties) -> str:
    return _icon(font, include_circle=False)


def _icon_circle(font: FontProperties) -> str:
    return _icon(font, include_circle=True)


def _render_assets() -> tuple[dict[Path, str], dict[Path, bytes]]:
    if not FONT_PATH.exists():
        raise FileNotFoundError(f"Missing vendored font: {FONT_PATH}")
    font = FontProperties(fname=str(FONT_PATH))
    svg_assets = {
        FULL_LOGO: _wordmark(font, include_tagline=True),
        MEDIUM_LOGO: _wordmark(font, include_tagline=False),
        ICON_LOGO: _icon_plain(font),
        ICON_CIRCLE_LOGO: _icon_circle(font),
    }
    pdf_assets = {
        FULL_LOGO_PDF: _wordmark_pdf(font),
    }
    return svg_assets, pdf_assets


def _write_assets(svg_assets: dict[Path, str], pdf_assets: dict[Path, bytes]) -> None:
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    for path, content in svg_assets.items():
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
    for path, content in pdf_assets.items():
        path.write_bytes(content)


def _check_assets(svg_assets: dict[Path, str], pdf_assets: dict[Path, bytes]) -> int:
    stale: list[str] = []
    for path, expected in svg_assets.items():
        if not path.exists() or path.read_text(encoding="utf-8") != expected:
            stale.append(str(path.relative_to(ROOT)))
    for path, expected in pdf_assets.items():
        if not path.exists() or path.read_bytes() != expected:
            stale.append(str(path.relative_to(ROOT)))

    if stale:
        print("Stale or missing logo assets:")
        for item in stale:
            print(f"  - {item}")
        print("Run: python branding/logo/generate_logos.py")
        return 1

    print("Logo assets are up to date.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if generated logo assets differ from committed files",
    )
    args = parser.parse_args(argv)

    svg_assets, pdf_assets = _render_assets()
    if args.check:
        return _check_assets(svg_assets, pdf_assets)

    _write_assets(svg_assets, pdf_assets)
    for path in [*svg_assets, *pdf_assets]:
        print(f"Wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
