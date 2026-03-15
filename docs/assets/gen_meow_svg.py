"""Generate a cat-shaped optical mode profile SVG from scratch.

Approach: compute a 2D field on a pixel grid using a cat-shaped boundary
(via distance transform) and heart-shaped peak, then extract contour
polygons with matplotlib and export as layered SVG paths.

Usage:
    python gen_meow_svg.py
"""

from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, gaussian_filter1d

DST = Path(__file__).parent / "meow.svg"

W, H = 600, 720
N_LEVELS = 14
SMOOTH_SIGMA = 10

COLORS = [
    "#0b1034",
    "#161848",
    "#272362",
    "#3d207a",
    "#5c2590",
    "#7e2a8e",
    "#a52e7a",
    "#c83465",
    "#e04550",
    "#ec6530",
    "#f08828",
    "#f5a623",
    "#f8cc50",
    "#fae878",
]


def make_cat_mask(w, h):
    """Create a boolean mask of a cat silhouette on a w x h grid.

    Uses a polar implicit function: for each pixel, compare distance
    from center to an angle-dependent boundary radius that encodes
    ears, ear-dip, waist pinch, and hip bulge.
    """
    y_grid, x_grid = np.mgrid[0:h, 0:w]
    cx, cy = w / 2, h * 0.50
    xn = (x_grid - cx) / (w * 0.48)
    yn = -(y_grid - cy) / (h * 0.48)  # +y is up

    theta = np.arctan2(xn, yn)  # 0=up, +-pi=down

    # Base radius: elliptical, wider than tall
    r_base = 0.78 / np.sqrt((np.cos(theta) / 0.78) ** 2 + (np.sin(theta) / 0.90) ** 2)

    # Ears: bumps at upper-left and upper-right
    ear_angle_l = 0.55
    ear_angle_r = -0.55
    ear_width = 0.12
    ear_height = 0.22
    r_ears = ear_height * np.exp(
        -((theta - ear_angle_l) ** 2) / (2 * ear_width**2)
    ) + ear_height * np.exp(-((theta - ear_angle_r) ** 2) / (2 * ear_width**2))

    # Dip between ears (at top, theta=0)
    dip_width = 0.06
    r_dip = -0.16 * np.exp(-(theta**2) / (2 * dip_width**2))

    # Waist pinch on left and right sides
    waist_width = 0.15
    waist_depth = 0.16
    r_waist = -(
        waist_depth * np.exp(-((theta - np.pi / 2) ** 2) / (2 * waist_width**2))
        + waist_depth * np.exp(-((theta + np.pi / 2) ** 2) / (2 * waist_width**2))
    )

    # Wider hips below waist
    hip_angle_l = 1.8
    hip_angle_r = -1.8
    hip_w = 0.2
    hip_h = 0.06
    r_hips = hip_h * np.exp(
        -((theta - hip_angle_l) ** 2) / (2 * hip_w**2)
    ) + hip_h * np.exp(-((theta - hip_angle_r) ** 2) / (2 * hip_w**2))

    r_total = r_base + r_ears + r_dip + r_waist + r_hips
    r_pixel = np.sqrt(xn**2 + yn**2)
    mask = r_pixel < r_total

    # Smooth edges
    mask_float = gaussian_filter(mask.astype(float), sigma=3)
    mask = mask_float > 0.5

    return mask


def make_heart_peak(w, h):
    """Create a 2D peak shaped like a heart using the implicit heart equation."""
    y_grid, x_grid = np.mgrid[0:h, 0:w]
    cx, cy_heart = w / 2, h * 0.35
    sx, sy = 155, 125
    xn = (x_grid - cx) / sx
    yn = -(y_grid - cy_heart) / sy

    # Heart implicit: (x^2 + y^2 - 1)^3 - x^2 * y^3 < 0
    heart_val = (xn**2 + yn**2 - 1) ** 3 - xn**2 * yn**3
    peak = 1.0 / (1.0 + np.exp(np.clip(heart_val * 2.0, -500, 500)))

    return peak


def make_mode_field(w, h):
    """Compute the 2D mode profile: heart-peak shaped field inside cat boundary."""
    mask = make_cat_mask(w, h)

    # Distance from boundary (inside the mask)
    dist = distance_transform_edt(mask).astype(float)
    dist_max = dist.max()
    dist_norm = dist / dist_max

    heart = make_heart_peak(w, h)

    # Blend: more distance-based near edges, more heart-based deep inside
    blend = np.clip(dist_norm * 2.5, 0, 1)
    field = dist_norm * (1 - blend * 0.6) + heart * (0.2 + blend * 0.6)

    # Normalize to [0, 1] inside mask
    fmax = field[mask].max()
    field = field / fmax

    # Set outside to negative
    field[~mask] = -0.5

    # Smooth (but re-zero outside after)
    field = gaussian_filter(field, sigma=SMOOTH_SIGMA)
    field[~mask] = -0.5

    return field, mask


def extract_contour_lines(field, levels):
    """Extract contour lines at given levels using matplotlib.
    Returns list of lists of Nx2 polygon arrays, one per level.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(1, 1))
    cs = ax.contour(field, levels=levels)
    plt.close(fig)

    return [seg_list for seg_list in cs.allsegs]


def simplify_polygon(vs, tolerance=1.5):
    """Ramer-Douglas-Peucker simplification for a closed polygon."""
    if len(vs) < 5:
        return vs

    def rdp(points, eps):
        if len(points) < 3:
            return points
        start, end = points[0], points[-1]
        line_vec = end - start
        line_len = np.linalg.norm(line_vec)
        if line_len < 1e-10:
            dists = np.linalg.norm(points - start, axis=1)
        else:
            line_unit = line_vec / line_len
            proj = np.dot(points - start, line_unit)
            closest = start + np.outer(proj, line_unit)
            dists = np.linalg.norm(points - closest, axis=1)
        idx = np.argmax(dists)
        if dists[idx] > eps:
            left = rdp(points[: idx + 1], eps)
            right = rdp(points[idx:], eps)
            return np.vstack([left[:-1], right])
        return np.array([start, end])

    return rdp(vs, tolerance)


def polygon_to_svg_path(vertices):
    """Convert Nx2 polygon vertices to a smooth SVG cubic-bezier path."""
    if len(vertices) < 3:
        return ""

    vs = np.array(vertices)
    if np.allclose(vs[0], vs[-1]):
        vs = vs[:-1]
    if len(vs) < 3:
        return ""

    # Smooth with wrap-around Gaussian
    n = len(vs)
    if n > 8:
        pad = min(n // 4, 30)
        xs = np.concatenate([vs[-pad:, 0], vs[:, 0], vs[:pad, 0]])
        ys = np.concatenate([vs[-pad:, 1], vs[:, 1], vs[:pad, 1]])
        sigma = max(3.0, n / 60)
        xs = gaussian_filter1d(xs, sigma=sigma)[pad : pad + n]
        ys = gaussian_filter1d(ys, sigma=sigma)[pad : pad + n]
        vs = np.column_stack([xs, ys])

    # Simplify to reduce point count
    vs = simplify_polygon(vs, tolerance=1.2)
    n = len(vs)
    if n < 3:
        return ""

    # Build SVG path with cubic bezier (Catmull-Rom interpolation)
    parts = [f"M{vs[0, 0]:.1f},{vs[0, 1]:.1f}"]
    for i in range(n):
        p0 = vs[(i - 1) % n]
        p1 = vs[i % n]
        p2 = vs[(i + 1) % n]
        p3 = vs[(i + 2) % n]
        cp1x = p1[0] + (p2[0] - p0[0]) / 6
        cp1y = p1[1] + (p2[1] - p0[1]) / 6
        cp2x = p2[0] - (p3[0] - p1[0]) / 6
        cp2y = p2[1] - (p3[1] - p1[1]) / 6
        parts.append(
            f"C{cp1x:.1f},{cp1y:.1f} {cp2x:.1f},{cp2y:.1f} {p2[0]:.1f},{p2[1]:.1f}"
        )
    parts.append("Z")
    return "".join(parts)


def main():
    print("Computing mode field...")
    field, mask = make_mode_field(W, H)

    # Contour levels from cat boundary to peak
    levels = np.linspace(0.08, 0.95, N_LEVELS)

    print("Extracting contours...")
    contour_lines = extract_contour_lines(field, levels)

    # Build SVG: white background, then filled contours from outermost to innermost
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
        f'  <rect width="{W}" height="{H}" fill="white"/>',
    ]

    for i, polygons in enumerate(contour_lines):
        if i >= len(COLORS):
            break
        color = COLORS[i]
        valid = [p for p in polygons if len(p) > 20]
        if not valid:
            continue
        largest = max(valid, key=len)
        path_d = polygon_to_svg_path(largest)
        if path_d:
            lines.append(f'  <path d="{path_d}" fill="{color}"/>')
        print(f"  Layer {i}: {color} ({len(largest)} pts)")

    lines.append("</svg>")
    svg_out = "\n".join(lines) + "\n"
    DST.write_text(svg_out)
    print(f"\nWritten {DST} ({len(svg_out)} bytes)")


if __name__ == "__main__":
    main()
