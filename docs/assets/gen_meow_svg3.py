"""Generate a cat-shaped optical mode profile SVG with rib waveguide cross-section.

Style: rib waveguide T-shape (core + slab) in light blue behind cat-shaped
contour lines (unfilled outlines) in the inferno color palette.

Usage:
    python gen_meow_svg3.py
"""

from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, gaussian_filter1d

DST = Path(__file__).parent / "meow.svg"

W_FIELD = 500  # field computation width
H_FIELD = 500  # field computation height (generous, will be cropped)
N_LEVELS = 5
SMOOTH_SIGMA = 8

# Contour line colors: dark navy -> purple -> red -> orange -> yellow
LINE_COLORS = [
    "#0b1034",
    "#3d207a",
    "#952c85",
    "#e04550",
    "#f08828",
]

LINE_WIDTHS = [3.5, 3.0, 2.5, 2.2, 2.0]

# Waveguide cross-section
WG_CLAD = "#e8eef5"
WG_CORE = "#d0dff0"
WG_SLAB = "#bdd0e6"


def make_cat_mask(w, h, cx, cy, scale):
    """Create a boolean mask of a cat face silhouette.

    Round head with prominent triangular ears, like a cartoon cat face.
    """
    y_grid, x_grid = np.mgrid[0:h, 0:w]
    xn = (x_grid - cx) / scale
    yn = -(y_grid - cy) / scale

    theta = np.arctan2(xn, yn)

    # Round head (nearly circular, slightly wider than tall)
    r_base = 0.75 / np.sqrt((np.cos(theta) / 0.82) ** 2 + (np.sin(theta) / 0.82) ** 2)

    # Big pointy triangular ears
    ear_height = 0.40
    ear_spread = 0.58  # angle from top center
    ear_width = 0.09  # narrow = pointy
    r_ears = ear_height * np.exp(
        -((theta - ear_spread) ** 2) / (2 * ear_width**2)
    ) + ear_height * np.exp(-((theta + ear_spread) ** 2) / (2 * ear_width**2))

    # V-dip between ears
    r_dip = -0.22 * np.exp(-(theta**2) / (2 * 0.05**2))

    # Slight chin flattening at bottom
    r_chin = -0.06 * np.exp(-((theta - np.pi) ** 2) / (2 * 0.15**2))
    r_chin += -0.06 * np.exp(-((theta + np.pi) ** 2) / (2 * 0.15**2))

    r_total = r_base + r_ears + r_dip + r_chin
    r_pixel = np.sqrt(xn**2 + yn**2)
    mask = r_pixel < r_total

    mask_float = gaussian_filter(mask.astype(float), sigma=3)
    return mask_float > 0.5


def make_heart_peak(w, h, cx, cy_heart, sx, sy):
    """Create a 2D heart-shaped peak."""
    y_grid, x_grid = np.mgrid[0:h, 0:w]
    xn = (x_grid - cx) / sx
    yn = -(y_grid - cy_heart) / sy

    heart_val = (xn**2 + yn**2 - 1) ** 3 - xn**2 * yn**3
    return 1.0 / (1.0 + np.exp(np.clip(heart_val * 2.0, -500, 500)))


def make_mode_field(w, h, cx, cy, scale):
    """Compute the 2D mode profile inside a cat-shaped boundary."""
    mask = make_cat_mask(w, h, cx, cy, scale)

    dist = distance_transform_edt(mask).astype(float)
    dist_norm = dist / dist.max()

    heart = make_heart_peak(w, h, cx, cy - scale * 0.15, scale * 0.55, scale * 0.45)

    blend = np.clip(dist_norm * 2.5, 0, 1)
    field = dist_norm * (1 - blend * 0.6) + heart * (0.2 + blend * 0.6)

    fmax = field[mask].max()
    field = field / fmax
    field[~mask] = -0.5

    field = gaussian_filter(field, sigma=SMOOTH_SIGMA)
    field[~mask] = -0.5

    return field, mask


def extract_contour_lines(field, levels):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(1, 1))
    cs = ax.contour(field, levels=levels)
    plt.close(fig)
    return [seg_list for seg_list in cs.allsegs]


def simplify_polygon(vs, tolerance=1.5):
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
    if len(vertices) < 3:
        return ""
    vs = np.array(vertices)
    if np.allclose(vs[0], vs[-1]):
        vs = vs[:-1]
    if len(vs) < 3:
        return ""

    n = len(vs)
    if n > 8:
        pad = min(n // 4, 30)
        xs = np.concatenate([vs[-pad:, 0], vs[:, 0], vs[:pad, 0]])
        ys = np.concatenate([vs[-pad:, 1], vs[:, 1], vs[:pad, 1]])
        sigma = max(3.0, n / 60)
        xs = gaussian_filter1d(xs, sigma=sigma)[pad : pad + n]
        ys = gaussian_filter1d(ys, sigma=sigma)[pad : pad + n]
        vs = np.column_stack([xs, ys])

    vs = simplify_polygon(vs, tolerance=1.0)
    n = len(vs)
    if n < 3:
        return ""

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
    # Compute field on generous canvas, then auto-crop
    W, H = W_FIELD, H_FIELD
    cat_cx = W / 2
    cat_cy = H * 0.40
    cat_scale = W * 0.38

    print("Computing mode field...")
    field, mask = make_mode_field(W, H, cat_cx, cat_cy, cat_scale)

    levels = np.linspace(0.10, 0.92, N_LEVELS)

    print("Extracting contours...")
    contour_lines = extract_contour_lines(field, levels)

    # --- Rib waveguide cross-section geometry ---
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    cat_top = rmin
    cat_bottom = rmax
    cat_h = cat_bottom - cat_top

    # Core: tall rect, narrower than cat, extends above ears
    core_w = (cmax - cmin) * 0.72
    core_x = cat_cx - core_w / 2
    core_y = cat_top - 14
    core_bottom = cat_bottom - cat_h * 0.10
    core_h = core_bottom - core_y

    # Slab: wide short rect at bottom of core
    slab_h = cat_h * 0.14
    slab_y = core_bottom - slab_h * 0.30
    slab_w = W * 0.80
    slab_x = (W - slab_w) / 2

    # Cladding
    clad_pad = 14
    clad_x = slab_x - clad_pad
    clad_y = core_y - clad_pad
    clad_w = slab_w + 2 * clad_pad
    clad_h = (slab_y + slab_h) - clad_y + clad_pad

    # Auto-crop: viewBox fits the cladding with small margin
    margin = 12
    vb_x = clad_x - margin
    vb_y = clad_y - margin
    vb_w = clad_w + 2 * margin
    vb_h = clad_h + 2 * margin

    # --- Build SVG ---
    out_w = int(vb_w)
    out_h = int(vb_h)
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{vb_x:.0f} {vb_y:.0f} {vb_w:.0f} {vb_h:.0f}" width="{out_w}" height="{out_h}">',
        f'  <rect x="{vb_x:.0f}" y="{vb_y:.0f}" width="{vb_w:.0f}" height="{vb_h:.0f}" fill="white"/>',
        # Cladding
        f'  <rect x="{clad_x:.1f}" y="{clad_y:.1f}" width="{clad_w:.1f}" height="{clad_h:.1f}" fill="{WG_CLAD}"/>',
        # Core (tall rect)
        f'  <rect x="{core_x:.1f}" y="{core_y:.1f}" width="{core_w:.1f}" height="{core_h:.1f}" fill="{WG_CORE}"/>',
        # Slab (wide rect)
        f'  <rect x="{slab_x:.1f}" y="{slab_y:.1f}" width="{slab_w:.1f}" height="{slab_h:.1f}" fill="{WG_SLAB}"/>',
    ]

    # Contour lines
    for i, polygons in enumerate(contour_lines):
        if i >= len(LINE_COLORS):
            break
        color = LINE_COLORS[i]
        sw = LINE_WIDTHS[i]
        valid = [p for p in polygons if len(p) > 20]
        if not valid:
            continue
        largest = max(valid, key=len)
        path_d = polygon_to_svg_path(largest)
        if path_d:
            svg.append(
                f'  <path d="{path_d}" fill="none" stroke="{color}" '
                f'stroke-width="{sw}" stroke-linejoin="round"/>'
            )
        print(f"  Layer {i}: {color} sw={sw} ({len(largest)} pts)")

    svg.append("</svg>")
    svg_out = "\n".join(svg) + "\n"
    DST.write_text(svg_out)
    print(f"\nWritten {DST} ({len(svg_out)} bytes)")


if __name__ == "__main__":
    main()
