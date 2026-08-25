"""
Generate publication-quality 3D SPD Cone and Riemannian Manifold illustrations
for the HiRes-VIC ICRA paper overview figure.

Outputs:
  - assets/spd_manifold_diagram.png (High-DPI PNG with transparency)
  - assets/spd_manifold_diagram.pdf (Vector PDF for paper inclusion)
  - assets/spd_manifold_diagram.svg (Vector SVG for Draw.io / Inkscape)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patheffects as pe

def generate_spd_figure(output_dir="assets"):
    os.makedirs(output_dir, exist_ok=True)
    
    # -------------------------------------------------------------
    # 1. Figure & Typography Setup
    # -------------------------------------------------------------
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "cm",
        "figure.dpi": 300
    })
    
    fig = plt.figure(figsize=(9.5, 7.5), facecolor="none")
    ax = fig.add_subplot(111, projection="3d", facecolor="none")
    
    ax.set_axis_off()
    ax.grid(False)
    
    # -------------------------------------------------------------
    # 2. SPD Cone Surface (Lorentz Cone in Mandel coordinates)
    #    u = (x+y)/√2 > 0, u^2 > v^2 + w^2
    # -------------------------------------------------------------
    u_vals = np.linspace(0.12, 2.55, 80)
    theta_vals = np.linspace(0, 2 * np.pi, 100)
    U, THETA = np.meshgrid(u_vals, theta_vals)
    
    slope = 0.95
    R = U * slope
    V = R * np.cos(THETA)
    W = R * np.sin(THETA)
    
    # Semi-transparent cone surface
    ax.plot_surface(
        V, W, U,
        cmap="Blues_r",
        alpha=0.18,
        rstride=2,
        cstride=2,
        linewidth=0.25,
        edgecolor="#9ecae1",
        antialiased=True,
        shade=True
    )
    
    # Concentric Riemannian distance contour rings
    for h in [0.75, 1.35, 1.95, 2.50]:
        r_ring = h * slope
        th = np.linspace(0, 2*np.pi, 120)
        ax.plot(
            r_ring * np.cos(th), r_ring * np.sin(th), np.full_like(th, h),
            color="#4292c6", alpha=0.35, lw=0.8, linestyle=":"
        )

    # -------------------------------------------------------------
    # 3. Key Points on the SPD Cone
    # -------------------------------------------------------------
    # K_LLM: Upper Left
    u_LLM = 2.15
    th_LLM = np.pi * 0.88
    r_LLM = u_LLM * slope * 0.82
    pt_K_LLM = np.array([r_LLM * np.cos(th_LLM), r_LLM * np.sin(th_LLM), u_LLM])
    
    # Base Identity I: Lower Front
    u_I = 1.05
    th_I = -np.pi * 0.28
    r_I = u_I * slope * 0.78
    pt_I = np.array([r_I * np.cos(th_I), r_I * np.sin(th_I), u_I])

    # -------------------------------------------------------------
    # 4. Tangent Plane T_I at Point I
    # -------------------------------------------------------------
    t1 = np.array([0.75, 0.40, 0.45])
    t1 = t1 / np.linalg.norm(t1) * 1.05
    t2 = np.array([-0.25, 0.85, 0.35])
    t2 = t2 / np.linalg.norm(t2) * 0.80
    
    c1 = pt_I - 0.55*t1 - 0.55*t2
    c2 = pt_I + 1.25*t1 - 0.55*t2
    c3 = pt_I + 1.25*t1 + 1.15*t2
    c4 = pt_I - 0.55*t1 + 1.15*t2
    
    plane_verts = [[c1, c2, c3, c4]]
    plane = Poly3DCollection(
        plane_verts, alpha=0.36, facecolor="#fed976", edgecolor="#d94801",
        linewidth=1.3, linestyle="--"
    )
    ax.add_collection3d(plane)

    # -------------------------------------------------------------
    # 5. Tangent Vector a_RL ∈ T_I
    # -------------------------------------------------------------
    vec_a = 0.90 * t1 + 0.45 * t2
    pt_tip = pt_I + vec_a
    
    ax.quiver(
        pt_I[0], pt_I[1], pt_I[2],
        vec_a[0], vec_a[1], vec_a[2],
        color="#d94801", lw=3.2, arrow_length_ratio=0.15
    )

    # -------------------------------------------------------------
    # 6. K_RL and Exponential Map (on the Right side)
    # -------------------------------------------------------------
    u_RL = 1.70
    th_RL = np.pi * 0.05
    r_RL = u_RL * slope * 0.86
    pt_K_RL = np.array([r_RL * np.cos(th_RL), r_RL * np.sin(th_RL), u_RL])
    
    # Projection arc from tangent tip to K_RL
    t_proj = np.linspace(0, 1, 35)
    arc_proj = np.outer(1 - t_proj, pt_tip) + np.outer(t_proj, pt_K_RL)
    arc_proj[:, 2] += np.sin(t_proj * np.pi) * 0.16
    ax.plot(arc_proj[:, 0], arc_proj[:, 1], arc_proj[:, 2], color="#cb181d", linestyle="--", lw=2.2)

    # -------------------------------------------------------------
    # 7. Geodesic Curve γ(t) between K_LLM (left) and K_RL (right)
    # -------------------------------------------------------------
    t_geo = np.linspace(0, 1, 80)
    geo_pts = np.zeros((len(t_geo), 3))
    for i, t in enumerate(t_geo):
        p = (1 - t) * pt_K_LLM + t * pt_K_RL
        p[2] -= 0.45 * np.sin(t * np.pi)
        geo_pts[i] = p
        
    ax.plot(geo_pts[:, 0], geo_pts[:, 1], geo_pts[:, 2], color="#6a51a3", lw=4.0, linestyle="-")

    # Blended point K_blend = γ(1 - w)
    idx_blend = int(len(t_geo) * 0.42)
    pt_blend = geo_pts[idx_blend]

    # -------------------------------------------------------------
    # 8. Scatter Key Points
    # -------------------------------------------------------------
    ax.scatter([pt_I[0]], [pt_I[1]], [pt_I[2]], color="#111111", s=85, zorder=20)
    ax.scatter([pt_tip[0]], [pt_tip[1]], [pt_tip[2]], color="#d94801", s=45, zorder=20)
    ax.scatter([pt_K_RL[0]], [pt_K_RL[1]], [pt_K_RL[2]], color="#e6550d", s=120, edgecolors="#7f2704", lw=2, zorder=20)
    ax.scatter([pt_K_LLM[0]], [pt_K_LLM[1]], [pt_K_LLM[2]], color="#08519c", s=120, edgecolors="#02254b", lw=2, zorder=20)
    ax.scatter([pt_blend[0]], [pt_blend[1]], [pt_blend[2]], color="#fec44f", s=220, marker="*", edgecolors="#800026", lw=1.8, zorder=25)

    # -------------------------------------------------------------
    # 9. Clean Annotations with White Halo Stroke
    # -------------------------------------------------------------
    halo = [pe.withStroke(linewidth=4.5, foreground="white")]
    halo_strong = [pe.withStroke(linewidth=6.0, foreground="white")]
    
    # 1. Manifold Name S^3_++ (left flank of the cone)
    ax.text(-2.3, -0.6, 1.75, r"$\mathcal{S}_{++}^3$", fontsize=20, fontweight="bold", color="#08519c", path_effects=halo_strong)
    
    # 2. Tangent Space T_I S^3_++ (upper right corner of tangent plane)
    ax.text(c3[0] - 0.15, c3[1] + 0.35, c3[2] + 0.18, r"$\mathcal{T}_{\mathbf{I}}\mathcal{S}_{++}^3$", fontsize=14, color="#b30000", fontweight="bold", path_effects=halo)
    
    # 3. Identity Base Point I
    ax.text(pt_I[0] - 0.32, pt_I[1] - 0.25, pt_I[2] - 0.18, r"$\mathbf{I}$", fontsize=14, fontweight="bold", color="#111111", path_effects=halo)
    
    # 4. Action a_RL (below the arrow)
    mid_vec = pt_I + 0.45 * vec_a
    ax.text(mid_vec[0] + 0.12, mid_vec[1] - 0.32, mid_vec[2] - 0.08, r"$\mathbf{a}_{\mathrm{RL}} \in \mathbb{R}^6$", fontsize=12, color="#d94801", fontweight="bold", path_effects=halo)
    
    # 5. Exponential Map Exp_I(.) (above projection arc)
    mid_proj = arc_proj[18]
    ax.text(mid_proj[0] + 0.18, mid_proj[1] + 0.12, mid_proj[2] + 0.20, r"$\mathrm{Exp}_{\mathbf{I}}(\cdot)$", fontsize=12, color="#cb181d", fontweight="bold", path_effects=halo)
    
    # 6. K_RL (right side)
    ax.text(pt_K_RL[0] + 0.25, pt_K_RL[1] + 0.20, pt_K_RL[2] - 0.10, r"$\mathbf{K}_{\mathrm{RL}}$", fontsize=14, fontweight="bold", color="#a63603", path_effects=halo)
    
    # 7. K_LLM (above top-left point)
    ax.text(pt_K_LLM[0] - 0.25, pt_K_LLM[1] + 0.25, pt_K_LLM[2] + 0.20, r"$\mathbf{K}_{\mathrm{LLM}}$", fontsize=14, fontweight="bold", color="#08519c", path_effects=halo)
    
    # 8. Geodesic Curve Label gamma(t)
    mid_geo_label = geo_pts[18]
    ax.text(mid_geo_label[0] - 0.35, mid_geo_label[1] - 0.25, mid_geo_label[2] + 0.22, r"$\gamma(t)$", fontsize=13, fontweight="bold", color="#54278f", path_effects=halo)
    
    # 9. Blended Point K_blend
    ax.text(pt_blend[0] - 0.55, pt_blend[1] - 0.35, pt_blend[2] + 0.28, r"$\mathbf{K}_{\mathrm{blend}} = \gamma(1-w)$", fontsize=12, fontweight="bold", color="#4a1486", path_effects=halo)

    # -------------------------------------------------------------
    # 10. View Angle & Rendering
    # -------------------------------------------------------------
    ax.view_init(elev=22, azim=-60)
    
    ax.set_xlim([-2.5, 2.5])
    ax.set_ylim([-2.5, 2.5])
    ax.set_zlim([0, 2.8])
    
    plt.tight_layout()
    
    png_path = os.path.join(output_dir, "spd_manifold_diagram.png")
    pdf_path = os.path.join(output_dir, "spd_manifold_diagram.pdf")
    svg_path = os.path.join(output_dir, "spd_manifold_diagram.svg")
    
    plt.savefig(png_path, dpi=400, bbox_inches="tight", transparent=True)
    plt.savefig(pdf_path, bbox_inches="tight", transparent=True)
    plt.savefig(svg_path, bbox_inches="tight", transparent=True)
    plt.close()
    
    print("Done!")

if __name__ == "__main__":
    generate_spd_figure()
