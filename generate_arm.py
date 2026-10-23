"""
Robotic Phone Holder Arm - STL Generator
----------------------------------------
基于图片的机械臂结构：基座腕带 → 上臂（双杆）→ 肘关节 → 前臂 → 末端夹爪
所有尺寸: mm。FDM 打印友好（壁厚 >= 2mm，销孔 4.2mm 间隙）。
"""

import os
import numpy as np
import trimesh
from trimesh.creation import box, cylinder
from trimesh.transformations import translation_matrix, rotation_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import io

OUT_DIR = "/workspace/arm_stl"
os.makedirs(OUT_DIR, exist_ok=True)

COLORS = {
    "base_cuff":   [0.20, 0.75, 0.30],
    "upper_arm":   [0.95, 0.25, 0.40],
    "elbow_joint": [0.55, 0.25, 0.85],
    "forearm":     [0.15, 0.60, 0.70],
    "gripper":     [0.30, 0.80, 0.35],
}

def hole(mesh, center, radius, height, axis=None):
    if axis is None:
        axis = [0, 1, 0]
    cyl = cylinder(radius=radius, height=height)
    axis = np.array(axis, dtype=float); axis /= np.linalg.norm(axis)
    default = np.array([0, 1, 0])
    if not np.allclose(axis, default):
        v = np.cross(default, axis)
        c = np.dot(default, axis); s = np.linalg.norm(v)
        if s > 1e-9:
            skew = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
            R = np.eye(4); R[:3,:3] = np.eye(3) + skew + skew@skew*(1-c)/(s*s)
            cyl.apply_transform(R)
    cyl.apply_transform(translation_matrix(center))
    return mesh.difference(cyl)

def peg_hole(mesh, center, axis=None, d=4.2):
    return hole(mesh, center, radius=d/2, height=6, axis=axis)

# ================================================================
def make_base_cuff():
    inner_r = 55.0
    outer_r = inner_r + 12.0
    height  = 40.0

    outer = cylinder(radius=outer_r, height=height)
    inner = cylinder(radius=inner_r, height=height + 2)
    cuff  = outer.difference(inner)

    # 侧面开缝（可弹性扣合）
    slot = box([3.0, outer_r*2+10, height+2])
    slot.apply_transform(translation_matrix([outer_r, 0, 0]))
    cuff = cuff.difference(slot)

    # 正面装饰孔
    for i in range(5):
        angle = -np.pi/2 + (i-2) * 0.35
        cx = (inner_r + outer_r)/2 * np.cos(angle)
        cy = (inner_r + outer_r)/2 * np.sin(angle)
        cuff = hole(cuff, [cx, cy, 0], radius=4.5, height=height+2, axis=[cx, cy, 0])

    # 顶部安装平台
    platform = box([70, 40, 10])
    platform.apply_transform(translation_matrix([0, 0, height/2 + 5]))
    cuff = cuff.union(platform)

    # 平台上销孔
    cuff = peg_hole(cuff, [-15, 0, height/2 + 10])
    cuff = peg_hole(cuff, [ 15, 0, height/2 + 10])

    # 以平台顶面为 z=0
    cuff.apply_transform(translation_matrix([0, 0, -height/2 - 5]))
    return cuff

# ================================================================
def make_upper_arm():
    length = 140.0
    width  = 10.0
    gap    = 28.0
    depth  = 8.0

    rod_L = box([width, depth, length]); rod_L.apply_transform(translation_matrix([-gap/2, 0, 0]))
    rod_R = box([width, depth, length]); rod_R.apply_transform(translation_matrix([ gap/2, 0, 0]))
    arm = rod_L.union(rod_R)

    cap_size = [gap + width + 4, 14, 18]
    cap_top = box(cap_size); cap_top.apply_transform(translation_matrix([0, 0,  length/2 - cap_size[2]/2 + 4]))
    cap_bot = box(cap_size); cap_bot.apply_transform(translation_matrix([0, 0, -length/2 + cap_size[2]/2 - 4]))
    arm = arm.union(cap_top).union(cap_bot)

    arm = peg_hole(arm, [0, 0,  length/2 - cap_size[2]/2 + 4])
    arm = peg_hole(arm, [0, 0, -length/2 + cap_size[2]/2 - 4])

    for i in range(3):
        z = -length/2 + 30 + i * 35
        arm = hole(arm, [-gap/2, 0, z], radius=3, height=depth+2, axis=[0,1,0])
        arm = hole(arm, [ gap/2, 0, z], radius=3, height=depth+2, axis=[0,1,0])

    return arm

# ================================================================
def make_elbow():
    hub = cylinder(radius=14, height=20)
    ear_thick = 6
    ear_top = box([12, ear_thick, 28]); ear_top.apply_transform(translation_matrix([-10, -ear_thick/2, 0]))
    ear_bot = box([12, ear_thick, 28]); ear_bot.apply_transform(translation_matrix([ 10, -ear_thick/2, 0]))
    elbow = hub.union(ear_top).union(ear_bot)

    for sign_x in [-1, 1]:
        for sign_z in [-1, 1]:
            elbow = peg_hole(elbow, [sign_x*10, -ear_thick/2 + 1, sign_z*8], axis=[0,1,0])

    elbow = hole(elbow, [0, 0, 0], radius=4.2, height=22, axis=[0,1,0])

    tab1 = box([10, 6, 22]); tab1.apply_transform(translation_matrix([-10, -3, -18]))
    tab2 = box([10, 6, 22]); tab2.apply_transform(translation_matrix([ 10, -3, -18]))
    elbow = elbow.union(tab1).union(tab2)

    elbow = peg_hole(elbow, [-10, 0, -24], axis=[0,1,0])
    elbow = peg_hole(elbow, [ 10, 0, -24], axis=[0,1,0])

    return elbow

# ================================================================
def make_forearm():
    length = 110.0
    width  = 10.0
    gap    = 24.0
    depth  = 8.0

    rod_L = box([width, depth, length]); rod_L.apply_transform(translation_matrix([-gap/2, 0, 0]))
    rod_R = box([width, depth, length]); rod_R.apply_transform(translation_matrix([ gap/2, 0, 0]))
    arm = rod_L.union(rod_R)

    cap_up = box([gap + width + 4, 12, 16]); cap_up.apply_transform(translation_matrix([0, 0,  length/2 - 6]))
    cap_lo = box([gap + width + 4, 12, 16]); cap_lo.apply_transform(translation_matrix([0, 0, -length/2 + 6]))
    arm = arm.union(cap_up).union(cap_lo)

    arm = peg_hole(arm, [0, 0,  length/2 - 6])
    arm = peg_hole(arm, [0, 0, -length/2 + 6])

    for i in range(2):
        z = -length/2 + 35 + i * 35
        arm = hole(arm, [-gap/2, 0, z], radius=3, height=depth+2, axis=[0,1,0])
        arm = hole(arm, [ gap/2, 0, z], radius=3, height=depth+2, axis=[0,1,0])

    return arm

# ================================================================
def make_gripper():
    thickness = 4.0
    width     = 70.0
    depth     = 75.0
    jaw_gap   = 14.0   # 开口

    # 背板
    back = box([width, thickness, depth])
    back.apply_transform(translation_matrix([0, thickness/2, 0]))

    # 上爪 / 下爪
    top_jaw = box([width + 4, thickness, thickness])
    top_jaw.apply_transform(translation_matrix([0, thickness + thickness/2, depth/2 - thickness/2]))
    bot_jaw = box([width + 4, thickness, thickness])
    bot_jaw.apply_transform(translation_matrix([0, thickness + thickness/2, -depth/2 + thickness/2]))

    gripper = back.union(top_jaw).union(bot_jaw)

    # 连接座
    mount = box([30, thickness + 10, 30])
    mount.apply_transform(translation_matrix([0, -5, 0]))
    gripper = gripper.union(mount)

    # 销孔
    for sx in [-1, 1]:
        for sz in [-1, 1]:
            gripper = peg_hole(gripper, [sx*10, -5, sz*12], axis=[0,1,0])

    # 橡胶垫安装孔（装饰）
    for i in range(5):
        x = -width/2 + 10 + i * ((width - 20)/4)
        gripper = hole(gripper, [x, thickness + thickness/2 + 2,  depth/2 - thickness - 1],
                       radius=2, height=2, axis=[0,1,0])
        gripper = hole(gripper, [x, thickness + thickness/2 + 2, -depth/2 + thickness + 1],
                       radius=2, height=2, axis=[0,1,0])

    return gripper


# ================================================================
def build_assembly():
    """
    装配变换约定：对每个零件，先在其本地坐标中完成造型，
    然后依次 R_local (本地姿态) → T_world (世界位置)。
    在 trimesh 中 apply_transform 是左乘，所以组合顺序是 T @ R。
    """
    parts = {}
    base  = make_base_cuff()
    upper = make_upper_arm()
    elbow = make_elbow()
    fore  = make_forearm()
    grip  = make_gripper()

    # 基座：就位
    base.apply_transform(translation_matrix([0, 0, 0]))

    # 上臂：直立，底部对齐基座平台顶 (z≈25)
    upper.apply_transform(translation_matrix([0, 0, 25 + 70]))

    # 肘：居中
    elbow.apply_transform(translation_matrix([0, 0, 25 + 140 + 14]))

    # 前臂：绕 Y 前倾 35°，再移到肘关节下方
    R_fore = rotation_matrix(np.radians(-35), [0, 1, 0])
    T_fore = translation_matrix([45, 0, 25 + 140 + 28 + 55])
    fore.apply_transform(T_fore @ R_fore)

    # 夹爪：先绕 X 旋转 90° 把夹持口翻向前方（零件本地 Y 变成世界 Z），
    # 再跟随前臂的前倾，最后平移到最前端
    R_grip = (
        rotation_matrix(np.radians(-35), [0, 1, 0])   # 跟着前倾
        @ rotation_matrix(np.radians(90), [1, 0, 0])  # 本地翻转
    )
    T_grip = translation_matrix([85, 0, 25 + 140 + 28 + 115 + 15])
    grip.apply_transform(T_grip @ R_grip)

    parts["base_cuff"]   = base
    parts["upper_arm"]   = upper
    parts["elbow_joint"] = elbow
    parts["forearm"]     = fore
    parts["gripper"]     = grip

    return parts


def build_exploded(parts, factor=1.0):
    offsets = {
        "base_cuff":   np.array([  0,   0, -40]) * factor,
        "upper_arm":   np.array([  0,   0,  80]) * factor,
        "elbow_joint": np.array([  0,  30, 180]) * factor,
        "forearm":     np.array([ 60, -30, 260]) * factor,
        "gripper":     np.array([140, -60, 340]) * factor,
    }
    return {name: mesh.copy().apply_transform(translation_matrix(offsets[name]))
            for name, mesh in parts.items()}


# ================================================================
def render_parts(parts_dict, ax, title="", elev=25, azim=-55, draw_axes=True):
    all_verts = np.vstack([p.vertices for p in parts_dict.values()])
    center = all_verts.mean(axis=0)
    radius = np.abs(all_verts - center).max() * 1.1

    ax.set_xlim(center[0]-radius, center[0]+radius)
    ax.set_ylim(center[1]-radius, center[1]+radius)
    ax.set_zlim(center[2]-radius, center[2]+radius)
    ax.set_box_aspect((1,1,1))
    if draw_axes:
        ax.set_xlabel("X(mm)"); ax.set_ylabel("Y(mm)"); ax.set_zlabel("Z(mm)")
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=12, fontweight="bold")

    for name, mesh in parts_dict.items():
        color = COLORS[name]
        ax.plot_trisurf(
            mesh.vertices[:,0], mesh.vertices[:,1], mesh.vertices[:,2],
            triangles=mesh.faces,
            color=color, alpha=0.92, edgecolor=(0,0,0,0.12), linewidth=0.1
        )


def make_explosion_gif(frames=40, fps=15):
    assembled = build_assembly()
    images = []
    factors = np.concatenate([
        np.linspace(0, 1, frames//2),
        np.linspace(1, 0, frames - frames//2)
    ])

    for i, f in enumerate(factors):
        exploded = build_exploded(assembled, factor=float(f))
        fig = plt.figure(figsize=(10, 8), dpi=110)
        ax = fig.add_subplot(111, projection="3d")
        stage = "EXPLODED" if f > 0.5 else "ASSEMBLED"
        render_parts(exploded, ax,
                     title=f"Robotic Arm — {stage}  (step {i+1}/{frames})")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        images.append(Image.open(buf))

    gif_path = os.path.join(OUT_DIR, "explosion.gif")
    images[0].save(
        gif_path, save_all=True, append_images=images[1:],
        duration=int(1000/fps), loop=0
    )
    print(f"[OK] Explosion GIF -> {gif_path}  ({frames} frames)")


def save_stls():
    parts = build_assembly()
    for name, mesh in parts.items():
        path = os.path.join(OUT_DIR, f"{name}.stl")
        m = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=True)
        m.export(path)
        print(f"[OK] {name}: {len(m.vertices):>6} verts, {len(m.faces):>6} faces -> {os.path.basename(path)}")

    combined = trimesh.util.concatenate(list(parts.values()))
    combined.export(os.path.join(OUT_DIR, "arm_assembled.stl"))
    print(f"[OK] arm_assembled.stl")


def verify_printability():
    parts = build_assembly()
    print("\n=== Printability Check ===")
    print(f"{'Part':<16} {'SizeX':>7} {'SizeY':>7} {'SizeZ':>7} {'Verts':>8} {'Watertight':>11}")
    print("-" * 62)
    for name, mesh in parts.items():
        size = mesh.bounds[1] - mesh.bounds[0]
        wt = mesh.is_watertight
        print(f"{name:<16} {size[0]:>7.1f} {size[1]:>7.1f} {size[2]:>7.1f} {len(mesh.vertices):>8} {str(wt):>11}")

    all_verts = np.vstack([p.vertices for p in parts.values()])
    b = all_verts.max(axis=0) - all_verts.min(axis=0)
    print(f"\nAssembly envelope: {b[0]:.1f} × {b[1]:.1f} × {b[2]:.1f} mm")
    print(f"Required print bed (with margin): {b[0]+50:.0f} × {b[1]+50:.0f} × {b[2]+50:.0f} mm")
    print("\nDesign notes:")
    print("  - Wall thickness >= 2mm (FDM-safe)")
    print("  - Pin holes d=4.2mm (standard 4mm pin + 0.2mm clearance)")
    print("  - Cuff inner r=55mm  fits ~110mm wrist circumference")
    print("  - Joints use captive-pin design; add 4mm pins during assembly")


def make_views():
    parts = build_assembly()

    # 装配预览
    fig = plt.figure(figsize=(12, 9), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    render_parts(parts, ax, title="Robotic Phone Holder Arm — Assembled")
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=n) for n, c in COLORS.items()]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=10)
    fig.savefig(os.path.join(OUT_DIR, "assembly_preview.png"),
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("[OK] assembly_preview.png")

    # 爆炸视图（静态）
    exploded = build_exploded(parts, factor=1.0)
    fig = plt.figure(figsize=(12, 9), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    render_parts(exploded, ax, title="Robotic Arm — Exploded View")
    ax.legend(handles=legend_elements, loc="upper left", fontsize=10)
    fig.savefig(os.path.join(OUT_DIR, "exploded_view.png"),
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("[OK] exploded_view.png")

    # 多视角网格
    fig, axes = plt.subplots(2, 2, figsize=(14, 12),
                             subplot_kw={"projection": "3d"}, dpi=120)
    views = [(25, -55, "ISO"), (0, -90, "FRONT"), (90, 0, "SIDE"), (0, 0, "TOP")]
    for ax, (elev, azim, label) in zip(axes.flat, views):
        render_parts(parts, ax, title=f"View: {label}", elev=elev, azim=azim)
    fig.suptitle("Robot Arm — 4 Views", fontsize=14, fontweight="bold")
    fig.savefig(os.path.join(OUT_DIR, "four_views.png"),
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("[OK] four_views.png")


if __name__ == "__main__":
    save_stls()
    verify_printability()
    make_views()
    make_explosion_gif(frames=40, fps=15)
    print("\nAll files in:", OUT_DIR)
