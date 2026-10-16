"""Minimal URDF -> MuJoCo MJCF converter for DeepRobotics M20 (temporary build script)."""
import math
import os
import re
import shutil
import xml.etree.ElementTree as ET

URDF = "/home/ancn/rl_sar/src/rl_sar_zoo/M20_description/urdf/m20.urdf"
OUT_DIR = "/home/ancn/rl_sar/src/rl_sar_zoo/M20_description/mjcf"
ASSET_DIR = os.path.join(OUT_DIR, "assets")
MESH_SRC = "/home/ancn/rl_sar/src/rl_sar_zoo/M20_description/meshes"

# MuJoCo rotates a hinge by (qpos - ref), NOT by qpos. Verified with mj_forward on
# a scratch mjData: with ref="-0.6" the hipy link keeps its unrotated URDF offset
# when qpos == -0.6, and is rotated by +0.6 rad when qpos == 0. So ref silently
# offsets the whole kinematics: qpos == ref means "the URDF zero pose", the legs
# become unable to move the base and the robot stands at the ZERO-pose height
# (0.590) instead of the standing height (0.527) - this is the root cause of
# "站姿高度偏低" in M20_DEPLOY_NOTES.md. go2w / b2w MJCF carry no ref either.
# => no ref anywhere; qpos0 is the zero pose and BASE_Z must match it.
# FK (parsed from m20.urdf, wheel radius 0.09):
#   zero pose     wheel centre z = -0.5000 -> base 0.590
#   standing pose wheel centre z = -0.4366 -> base 0.527
#   lying pose    wheel centre z = -0.3466 -> base 0.437
BASE_Z = 0.590  # zero pose: wheels exactly on the ground at spawn
STANDING_Z = 0.527
LYING_Z = 0.437
SPAWN_POSE = {"fl_hipx_joint": 0.0, "fl_hipy_joint": -0.6, "fl_knee_joint": 1.0,
              "fr_hipx_joint": 0.0, "fr_hipy_joint": -0.6, "fr_knee_joint": 1.0,
              "hl_hipx_joint": 0.0, "hl_hipy_joint":  0.6, "hl_knee_joint": -1.0,
              "hr_hipx_joint": 0.0, "hr_hipy_joint":  0.6, "hr_knee_joint": -1.0,
              "fl_wheel_joint": 0.0, "fr_wheel_joint": 0.0,
              "hl_wheel_joint": 0.0, "hr_wheel_joint": 0.0}

LEG_JOINTS = [
    "fl_hipx_joint", "fl_hipy_joint", "fl_knee_joint",
    "fr_hipx_joint", "fr_hipy_joint", "fr_knee_joint",
    "hl_hipx_joint", "hl_hipy_joint", "hl_knee_joint",
    "hr_hipx_joint", "hr_hipy_joint", "hr_knee_joint",
]
WHEEL_JOINTS = ["fl_wheel_joint", "fr_wheel_joint", "hl_wheel_joint", "hr_wheel_joint"]
POLICY_JOINTS = LEG_JOINTS + WHEEL_JOINTS


def f(v):
    return f"{v:.6g}"


def vec(vals):
    return " ".join(f(v) for v in vals)


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return [[1, 0, 0], [0, c, -s], [0, s, c]]


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return [[c, 0, s], [0, 1, 0], [-s, 0, c]]


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return [[c, -s, 0], [s, c, 0], [0, 0, 1]]


def matmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def rpy_to_quat(r, p, y):
    # URDF rpy: R = Rz(y) * Ry(p) * Rx(r)
    m = matmul(rot_z(y), matmul(rot_y(p), rot_x(r)))
    tr = m[0][0] + m[1][1] + m[2][2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        qw = 0.25 * s
        qx = (m[2][1] - m[1][2]) / s
        qy = (m[0][2] - m[2][0]) / s
        qz = (m[1][0] - m[0][1]) / s
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2
        qw = (m[2][1] - m[1][2]) / s
        qx = 0.25 * s
        qy = (m[0][1] + m[1][0]) / s
        qz = (m[0][2] + m[2][0]) / s
    elif m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2
        qw = (m[0][2] - m[2][0]) / s
        qx = (m[0][1] + m[1][0]) / s
        qy = 0.25 * s
        qz = (m[1][2] + m[2][1]) / s
    else:
        s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2
        qw = (m[1][0] - m[0][1]) / s
        qx = (m[0][2] + m[2][0]) / s
        qy = (m[1][2] + m[2][1]) / s
        qz = 0.25 * s
    return [qw, qx, qy, qz]  # MuJoCo order: w x y z


def parse_origin(el):
    if el is None:
        return [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]
    xyz = [float(x) for x in el.get("xyz", "0 0 0").split()]
    rpy = [float(x) for x in el.get("rpy", "0 0 0").split()]
    return xyz, rpy_to_quat(*rpy)


def mat_to_quat(m):
    tr = m[0][0] + m[1][1] + m[2][2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        qw = 0.25 * s
        qx = (m[2][1] - m[1][2]) / s
        qy = (m[0][2] - m[2][0]) / s
        qz = (m[1][0] - m[0][1]) / s
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2
        qw = (m[2][1] - m[1][2]) / s
        qx = 0.25 * s
        qy = (m[0][1] + m[1][0]) / s
        qz = (m[0][2] + m[2][0]) / s
    elif m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2
        qw = (m[0][2] - m[2][0]) / s
        qx = (m[0][1] + m[1][0]) / s
        qy = 0.25 * s
        qz = (m[1][2] + m[2][1]) / s
    else:
        s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2
        qw = (m[1][0] - m[0][1]) / s
        qx = (m[0][2] + m[2][0]) / s
        qy = (m[1][2] + m[2][1]) / s
        qz = 0.25 * s
    return [qw, qx, qy, qz]


def principal_inertia(quat_origin, inert):
    """Return (quat, [ixx, iyy, izz]) with inertia expressed in principal axes."""
    import numpy as np
    ixx, iyy, izz, ixy, ixz, iyz = inert
    I = np.array([[ixx, ixy, ixz],
                  [ixy, iyy, iyz],
                  [ixz, iyz, izz]])
    lam, V = np.linalg.eigh(I)
    if np.linalg.det(V) < 0:
        V[:, 0] *= -1
    # rotation of inertia frame within link frame
    R_o = quat_to_mat(quat_origin)
    R_total = R_o @ V
    return mat_to_quat(R_total.tolist()), [float(x) for x in lam]


def quat_to_mat(q):
    w, x, y, z = q
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]


def nums(text, n=3):
    return [float(x) for x in text.split()[:n]]


tree = ET.parse(URDF)
root = tree.getroot()

links = {}
for link in root.findall("link"):
    name = link.get("name")
    info = {"inertial": None, "visuals": [], "collisions": []}
    inertial = link.find("inertial")
    if inertial is not None:
        pos, quat = parse_origin(inertial.find("origin"))
        mass = float(inertial.find("mass").get("value"))
        inertia = inertial.find("inertia")
        ixx = float(inertia.get("ixx")); iyy = float(inertia.get("iyy")); izz = float(inertia.get("izz"))
        ixy = float(inertia.get("ixy")); ixz = float(inertia.get("ixz")); iyz = float(inertia.get("iyz"))
        info["inertial"] = (pos, quat, mass, [ixx, iyy, izz, ixy, ixz, iyz])
    for vis in link.findall("visual"):
        pos, quat = parse_origin(vis.find("origin"))
        mesh = vis.find("geometry/mesh")
        if mesh is None:
            continue
        info["visuals"].append((os.path.basename(mesh.get("filename")), pos, quat))
    for col in link.findall("collision"):
        pos, quat = parse_origin(col.find("origin"))
        geo = col.find("geometry")
        if geo.find("box") is not None:
            size = nums(geo.find("box").get("size"))
            info["collisions"].append(("box", [size[0] / 2, size[1] / 2, size[2] / 2], pos, quat))
        elif geo.find("cylinder") is not None:
            cyl = geo.find("cylinder")
            r = float(cyl.get("radius")); l = float(cyl.get("length"))
            info["collisions"].append(("cylinder", [r, l / 2], pos, quat))
        elif geo.find("sphere") is not None:
            r = float(geo.find("sphere").get("radius"))
            info["collisions"].append(("sphere", [r], pos, quat))
    links[name] = info

joints = {}
children = {}
for j in root.findall("joint"):
    name = j.get("name")
    parent = j.find("parent").get("link")
    child = j.get("link") if False else j.find("child").get("link")
    pos, quat = parse_origin(j.find("origin"))
    axis = [float(x) for x in j.find("axis").get("xyz", "1 0 0").split()]
    jtype = j.get("type")
    lim = j.find("limit")
    rng = None
    effort = None
    if lim is not None:
        effort = float(lim.get("effort")) if lim.get("effort") else None
        if lim.get("lower") is not None and lim.get("upper") is not None:
            rng = (float(lim.get("lower")), float(lim.get("upper")))
    joints[name] = dict(parent=parent, child=child, pos=pos, quat=quat, axis=axis,
                        type=jtype, range=rng, effort=effort)
    children.setdefault(parent, []).append(name)

root_link = None
for lname in links:
    if not any(j["child"] == lname for j in joints.values()):
        root_link = lname
        break

mesh_files = sorted({m for info in links.values() for m, _, _ in info["visuals"]})
os.makedirs(ASSET_DIR, exist_ok=True)
for mf in mesh_files:
    src = os.path.join(MESH_SRC, mf)
    if os.path.exists(src):
        shutil.copy(src, os.path.join(ASSET_DIR, mf))

L = []


def indent(depth):
    return "  " * depth


def emit_body(link_name, depth):
    info = links[link_name]
    is_root = link_name == root_link
    if is_root:
        L.append(f'{indent(depth)}<body name="{link_name}" pos="0 0 {f(BASE_Z)}" childclass="M20">')
    else:
        pos, quat = None, None
        for jn, j in joints.items():
            if j["child"] == link_name:
                pos, quat = j["pos"], j["quat"]
                break
        L.append(f'{indent(depth)}<body name="{link_name}" pos="{vec(pos)}" quat="{vec(quat)}">')

    if is_root:
        L.append(f'{indent(depth+1)}<freejoint />')
    else:
        jn = [n for n, j in joints.items() if j["child"] == link_name][0]
        j = joints[jn]
        # NOTE: no ref= attribute on purpose - MuJoCo would rotate by (qpos - ref).
        attrs = f'name="{jn}" type="hinge" axis="{vec(j["axis"])}"'
        if j["range"]:
            attrs += f' range="{f(j["range"][0])} {f(j["range"][1])}"'
        else:
            attrs += ' limited="false"'
        if jn in WHEEL_JOINTS:
            attrs += ' damping="0.0" armature="0.0" frictionloss="0.0"'
        L.append(f'{indent(depth+1)}<joint {attrs} />')

    if info["inertial"]:
        pos, quat, mass, inert = info["inertial"]
        pquat, diag = principal_inertia(quat, inert)
        L.append(f'{indent(depth+1)}<inertial pos="{vec(pos)}" quat="{vec(pquat)}" mass="{f(mass)}" '
                 f'diaginertia="{vec(diag)}" />')

    for meshfile, pos, quat in info["visuals"]:
        mesh_name = os.path.splitext(meshfile)[0]
        L.append(f'{indent(depth+1)}<geom mesh="{mesh_name}" material="gray" class="visual" '
                 f'pos="{vec(pos)}" quat="{vec(quat)}" />')

    for ci, (gtype, size, pos, quat) in enumerate(info["collisions"]):
        cls = "wheel" if (gtype == "cylinder" and link_name.endswith("wheel")) else "collision"
        L.append(f'{indent(depth+1)}<geom name="col_{link_name}_{ci}" type="{gtype}" size="{vec(size)}" pos="{vec(pos)}" '
                 f'quat="{vec(quat)}" class="{cls}" />')

    if is_root:
        L.append(f'{indent(depth+1)}<site name="imu" pos="0 0 0" />')

    for jn in children.get(link_name, []):
        emit_body(joints[jn]["child"], depth + 1)
    L.append(f'{indent(depth)}</body>')


L.append('<mujoco model="M20">')
L.append('  <compiler angle="radian" meshdir="assets" autolimits="true" />')
L.append('  <option cone="elliptic" impratio="100" />')
L.append('  <default>')
L.append('    <default class="M20">')
# wheel-ground traction is what carries a wheeled policy; 0.8 was too slippery to
# stand reliably, 1.2 is the value tuned in M20_DEPLOY_NOTES.md (kept here so that
# regenerating M20.xml does not silently revert the manual tuning).
L.append('      <geom friction=\"1.2 0.05 0.02\" margin=\"0.001\" condim=\"3\" />')
# Isaac DCMotor sets friction=0.0 and provides all damping through the actuator,
# so the joint itself must carry no damping / frictionloss.
L.append('      <joint damping="0.0" armature="0.0" frictionloss="0.0" />')
L.append('      <default class="visual">')
L.append('        <geom type="mesh" contype="0" conaffinity="0" group="2" />')
L.append('      </default>')
L.append('      <default class="collision">')
L.append('        <geom group="3" />')
L.append('      </default>')
L.append('      <default class="wheel">')
L.append('        <geom group="3" condim="3" friction="1.2 0.05 0.02" priority="1" />')
L.append('      </default>')
L.append('    </default>')
L.append('  </default>')
L.append('  <asset>')
L.append('    <material name="gray" rgba="0.75 0.75 0.75 1" />')
for mf in mesh_files:
    L.append(f'    <mesh file="{mf}" />')
L.append('  </asset>')
L.append('  <worldbody>')
emit_body(root_link, 2)
L.append('  </worldbody>')

# joint index order used by MuJoCo = depth-first order of the kinematic tree.
# rl_sar indexes both mj_data->ctrl and sensordata via joint_mapping[policy_index],
# so <actuator> and the joint* <sensor> lists MUST be emitted in this MJCF order.
mjcf_order = []


def walk(link_name):
    for jn in children.get(link_name, []):
        mjcf_order.append(jn)
        walk(joints[jn]["child"])


walk(root_link)
mapping = [mjcf_order.index(jn) for jn in POLICY_JOINTS]
print("MJCF joint order:", mjcf_order)
print("joint_mapping (policy -> mjcf):", mapping)

L.append('  <actuator>')
for jn in mjcf_order:
    j = joints[jn]
    eff = j["effort"] if j["effort"] else 20.0
    L.append(f'    <motor name="{jn}" joint="{jn}" ctrlrange="-{f(eff)} {f(eff)}" />')
L.append('  </actuator>')

L.append('  <sensor>')
for jn in mjcf_order:
    L.append(f'    <jointpos name="{jn}_pos" joint="{jn}" />')
for jn in mjcf_order:
    L.append(f'    <jointvel name="{jn}_vel" joint="{jn}" />')
for jn in mjcf_order:
    L.append(f'    <jointactuatorfrc name="{jn}_torque" joint="{jn}" noise="0.01" />')
L.append('    <framequat name="imu_quat" objtype="site" objname="imu" />')
L.append('    <gyro name="imu_gyro" site="imu" />')
L.append('    <accelerometer name="imu_acc" site="imu" />')
L.append('    <framepos name="frame_pos" objtype="site" objname="imu" />')
L.append('    <framelinvel name="frame_vel" objtype="site" objname="imu" />')
L.append('  </sensor>')

default_dof = [0.0, -0.6, 1.0, 0.0, -0.6, 1.0, 0.0, 0.6, -1.0, 0.0, 0.6, -1.0, 0.0, 0.0, 0.0, 0.0]
# qpos / ctrl in the keyframe follow the MJCF joint order
default_dof_mjcf = [default_dof[mapping.index(k)] for k in range(len(mjcf_order))]
crouch_dof = [0.0, -0.9, 1.6, 0.0, -0.9, 1.6, 0.0, 0.9, -1.6, 0.0, 0.9, -1.6, 0.0, 0.0, 0.0, 0.0]
crouch_mjcf = [crouch_dof[mapping.index(k)] for k in range(len(mjcf_order))]
L.append('  <keyframe>')
L.append(f'    <key name="home" qpos="0 0 {f(STANDING_Z)} 1 0 0 0 {vec(default_dof_mjcf)}" '
         f'ctrl="{vec(default_dof_mjcf)}" />')
L.append(f'    <key name="lying" qpos="0 0 {f(LYING_Z)} 1 0 0 0 {vec(crouch_mjcf)}" '
         f'ctrl="{vec(crouch_mjcf)}" />')
L.append('  </keyframe>')
L.append('</mujoco>')

with open(os.path.join(OUT_DIR, "M20.xml"), "w") as fh:
    fh.write("\n".join(L) + "\n")

print("wrote", os.path.join(OUT_DIR, "M20.xml"))
print("base link:", root_link, "links:", len(links), "joints:", len(joints))
