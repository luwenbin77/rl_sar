"""URDF -> MuJoCo MJCF converter for the ancnw wheeled quadruped.

Adapted from policy/M20/tools_urdf2mjcf.py. Regenerate with:

    python3 policy/ancnw/tools_urdf2mjcf.py

Input : policy/ancnw/ancnw_description/urdf/ancnw.urdf (+ meshes/)
Output: src/rl_sar_zoo/ancnw_description/mjcf/ancnw.xml  (scene.xml is handwritten)

Conventions (same as M20, see M20_DEPLOY_NOTES.md):
  * <actuator> and the joint* <sensor> lists are emitted in MJCF (depth-first
    kinematic-tree) order, because rl_sim_mujoco.cpp indexes mj_data->ctrl and
    mj_data->sensordata through joint_mapping[policy_index].
  * POLICY order (what the .pt actor sees, indices 0..15) is
      0..11  12 leg joints   fl(hip,thigh,calf), fr(...), hl(...), hr(...)
      12..15 4 wheel joints  fl_wheel, fr_wheel, hl_wheel, hr_wheel
    The URDF/MJCF tree order interleaves the wheels (3,7,11,15), so
    joint_mapping = [0,1,2,4,5,6,8,9,10,12,13,14,3,7,11,15].
  * DO NOT emit joint/@ref. MuJoCo rotates a hinge by (qpos - ref), not by qpos
    (verified with mj_forward on a scratch mjData: with ref="0.6" the thigh link
    keeps its unrotated URDF offset when qpos==0.6, and is rotated by -0.6 rad
    when qpos==0). So ref silently offsets the whole kinematics and the robot
    stands ~0.12 m too low with "dead" joints - the legs cannot move the base at
    all and the robot behaves like a rigid cart. go2w/b2w MJCF carry no ref.
  * Consequently qpos0 (== ref, i.e. 0 here) is the ZERO pose, so BASE_Z is the
    zero-pose standing height. GetUp then lifts the robot to the standing pose
    (STANDING_Z). The spawn must still be a vertical settle, not a topple
    (M20 lesson: a standing spawn that topples onto its side cannot be recovered).
  * Joint damping/armature/frictionloss are zeroed: the Isaac/robot_lab DCMotor
    provides all damping through the actuator (M20 finding).
"""
import math
import os
import shutil
import xml.etree.ElementTree as ET

import numpy as np

URDF = "/home/ancn/rl_sar/policy/ancnw/ancnw_description/urdf/ancnw.urdf"
OUT_DIR = "/home/ancn/rl_sar/src/rl_sar_zoo/ancnw_description/mjcf"
ASSET_DIR = os.path.join(OUT_DIR, "assets")
MESH_SRC = "/home/ancn/rl_sar/policy/ancnw/ancnw_description/meshes"

# Standing pose. FK check (all four wheel centres at z = -0.549, symmetric):
#   front legs (fl, fr): hip 0, thigh +0.6, calf -1.0   (axes are 0 -1 0)
#   rear  legs (hl, hr): hip 0, thigh -0.6, calf +1.0
#   wheel radius 0.11  ->  STANDING_Z = 0.549 + 0.11 = 0.659
#
# qpos0 cannot carry that pose (joint/@ref is banned, see the module docstring),
# so the robot spawns in the ZERO pose and GetUp lifts it:
#   zero pose wheel centre z = -0.4274 (thigh->calf -0.19799 + calf->wheel -0.22936)
#   -> BASE_Z = 0.4274 + 0.11 = 0.537
STANDING_Z = 0.659
BASE_Z = 0.537
CROUCH_Z = 0.652
WHEEL_RADIUS = 0.11

FRONT_LEGS = ["fl", "fr"]
REAR_LEGS = ["hl", "hr"]
LEGS = FRONT_LEGS + REAR_LEGS

# default (standing) joint angles, POLICY order
DEFAULT_POSE = {}
for leg in FRONT_LEGS:
    DEFAULT_POSE[f"{leg}_hip_joint"] = 0.0
    DEFAULT_POSE[f"{leg}_thigh_joint"] = 0.6
    DEFAULT_POSE[f"{leg}_calf_joint"] = -1.0
for leg in REAR_LEGS:
    DEFAULT_POSE[f"{leg}_hip_joint"] = 0.0
    DEFAULT_POSE[f"{leg}_thigh_joint"] = -0.6
    DEFAULT_POSE[f"{leg}_calf_joint"] = 1.0
for leg in LEGS:
    DEFAULT_POSE[f"{leg}_wheel_joint"] = 0.0

# folded pre-stand waypoint used by the FSM GetUp (POLICY order).
# The 45 deg calf geometry limits how far the leg folds; (1.0, 1.9) keeps the
# wheels ~7 mm off the ground (FK: wheel centre z = -0.542 vs -0.549 standing).
CROUCH_POSE = dict(DEFAULT_POSE)
for leg in FRONT_LEGS:
    CROUCH_POSE[f"{leg}_thigh_joint"] = 1.0
    CROUCH_POSE[f"{leg}_calf_joint"] = -1.9
for leg in REAR_LEGS:
    CROUCH_POSE[f"{leg}_thigh_joint"] = -1.0
    CROUCH_POSE[f"{leg}_calf_joint"] = 1.9

LEG_JOINTS = [f"{leg}_{seg}_joint" for leg in LEGS for seg in ("hip", "thigh", "calf")]
WHEEL_JOINTS = [f"{leg}_wheel_joint" for leg in LEGS]
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
    m = matmul(rot_z(y), matmul(rot_y(p), rot_x(r)))
    return mat_to_quat(m)


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
    return [qw, qx, qy, qz]  # MuJoCo order: w x y z


def quat_to_mat(q):
    w, x, y, z = q
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]


def principal_inertia(quat_origin, inert):
    """Return (quat, [ixx, iyy, izz]) with the inertia expressed in principal axes."""
    ixx, iyy, izz, ixy, ixz, iyz = inert
    I = np.array([[ixx, ixy, ixz],
                  [ixy, iyy, iyz],
                  [ixz, iyz, izz]])
    lam, V = np.linalg.eigh(I)
    if np.linalg.det(V) < 0:
        V[:, 0] *= -1
    R_o = quat_to_mat(quat_origin)
    R_total = R_o @ V
    return mat_to_quat(R_total.tolist()), [float(x) for x in lam]


def parse_origin(el):
    if el is None:
        return [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]
    xyz = [float(x) for x in el.get("xyz", "0 0 0").split()]
    rpy = [float(x) for x in el.get("rpy", "0 0 0").split()]
    return xyz, rpy_to_quat(*rpy)


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
    child = j.find("child").get("link")
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
    else:
        print("WARNING missing mesh:", src)

L = []


def indent(depth):
    return "  " * depth


def emit_body(link_name, depth):
    info = links[link_name]
    is_root = link_name == root_link
    if is_root:
        L.append(f'{indent(depth)}<body name="{link_name}" pos="0 0 {f(BASE_Z)}" childclass="ancnw">')
    else:
        pos, quat = None, None
        for jn, j in joints.items():
            if j["child"] == link_name:
                pos, quat = j["pos"], j["quat"]
                break
        L.append(f'{indent(depth)}<body name="{link_name}" pos="{vec(pos)}" quat="{vec(quat)}">')

    if is_root:
        L.append(f'{indent(depth + 1)}<freejoint />')
    else:
        jn = [n for n, j in joints.items() if j["child"] == link_name][0]
        j = joints[jn]
        # NOTE: no ref= attribute on purpose - MuJoCo would rotate by (qpos - ref).
        attrs = f'name="{jn}" type="hinge" axis="{vec(j["axis"])}"'
        if j["range"]:
            attrs += f' range="{f(j["range"][0])} {f(j["range"][1])}"'
        else:
            attrs += ' limited="false"'
        L.append(f'{indent(depth + 1)}<joint {attrs} />')

    if info["inertial"]:
        pos, quat, mass, inert = info["inertial"]
        pquat, diag = principal_inertia(quat, inert)
        L.append(f'{indent(depth + 1)}<inertial pos="{vec(pos)}" quat="{vec(pquat)}" mass="{f(mass)}" '
                 f'diaginertia="{vec(diag)}" />')

    for meshfile, pos, quat in info["visuals"]:
        mesh_name = os.path.splitext(meshfile)[0]
        L.append(f'{indent(depth + 1)}<geom mesh="{mesh_name}" material="gray" class="visual" '
                 f'pos="{vec(pos)}" quat="{vec(quat)}" />')

    for ci, (gtype, size, pos, quat) in enumerate(info["collisions"]):
        cls = "wheel" if (gtype == "cylinder" and link_name.endswith("wheel")) else "collision"
        L.append(f'{indent(depth + 1)}<geom name="col_{link_name}_{ci}" type="{gtype}" size="{vec(size)}" pos="{vec(pos)}" '
                 f'quat="{vec(quat)}" class="{cls}" />')

    if is_root:
        L.append(f'{indent(depth + 1)}<site name="imu" pos="0 0 0" />')

    for jn in children.get(link_name, []):
        emit_body(joints[jn]["child"], depth + 1)
    L.append(f'{indent(depth)}</body>')


L.append('<mujoco model="ancnw">')
L.append('  <compiler angle="radian" meshdir="assets" autolimits="true" />')
L.append('  <option cone="elliptic" impratio="100" />')
L.append('  <default>')
L.append('    <default class="ancnw">')
# wheel-ground traction is what carries a wheeled policy; legs only need to not slip
L.append('      <geom friction="0.8 0.02 0.01" margin="0.001" condim="3" />')
# Isaac DCMotor sets friction=0.0 and provides all damping through the actuator,
# so the joint itself must carry no damping / frictionloss (M20 finding).
L.append('      <joint damping="0.0" armature="0.0" frictionloss="0.0" />')
L.append('      <default class="visual">')
L.append('        <geom type="mesh" contype="0" conaffinity="0" group="2" />')
L.append('      </default>')
L.append('      <default class="collision">')
L.append('        <geom group="3" />')
L.append('      </default>')
L.append('      <default class="wheel">')
L.append(f'        <geom group="3" condim="3" friction="1.2 0.02 0.01" priority="1" />')
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

# MuJoCo joint index order = depth-first order of the kinematic tree.
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
assert len(mjcf_order) == 16, mjcf_order

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

# keyframe qpos/ctrl follow the MJCF joint order; the free joint is
# "x y z qw qx qy qz". qpos0 is the zero pose (BASE_Z, all joints 0); these two
# keys hold the standing / crouch poses at their own base height.
def to_mjcf(pose_policy):
    return [pose_policy[POLICY_JOINTS[k]] for k in range(len(mjcf_order))]


home_mjcf = to_mjcf(DEFAULT_POSE)
crouch_mjcf = to_mjcf(CROUCH_POSE)
L.append('  <keyframe>')
L.append(f'    <key name="home" qpos="0 0 {f(STANDING_Z)} 1 0 0 0 {vec(home_mjcf)}" '
         f'ctrl="{vec(home_mjcf)}" />')
L.append(f'    <key name="crouch" qpos="0 0 {f(CROUCH_Z)} 1 0 0 0 {vec(crouch_mjcf)}" '
         f'ctrl="{vec(crouch_mjcf)}" />')
L.append('  </keyframe>')
L.append('</mujoco>')

os.makedirs(OUT_DIR, exist_ok=True)
with open(os.path.join(OUT_DIR, "ancnw.xml"), "w") as fh:
    fh.write("\n".join(L) + "\n")

print("wrote", os.path.join(OUT_DIR, "ancnw.xml"))
print("base link:", root_link, "links:", len(links), "joints:", len(joints))

# ---- sanity: FK of the emitted ref pose, wheel contact height -----------------
def fk_check(pose):
    pos = {"base_link": np.zeros(3)}
    R = {"base_link": np.eye(3)}

    def rec(link):
        for jn in children.get(link, []):
            j = joints[jn]
            p = pos[link] + R[link] @ np.array(j["pos"])
            rj = R[link] @ quat_to_mat(j["quat"])
            q = pose.get(jn, 0.0)
            axis = np.array(j["axis"])
            K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
            pos[j["child"]] = p
            R[j["child"]] = rj @ (np.eye(3) + np.sin(q) * K + (1 - np.cos(q)) * K @ K)
            rec(j["child"])

    rec("base_link")
    zs = {leg: pos[f"{leg}_wheel"][2] for leg in LEGS}
    print("  wheel centre z:", {k: round(float(v), 4) for k, v in zs.items()},
          "-> contact height", round(-min(zs.values()) + WHEEL_RADIUS, 4))
    return zs


print("default pose FK:")
fk_check(DEFAULT_POSE)
print("crouch pose FK:")
fk_check(CROUCH_POSE)
