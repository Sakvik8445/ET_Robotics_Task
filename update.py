import os
import time
import numpy as np
import mujoco
import mujoco.viewer
from mujoco import mjtObj, mj_id2name

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# -------------------------
# Load model and data
# -------------------------
with open("scene.xml", "r") as f:
    xml = f.read()
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

# -------------------------
# Utilities and diagnostics
# -------------------------
def id2name(obj_type, idx):
    n = mj_id2name(model, obj_type, idx)
    return n.decode() if isinstance(n, bytes) else n

print("Bodies:")
for i in range(model.nbody):
    print(i, id2name(mjtObj.mjOBJ_BODY, i))
print("\nJoints:")
for j in range(model.njnt):
    print(j, id2name(mjtObj.mjOBJ_JOINT, j), "qposadr=", model.jnt_qposadr[j], "dofadr=", model.jnt_dofadr[j])
print("\nActuators:")
for i in range(model.nu):
    print(i, id2name(mjtObj.mjOBJ_ACTUATOR, i), "trntype=", model.actuator_trntype[i])

def step(n=1):
    for _ in range(n):
        mujoco.mj_step(model, data)

# -------------------------
# IDs and configuration
# -------------------------
ee_bid = mujoco.mj_name2id(model, mjtObj.mjOBJ_BODY, "hand")
cup_bid = mujoco.mj_name2id(model, mjtObj.mjOBJ_BODY, "cup")
if ee_bid == -1 or cup_bid == -1:
    raise SystemExit("Required bodies 'hand' or 'cup' not found in scene.xml")

# assume first 7 joints are the arm
arm_joint_count = 7
arm_joint_indices = list(range(arm_joint_count))

# comfortable nominal pose
q_nom = np.array([0.0, -0.8, 0.0, -2.0, 0.0, 1.2, 0.6])

# Jacobian buffers
nv = model.nv
jacp = np.zeros((3, nv))
jacr = np.zeros((3, nv))
jac_point = np.zeros(3)

# -------------------------
# Quaternion and math helpers
# -------------------------
def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw
    ], dtype=float)

def quat_conj(q):
    q = np.array(q, dtype=float)
    q[1:] *= -1.0
    return q

def quat_inv(q):
    qc = quat_conj(q)
    norm2 = np.dot(q, q)
    if norm2 == 0:
        return qc
    return qc / norm2

def quat_rotate(v, q):
    qv = np.concatenate(([0.0], v))
    return quat_mul(quat_mul(q, qv), quat_conj(q))[1:]

def quat_to_rotvec(q):
    # q = [w, x, y, z]
    w = np.clip(q[0], -1.0, 1.0)
    v = q[1:4]
    theta = 2.0 * np.arccos(w)
    if theta < 1e-6:
        return 2.0 * v  # small-angle approx
    s = np.sqrt(1.0 - w*w)
    axis = v / (s + 1e-12)
    return axis * theta

def slerp(q0, q1, t):
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = np.dot(q0, q1)
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    DOT_THRESHOLD = 0.9995
    if dot > DOT_THRESHOLD:
        result = q0 + t*(q1 - q0)
        return result / np.linalg.norm(result)
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * t
    s0 = np.sin(theta_0 - theta) / sin_theta_0
    s1 = np.sin(theta) / sin_theta_0
    return s0*q0 + s1*q1

# -------------------------
# Readers
# -------------------------
def ee_pos():
    return data.xpos[ee_bid].copy()

def ee_quat():
    return data.xquat[ee_bid].copy()

def cup_pos():
    return data.xpos[cup_bid].copy()

def cup_quat():
    return data.xquat[cup_bid].copy()

def compute_arm_jacobian():
    mujoco.mj_jac(model, data, jacp, jacr, jac_point, ee_bid)
    cols = [int(model.jnt_dofadr[j]) for j in arm_joint_indices]
    Jp = jacp[:, cols].copy()
    Jr = jacr[:, cols].copy()
    return Jp, Jr

# -------------------------
# Kinematic controller parameters
# -------------------------
cartesian_pos_gain = 6.0   # position gain for task twist
cartesian_rot_gain = 6.0   # orientation gain for task twist
alpha_ns = 6.0             # nullspace posture gain
max_qstep = 0.06           # rad per sim step
damping_base = 1e-2        # base damping for DLS

# -------------------------
# Damped least squares helpers
# -------------------------
def damped_pinv(J, lam):
    # J: m x n (m <= n typically 6 x n)
    JJt = J.dot(J.T)
    m = JJt.shape[0]
    inv = np.linalg.inv(JJt + (lam**2) * np.eye(m))
    return J.T.dot(inv)

# -------------------------
# 6-DOF IK function (DLS + nullspace)
# -------------------------
def move_ee_to_jointspace_ik(target_pos, target_quat,
                             duration=1.0, dt=None,
                             Kp=cartesian_pos_gain, Kr=cartesian_rot_gain,
                             alpha=alpha_ns, lam=damping_base):
    """
    6-DOF IK with Damped Least Squares and nullspace posture control.
    target_quat is [w,x,y,z].
    """
    if dt is None:
        dt = model.opt.timestep
    steps = max(1, int(duration / dt))
    n_arm = len(arm_joint_indices)

    for _ in range(steps):
        # read current pose
        p_cur = data.xpos[ee_bid].copy()
        q_cur = data.xquat[ee_bid].copy()

        # position error
        e_p = target_pos - p_cur

        # orientation error (quaternion error -> rotation vector)
        q_err = quat_mul(target_quat, quat_conj(q_cur))
        e_r = quat_to_rotvec(q_err)

        # desired twist
        v = np.concatenate((Kp * e_p, Kr * e_r))  # 6-vector

        # Jacobian
        mujoco.mj_jac(model, data, jacp, jacr, jac_point, ee_bid)
        cols = [int(model.jnt_dofadr[j]) for j in arm_joint_indices]
        Jp = jacp[:, cols].copy()
        Jr = jacr[:, cols].copy()
        J = np.vstack((Jp, Jr))  # 6 x n_arm

        # adaptive damping: increase if JJt ill-conditioned
        try:
            svals = np.linalg.svd(J, compute_uv=False)
            sigma_min = svals[-1] if svals.size > 0 else 0.0
            # scale lambda: larger when sigma_min small
            lam_eff = lam * (1.0 + (1.0 / (sigma_min + 1e-6)))
            lam_eff = np.clip(lam_eff, lam, 1.0)
        except Exception:
            lam_eff = lam

        # DLS pseudo-inverse
        J_pinv = damped_pinv(J, lam_eff)

        # task solution
        qdot_task = J_pinv.dot(v)

        # nullspace projector
        JJ = J_pinv.dot(J)
        P_ns = np.eye(n_arm) - JJ

        # read current joint positions
        qcur_vec = np.array([data.qpos[model.jnt_qposadr[j]] for j in arm_joint_indices])

        # nullspace bias toward nominal pose
        qdot_ns = alpha * (q_nom - qcur_vec)

        qdot = qdot_task + P_ns.dot(qdot_ns)

        # integrate with clamping
        q_target = qcur_vec + qdot * dt
        dq = q_target - qcur_vec
        dq = np.clip(dq, -max_qstep, max_qstep)
        q_target = qcur_vec + dq

        # write qpos directly and zero velocities for stability
        for i, j in enumerate(arm_joint_indices):
            qaddr = model.jnt_qposadr[j]
            data.qpos[qaddr] = float(q_target[i])
            dof = int(model.jnt_dofadr[j])
            if 0 <= dof < model.nv:
                data.qvel[dof] = 0.0

        mujoco.mj_forward(model, data)
        step(1)

# -------------------------
# Gripper control (direct qpos)
# -------------------------
finger_j1_qposadr = None
finger_j2_qposadr = None
for j in range(model.njnt):
    jn = id2name(mjtObj.mjOBJ_JOINT, j)
    if jn == "finger_joint1":
        finger_j1_qposadr = model.jnt_qposadr[j]
    if jn == "finger_joint2":
        finger_j2_qposadr = model.jnt_qposadr[j]
print("Finger qpos addresses:", finger_j1_qposadr, finger_j2_qposadr)

def set_gripper_qpos(left_q, right_q):
    if finger_j1_qposadr is not None:
        data.qpos[finger_j1_qposadr] = float(left_q)
    if finger_j2_qposadr is not None:
        data.qpos[finger_j2_qposadr] = float(right_q)
    # zero finger velocities
    for j in range(model.njnt):
        qadr = model.jnt_qposadr[j]
        if qadr in (finger_j1_qposadr, finger_j2_qposadr):
            dof = int(model.jnt_dofadr[j])
            if 0 <= dof < model.nv:
                data.qvel[dof] = 0.0
    mujoco.mj_forward(model, data)
    for _ in range(6):
        mujoco.mj_step(model, data)

def open_gripper():
    set_gripper_qpos(0.02, 0.02)

def close_gripper():
    set_gripper_qpos(0.00, 0.00)

# -------------------------
# Grasp frame computation (position + desired orientation)
# -------------------------
def compute_grasp_frames():
    cp = cup_pos()
    cq = cup_quat()

    # approach direction: cup local -Z
    approach = quat_rotate(np.array([0.0, 0.0, -1.0]), cq)
    approach = approach / (np.linalg.norm(approach) + 1e-9)

    # tuned offsets
    pre_grasp = cp + approach * 0.06
    grasp_pose = cp + approach * 0.00
    lift_pose = cp + np.array([0.0, 0.0, 0.20])

    # desired gripper orientation: align gripper local +Z to -approach
    target_z = -approach / (np.linalg.norm(approach) + 1e-9)
    cur_hand_q = ee_quat()
    cur_z = quat_rotate(np.array([0.0, 0.0, 1.0]), cur_hand_q)
    v = np.cross(cur_z, target_z)
    s = np.linalg.norm(v)
    c = np.dot(cur_z, target_z)
    if s < 1e-6 and c > 0.9999:
        rot_quat = np.array([1.0, 0.0, 0.0, 0.0])
    else:
        axis = v / (s + 1e-12)
        angle = np.arctan2(s, c)
        qw = np.cos(angle/2.0)
        qxyz = axis * np.sin(angle/2.0)
        rot_quat = np.array([qw, qxyz[0], qxyz[1], qxyz[2]])
    desired_quat = quat_mul(rot_quat, cur_hand_q)

    return pre_grasp, grasp_pose, lift_pose, desired_quat

# -------------------------
# Last-meter lateral correction (hand-frame)
# -------------------------
def last_meter_correction(grasp_pose, max_offset=0.03):
    hand_p = ee_pos()
    hand_q = ee_quat()
    hand_xmat = data.xmat[ee_bid].reshape(3,3).copy()
    cup_p = cup_pos()
    world_offset = cup_p - hand_p
    rel_hand = hand_xmat.T.dot(world_offset)
    lateral = rel_hand[:2]
    corr = np.zeros(3)
    corr[0] = np.clip(lateral[0], -max_offset, max_offset)
    corr[1] = np.clip(lateral[1], -max_offset, max_offset)
    corr_world = hand_xmat.dot(corr)
    corrected = grasp_pose + corr_world
    return corrected

# -------------------------
# Manual attach with smooth interpolation
# -------------------------
attached = False
rel_pos_hand_to_cup = np.zeros(3)
rel_quat_hand_to_cup = np.array([1.0, 0.0, 0.0, 0.0])
attach_steps = 20

def attach_cup_to_hand():
    global attached, rel_pos_hand_to_cup, rel_quat_hand_to_cup
    hand_p = data.xpos[ee_bid].copy()
    hand_q = data.xquat[ee_bid].copy()
    cup_p = data.xpos[cup_bid].copy()
    cup_q = data.xquat[cup_bid].copy()
    hand_xmat = data.xmat[ee_bid].reshape(3,3).copy()
    world_offset = cup_p - hand_p
    rel_pos_hand_to_cup = hand_xmat.T.dot(world_offset)
    rel_quat_hand_to_cup = quat_mul(quat_inv(hand_q), cup_q)
    attached = True
    print("[ATTACH] stored rel_pos (hand frame):", rel_pos_hand_to_cup)

def detach_cup():
    global attached
    attached = False
    print("[DETACH] cup detached")

def update_attached_cup_smooth():
    if not attached:
        return
    hand_p = data.xpos[ee_bid].copy()
    hand_q = data.xquat[ee_bid].copy()
    hand_xmat = data.xmat[ee_bid].reshape(3,3).copy()
    rel_world = hand_xmat.dot(rel_pos_hand_to_cup)
    target_p = hand_p + rel_world
    target_q = quat_mul(hand_q, rel_quat_hand_to_cup)
    cur_p = data.xpos[cup_bid].copy()
    cur_q = data.xquat[cup_bid].copy()
    t = 1.0 / max(1, attach_steps)
    new_p = cur_p + (target_p - cur_p) * t
    new_q = slerp(cur_q, target_q, t)
    cup_joint_idx = None
    for j in range(model.njnt):
        if model.jnt_bodyid[j] == cup_bid:
            cup_joint_idx = j
            break
    if cup_joint_idx is not None:
        qposadr = int(model.jnt_qposadr[cup_joint_idx])
        data.qpos[qposadr:qposadr+3] = new_p
        data.qpos[qposadr+3:qposadr+7] = new_q
        dofadr = int(model.jnt_dofadr[cup_joint_idx])
        if dofadr >= 0:
            data.qvel[dofadr:dofadr+6] = 0.0
        mujoco.mj_forward(model, data)
    else:
        data.xpos[cup_bid] = new_p
        data.xquat[cup_bid] = new_q
        mujoco.mj_forward(model, data)

# -------------------------
# Initialize and settle
# -------------------------
for i, q in enumerate(q_nom):
    if i < model.njnt:
        data.qpos[model.jnt_qposadr[i]] = float(q)
for j in range(model.njnt):
    dof = int(model.jnt_dofadr[j])
    if 0 <= dof < model.nv:
        data.qvel[dof] = 0.0
mujoco.mj_forward(model, data)

for _ in range(40):
    step(1)

open_gripper()

# -------------------------
# Compute frames and run robust sequence using 6-DOF IK
# -------------------------
pre_grasp, grasp_pose, lift_pose, desired_quat = compute_grasp_frames()
corrected_grasp = last_meter_correction(grasp_pose, max_offset=0.03)
print("pre_grasp:", pre_grasp, "grasp_pose:", grasp_pose, "corrected_grasp:", corrected_grasp)

print("Starting pick sequence (6-DOF IK)...")
# Move to pre-grasp (position-only is fine; orientation will be handled by IK)
move_ee_to_jointspace_ik(pre_grasp, ee_quat(), duration=1.0, Kp=4.0, Kr=1.0, alpha=alpha_ns, lam=damping_base)

# Approach with full 6-DOF IK to align orientation and position
move_ee_to_jointspace_ik(corrected_grasp, desired_quat, duration=0.9, Kp=6.0, Kr=6.0, alpha=alpha_ns, lam=damping_base)

# Close gripper and attach
close_gripper()
attach_cup_to_hand()

# Lift while updating attached cup smoothly
lift_steps = max(1, int(1.0 / model.opt.timestep))
for _ in range(lift_steps):
    move_ee_to_jointspace_ik(lift_pose, desired_quat, duration=model.opt.timestep, Kp=4.0, Kr=4.0, alpha=alpha_ns, lam=damping_base)
    update_attached_cup_smooth()

print("Pick finished (cup attached).")

# Transport and place
place_target = cup_pos() + np.array([0.0, -0.25, 0.20])
move_ee_to_jointspace_ik(place_target, desired_quat, duration=1.2, Kp=4.0, Kr=2.0, alpha=alpha_ns, lam=damping_base)
lower_target = place_target + np.array([0.0, 0.0, -0.12])
move_ee_to_jointspace_ik(lower_target, desired_quat, duration=0.8, Kp=4.0, Kr=2.0, alpha=alpha_ns, lam=damping_base)

open_gripper()
detach_cup()
print("Place finished (cup detached).")

# -------------------------
# Viewer: keep open and responsive
# -------------------------
with mujoco.viewer.launch_passive(model, data) as viewer:
    try:
        viewer.cam.lookat[:] = data.xpos[cup_bid]
        viewer.cam.distance = 0.8
        viewer.cam.elevation = -20.0
        viewer.cam.azimuth = 90.0
    except Exception:
        pass
    print("Viewer running. Close the window or press Ctrl+C to exit.")
    try:
        while viewer.is_running():
            if attached:
                update_attached_cup_smooth()
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.001)
    except KeyboardInterrupt:
        print("Interrupted by user. Exiting.")
# update.py
# Robust 6-DOF IK pick-and-place with diagnostics and safe attach gating.
# Replace your existing update.py with this file. Keep scene.xml unchanged.

# import os
# import time
# import numpy as np
# import mujoco
# import mujoco.viewer
# from mujoco import mjtObj, mj_id2name

# os.chdir(os.path.dirname(os.path.abspath(__file__)))

# # -------------------------
# # Load model and data
# # -------------------------
# with open("scene.xml", "r") as f:
#     xml = f.read()
# model = mujoco.MjModel.from_xml_string(xml)
# data = mujoco.MjData(model)

# # -------------------------
# # Helpers
# # -------------------------
# def id2name(obj_type, idx):
#     n = mj_id2name(model, obj_type, idx)
#     return n.decode() if isinstance(n, bytes) else n

# def step(n=1):
#     for _ in range(n):
#         mujoco.mj_step(model, data)

# # -------------------------
# # Scene diagnostics
# # -------------------------
# print("Bodies:")
# for i in range(model.nbody):
#     print(i, id2name(mjtObj.mjOBJ_BODY, i))
# print("\nJoints:")
# for j in range(model.njnt):
#     print(j, id2name(mjtObj.mjOBJ_JOINT, j), "qposadr=", model.jnt_qposadr[j], "dofadr=", model.jnt_dofadr[j])
# print("\nActuators:")
# for i in range(model.nu):
#     print(i, id2name(mjtObj.mjOBJ_ACTUATOR, i), "trntype=", model.actuator_trntype[i])

# # -------------------------
# # IDs and configuration
# # -------------------------
# ee_bid = mujoco.mj_name2id(model, mjtObj.mjOBJ_BODY, "hand")
# cup_bid = mujoco.mj_name2id(model, mjtObj.mjOBJ_BODY, "cup")
# if ee_bid == -1 or cup_bid == -1:
#     raise SystemExit("Required bodies 'hand' or 'cup' not found in scene.xml")

# arm_joint_count = 7
# arm_joint_indices = list(range(arm_joint_count))

# # nominal pose
# q_nom = np.array([0.0, -0.8, 0.0, -2.0, 0.0, 1.2, 0.6])

# # Jacobian buffers
# nv = model.nv
# jacp = np.zeros((3, nv))
# jacr = np.zeros((3, nv))
# jac_point = np.zeros(3)

# # -------------------------
# # Math / quaternion helpers
# # -------------------------
# def quat_mul(a, b):
#     aw, ax, ay, az = a
#     bw, bx, by, bz = b
#     return np.array([
#         aw*bw - ax*bx - ay*by - az*bz,
#         aw*bx + ax*bw + ay*bz - az*by,
#         aw*by - ax*bz + ay*bw + az*bx,
#         aw*bz + ax*by - ay*bx + az*bw
#     ], dtype=float)

# def quat_conj(q):
#     q = np.array(q, dtype=float)
#     q[1:] *= -1.0
#     return q

# def quat_inv(q):
#     qc = quat_conj(q)
#     n2 = np.dot(q, q)
#     if n2 == 0:
#         return qc
#     return qc / n2

# def quat_rotate(v, q):
#     qv = np.concatenate(([0.0], v))
#     return quat_mul(quat_mul(q, qv), quat_conj(q))[1:]

# def quat_to_rotvec(q):
#     # q = [w, x, y, z]
#     w = np.clip(q[0], -1.0, 1.0)
#     v = q[1:4]
#     theta = 2.0 * np.arccos(w)
#     if theta < 1e-6:
#         return 2.0 * v
#     s = np.sqrt(max(0.0, 1.0 - w*w))
#     axis = v / (s + 1e-12)
#     return axis * theta

# def normalize_quat(q):
#     q = np.array(q, dtype=float)
#     n = np.linalg.norm(q)
#     if n == 0:
#         return np.array([1.0, 0.0, 0.0, 0.0])
#     return q / n

# def slerp(q0, q1, t):
#     q0 = q0 / np.linalg.norm(q0)
#     q1 = q1 / np.linalg.norm(q1)
#     dot = np.dot(q0, q1)
#     if dot < 0.0:
#         q1 = -q1
#         dot = -dot
#     DOT_THRESHOLD = 0.9995
#     if dot > DOT_THRESHOLD:
#         result = q0 + t*(q1 - q0)
#         return result / np.linalg.norm(result)
#     theta_0 = np.arccos(dot)
#     sin_theta_0 = np.sin(theta_0)
#     theta = theta_0 * t
#     s0 = np.sin(theta_0 - theta) / sin_theta_0
#     s1 = np.sin(theta) / sin_theta_0
#     return s0*q0 + s1*q1

# # -------------------------
# # Readers
# # -------------------------
# def ee_pos():
#     return data.xpos[ee_bid].copy()

# def ee_quat():
#     return data.xquat[ee_bid].copy()

# def cup_pos():
#     return data.xpos[cup_bid].copy()

# def cup_quat():
#     return data.xquat[cup_bid].copy()

# def compute_arm_jacobian():
#     mujoco.mj_jac(model, data, jacp, jacr, jac_point, ee_bid)
#     cols = [int(model.jnt_dofadr[j]) for j in arm_joint_indices]
#     Jp = jacp[:, cols].copy()
#     Jr = jacr[:, cols].copy()
#     return Jp, Jr

# # -------------------------
# # IK parameters
# # -------------------------
# cartesian_pos_gain = 6.0
# cartesian_rot_gain = 6.0
# alpha_ns = 6.0
# max_qstep = 0.06
# damping_base = 1e-2

# # -------------------------
# # Damped pseudo-inverse
# # -------------------------
# def damped_pinv(J, lam):
#     JJt = J.dot(J.T)
#     m = JJt.shape[0]
#     inv = np.linalg.inv(JJt + (lam**2) * np.eye(m))
#     return J.T.dot(inv)

# # -------------------------
# # 6-DOF IK with diagnostics and safe attach gating
# # -------------------------
# def move_ee_to_jointspace_ik(target_pos, target_quat,
#                              duration=1.0, Kp=None, Kr=None, alpha=None, lam=None):
#     if Kp is None: Kp = cartesian_pos_gain
#     if Kr is None: Kr = cartesian_rot_gain
#     if alpha is None: alpha = alpha_ns
#     if lam is None: lam = damping_base

#     target_quat = normalize_quat(target_quat)
#     dt = model.opt.timestep
#     steps = max(1, int(duration / dt))
#     n_arm = len(arm_joint_indices)

#     for step_i in range(steps):
#         # read current pose
#         p_cur = data.xpos[ee_bid].copy()
#         q_cur = data.xquat[ee_bid].copy()

#         # errors
#         e_p = target_pos - p_cur
#         q_err = quat_mul(target_quat, quat_conj(q_cur))
#         e_r = quat_to_rotvec(q_err)

#         # desired twist
#         v = np.concatenate((Kp * e_p, Kr * e_r))

#         # Jacobian
#         mujoco.mj_jac(model, data, jacp, jacr, jac_point, ee_bid)
#         cols = [int(model.jnt_dofadr[j]) for j in arm_joint_indices]
#         Jp = jacp[:, cols].copy()
#         Jr = jacr[:, cols].copy()
#         J = np.vstack((Jp, Jr))

#         # diagnostics: singular values
#         try:
#             svals = np.linalg.svd(J, compute_uv=False)
#             sigma_min = svals[-1] if svals.size > 0 else 0.0
#         except Exception:
#             sigma_min = 0.0

#         # adaptive damping
#         lam_eff = lam * (1.0 + (1.0 / (sigma_min + 1e-6)))
#         lam_eff = np.clip(lam_eff, lam, 1.0)

#         # DLS inverse
#         J_pinv = damped_pinv(J, lam_eff)

#         # task and nullspace
#         qdot_task = J_pinv.dot(v)
#         JJ = J_pinv.dot(J)
#         P_ns = np.eye(n_arm) - JJ

#         qcur_vec = np.array([data.qpos[model.jnt_qposadr[j]] for j in arm_joint_indices])
#         qdot_ns = alpha * (q_nom - qcur_vec)
#         qdot = qdot_task + P_ns.dot(qdot_ns)

#         # integrate and clamp
#         q_target = qcur_vec + qdot * dt
#         dq = q_target - qcur_vec
#         max_dq = np.max(np.abs(dq))
#         dq = np.clip(dq, -max_qstep, max_qstep)
#         q_target = qcur_vec + dq

#         # write and forward
#         for i, j in enumerate(arm_joint_indices):
#             qaddr = model.jnt_qposadr[j]
#             data.qpos[qaddr] = float(q_target[i])
#             dof = int(model.jnt_dofadr[j])
#             if 0 <= dof < model.nv:
#                 data.qvel[dof] = 0.0
#         mujoco.mj_forward(model, data)

#         # step and diagnostics print (throttled)
#         step(1)
#         if step_i % max(1, steps//10) == 0 or step_i < 6:
#             print(f"IK step {step_i+1}/{steps}: |e_p|={np.linalg.norm(e_p):.4f} m, |e_r|={np.linalg.norm(e_r):.4f} rad, sigma_min={sigma_min:.4e}, max_dq={max_dq:.4e}")

#     # final diagnostics
#     p_final = data.xpos[ee_bid].copy()
#     q_final = data.xquat[ee_bid].copy()
#     print("IK finished: final pos err", np.linalg.norm(target_pos - p_final), "final ori err", np.linalg.norm(quat_to_rotvec(quat_mul(target_quat, quat_conj(q_final)))))

# # -------------------------
# # Gripper control (direct qpos)
# # -------------------------
# finger_j1_qposadr = None
# finger_j2_qposadr = None
# for j in range(model.njnt):
#     jn = id2name(mjtObj.mjOBJ_JOINT, j)
#     if jn == "finger_joint1":
#         finger_j1_qposadr = model.jnt_qposadr[j]
#     if jn == "finger_joint2":
#         finger_j2_qposadr = model.jnt_qposadr[j]
# print("Finger qpos addresses:", finger_j1_qposadr, finger_j2_qposadr)

# def set_gripper_qpos(left_q, right_q):
#     if finger_j1_qposadr is not None:
#         data.qpos[finger_j1_qposadr] = float(left_q)
#     if finger_j2_qposadr is not None:
#         data.qpos[finger_j2_qposadr] = float(right_q)
#     for j in range(model.njnt):
#         qadr = model.jnt_qposadr[j]
#         if qadr in (finger_j1_qposadr, finger_j2_qposadr):
#             dof = int(model.jnt_dofadr[j])
#             if 0 <= dof < model.nv:
#                 data.qvel[dof] = 0.0
#     mujoco.mj_forward(model, data)
#     for _ in range(6):
#         mujoco.mj_step(model, data)

# def open_gripper():
#     set_gripper_qpos(0.02, 0.02)

# def close_gripper():
#     set_gripper_qpos(0.00, 0.00)

# # -------------------------
# # Grasp frames and last-meter correction
# # -------------------------
# def compute_grasp_frames():
#     cp = cup_pos()
#     cq = cup_quat()
#     approach = quat_rotate(np.array([0.0, 0.0, -1.0]), cq)
#     approach = approach / (np.linalg.norm(approach) + 1e-9)
#     pre_grasp = cp + approach * 0.06
#     grasp_pose = cp + approach * 0.00
#     lift_pose = cp + np.array([0.0, 0.0, 0.20])
#     # desired orientation: align gripper +Z to -approach
#     target_z = -approach / (np.linalg.norm(approach) + 1e-9)
#     cur_hand_q = ee_quat()
#     cur_z = quat_rotate(np.array([0.0, 0.0, 1.0]), cur_hand_q)
#     v = np.cross(cur_z, target_z)
#     s = np.linalg.norm(v)
#     c = np.dot(cur_z, target_z)
#     if s < 1e-6 and c > 0.9999:
#         rot_quat = np.array([1.0, 0.0, 0.0, 0.0])
#     else:
#         axis = v / (s + 1e-12)
#         angle = np.arctan2(s, c)
#         qw = np.cos(angle/2.0)
#         qxyz = axis * np.sin(angle/2.0)
#         rot_quat = np.array([qw, qxyz[0], qxyz[1], qxyz[2]])
#     desired_quat = quat_mul(rot_quat, cur_hand_q)
#     desired_quat = normalize_quat(desired_quat)
#     return pre_grasp, grasp_pose, lift_pose, desired_quat

# def last_meter_correction(grasp_pose, max_offset=0.03):
#     hand_p = ee_pos()
#     hand_xmat = data.xmat[ee_bid].reshape(3,3).copy()
#     cup_p = cup_pos()
#     world_offset = cup_p - hand_p
#     rel_hand = hand_xmat.T.dot(world_offset)
#     lateral = rel_hand[:2]
#     corr = np.zeros(3)
#     corr[0] = np.clip(lateral[0], -max_offset, max_offset)
#     corr[1] = np.clip(lateral[1], -max_offset, max_offset)
#     corr_world = hand_xmat.dot(corr)
#     return grasp_pose + corr_world

# # -------------------------
# # Manual attach with gating
# # -------------------------
# attached = False
# rel_pos_hand_to_cup = np.zeros(3)
# rel_quat_hand_to_cup = np.array([1.0, 0.0, 0.0, 0.0])
# attach_steps = 20

# def attach_cup_to_hand_safe(pos_thresh=0.06, ori_thresh=0.35):
#     """
#     Attach only if hand is close enough to cup.
#     pos_thresh in meters, ori_thresh in radians.
#     Returns True if attached, False otherwise.
#     """
#     global attached, rel_pos_hand_to_cup, rel_quat_hand_to_cup
#     hand_p = data.xpos[ee_bid].copy()
#     hand_q = data.xquat[ee_bid].copy()
#     cup_p = data.xpos[cup_bid].copy()
#     cup_q = data.xquat[cup_bid].copy()
#     dist = np.linalg.norm(cup_p - hand_p)
#     q_err = quat_mul(cup_q, quat_conj(hand_q))
#     ori_err = np.linalg.norm(quat_to_rotvec(q_err))
#     print(f"[ATTACH CHECK] distance={dist:.4f} m, ori_err={ori_err:.4f} rad (thresholds pos={pos_thresh}, ori={ori_thresh})")
#     if dist <= pos_thresh and ori_err <= ori_thresh:
#         hand_xmat = data.xmat[ee_bid].reshape(3,3).copy()
#         world_offset = cup_p - hand_p
#         rel_pos_hand_to_cup = hand_xmat.T.dot(world_offset)
#         rel_quat_hand_to_cup = quat_mul(quat_inv(hand_q), cup_q)
#         attached = True
#         print("[ATTACH] Cup attached to hand (safe). Stored rel_pos (hand frame):", rel_pos_hand_to_cup)
#         return True
#     else:
#         print("[ATTACH] Not attaching: hand too far or misaligned. Run extra IK or adjust approach.")
#         return False

# def update_attached_cup_smooth():
#     if not attached:
#         return
#     hand_p = data.xpos[ee_bid].copy()
#     hand_q = data.xquat[ee_bid].copy()
#     hand_xmat = data.xmat[ee_bid].reshape(3,3).copy()
#     rel_world = hand_xmat.dot(rel_pos_hand_to_cup)
#     target_p = hand_p + rel_world
#     target_q = quat_mul(hand_q, rel_quat_hand_to_cup)
#     cur_p = data.xpos[cup_bid].copy()
#     cur_q = data.xquat[cup_bid].copy()
#     t = 1.0 / max(1, attach_steps)
#     new_p = cur_p + (target_p - cur_p) * t
#     new_q = slerp(cur_q, target_q, t)
#     cup_joint_idx = None
#     for j in range(model.njnt):
#         if model.jnt_bodyid[j] == cup_bid:
#             cup_joint_idx = j
#             break
#     if cup_joint_idx is not None:
#         qposadr = int(model.jnt_qposadr[cup_joint_idx])
#         data.qpos[qposadr:qposadr+3] = new_p
#         data.qpos[qposadr+3:qposadr+7] = new_q
#         dofadr = int(model.jnt_dofadr[cup_joint_idx])
#         if dofadr >= 0:
#             data.qvel[dofadr:dofadr+6] = 0.0
#         mujoco.mj_forward(model, data)
#     else:
#         data.xpos[cup_bid] = new_p
#         data.xquat[cup_bid] = new_q
#         mujoco.mj_forward(model, data)

# # -------------------------
# # Initialize
# # -------------------------
# for i, q in enumerate(q_nom):
#     if i < model.njnt:
#         data.qpos[model.jnt_qposadr[i]] = float(q)
# for j in range(model.njnt):
#     dof = int(model.jnt_dofadr[j])
#     if 0 <= dof < model.nv:
#         data.qvel[dof] = 0.0
# mujoco.mj_forward(model, data)
# for _ in range(40):
#     step(1)
# open_gripper()

# # -------------------------
# # Compute frames and run sequence
# # -------------------------
# pre_grasp, grasp_pose, lift_pose, desired_quat = compute_grasp_frames()
# corrected_grasp = last_meter_correction(grasp_pose, max_offset=0.03)
# print("pre_grasp:", pre_grasp, "grasp_pose:", grasp_pose, "corrected_grasp:", corrected_grasp)

# print("Starting pick sequence (robust 6-DOF IK)...")

# # Move to pre-grasp (position-focused)
# move_ee_to_jointspace_ik(pre_grasp, ee_quat(), duration=1.0, Kp=4.0, Kr=1.0, alpha=alpha_ns, lam=damping_base)

# # Approach with full 6-DOF IK (position + orientation)
# move_ee_to_jointspace_ik(corrected_grasp, desired_quat, duration=1.0, Kp=8.0, Kr=8.0, alpha=alpha_ns, lam=damping_base)

# # Close gripper
# close_gripper()

# # Attempt safe attach; if it fails, run extra IK refinement and retry
# attached_ok = attach_cup_to_hand_safe(pos_thresh=0.06, ori_thresh=0.35)
# if not attached_ok:
#     print("Refining approach: running extra IK steps to reduce error before attach.")
#     # run a short orientation-focused IK to align better
#     move_ee_to_jointspace_ik(corrected_grasp, desired_quat, duration=0.6, Kp=6.0, Kr=10.0, alpha=alpha_ns, lam=damping_base)
#     # try attach again
#     attached_ok = attach_cup_to_hand_safe(pos_thresh=0.07, ori_thresh=0.45)
#     if not attached_ok:
#         print("Warning: attach still failed. Proceeding without attach (demo will continue).")

# # If attached, lift while updating cup
# if attached_ok:
#     lift_steps = max(1, int(1.0 / model.opt.timestep))
#     for _ in range(lift_steps):
#         move_ee_to_jointspace_ik(lift_pose, desired_quat, duration=model.opt.timestep, Kp=4.0, Kr=4.0, alpha=alpha_ns, lam=damping_base)
#         update_attached_cup_smooth()
#     print("Pick finished (cup attached).")
# else:
#     print("Pick aborted (cup not attached).")

# # Transport and place (still run IK to place even if not attached)
# place_target = cup_pos() + np.array([0.0, -0.25, 0.20])
# move_ee_to_jointspace_ik(place_target, desired_quat, duration=1.2, Kp=4.0, Kr=2.0, alpha=alpha_ns, lam=damping_base)
# lower_target = place_target + np.array([0.0, 0.0, -0.12])
# move_ee_to_jointspace_ik(lower_target, desired_quat, duration=0.8, Kp=4.0, Kr=2.0, alpha=alpha_ns, lam=damping_base)

# open_gripper()
# if attached_ok:
#     # detach and let cup settle
#     attached = False
#     print("[DETACH] cup detached")
# print("Place finished.")

# # -------------------------
# # Viewer: keep open and responsive
# # -------------------------
# with mujoco.viewer.launch_passive(model, data) as viewer:
#     try:
#         viewer.cam.lookat[:] = data.xpos[cup_bid]
#         viewer.cam.distance = 0.8
#         viewer.cam.elevation = -20.0
#         viewer.cam.azimuth = 90.0
#     except Exception:
#         pass
#     print("Viewer running. Close the window or press Ctrl+C to exit.")
#     try:
#         while viewer.is_running():
#             if attached:
#                 update_attached_cup_smooth()
#             mujoco.mj_step(model, data)
#             viewer.sync()
#             time.sleep(0.001)
#     except KeyboardInterrupt:
#         print("Interrupted by user. Exiting.")
