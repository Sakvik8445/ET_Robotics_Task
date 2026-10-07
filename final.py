# update.py (thread-safe MuJoCo access using a global lock)
import os
import time
import threading
import traceback
import numpy as np
import mujoco
import mujoco.viewer
from mujoco import mjtObj, mj_id2name

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load model and data
with open("scene.xml", "r") as f:
    xml = f.read()
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

# Global lock to serialize all mujoco/data access
mujoco_lock = threading.RLock()

def id2name(obj_type, idx):
    n = mj_id2name(model, obj_type, idx)
    return n.decode() if isinstance(n, bytes) else n

# Diagnostics (safe: use lock)
with mujoco_lock:
    print("Bodies:")
    for i in range(model.nbody):
        print(i, id2name(mjtObj.mjOBJ_BODY, i))
    print("\nJoints:")
    for j in range(model.njnt):
        print(j, id2name(mjtObj.mjOBJ_JOINT, j), "qposadr=", model.jnt_qposadr[j], "dofadr=", model.jnt_dofadr[j])
    print("\nActuators:")
    for i in range(model.nu):
        print(i, id2name(mjtObj.mjOBJ_ACTUATOR, i), "trntype=", model.actuator_trntype[i])

# Helper: step (locked)
def step(n=1):
    with mujoco_lock:
        for _ in range(n):
            mujoco.mj_step(model, data)

# IDs (safe to query without lock since model is static)
ee_bid = mujoco.mj_name2id(model, mjtObj.mjOBJ_BODY, "hand")
cup_bid = mujoco.mj_name2id(model, mjtObj.mjOBJ_BODY, "cup")
hand_bid = ee_bid
if ee_bid == -1 or cup_bid == -1:
    raise SystemExit("Required bodies 'hand' or 'cup' not found in scene.xml")

# Arm configuration
arm_joint_count = 7
arm_joint_indices = list(range(arm_joint_count))
arm_actuator_indices = list(range(arm_joint_count))

# Comfortable nominal pose
q_nom = np.array([0.0, -0.8, 0.0, -2.0, 0.0, 1.2, 0.6])

# Jacobian prealloc
nv = model.nv
jacp = np.zeros((3, nv))
jacr = np.zeros((3, nv))
jac_point = np.zeros(3)

# Safe wrappers for reading/writing data
def ee_pos():
    with mujoco_lock:
        return data.xpos[ee_bid].copy()

def ee_quat():
    with mujoco_lock:
        return data.xquat[ee_bid].copy()

def cup_pos():
    with mujoco_lock:
        return data.xpos[cup_bid].copy()

def cup_quat():
    with mujoco_lock:
        return data.xquat[cup_bid].copy()

def compute_arm_jacobian():
    with mujoco_lock:
        mujoco.mj_jac(model, data, jacp, jacr, jac_point, ee_bid)
        cols = [int(model.jnt_dofadr[j]) for j in arm_joint_indices]
        J_arm = jacp[:, cols].copy()
    return J_arm

# Quaternion helpers (pure numpy, no lock needed)
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

def quat_rotate(v, q):
    qv = np.concatenate(([0.0], v))
    return quat_mul(quat_mul(q, qv), quat_conj(q))[1:]

def quat_inv(q):
    qc = quat_conj(q)
    norm2 = np.dot(q, q)
    if norm2 == 0:
        return qc
    return qc / norm2

# Controller gains and limits
Kp_joint = 40.0
Kd_joint = 5.0
cartesian_vel_gain = 3.0
alpha_ns = 6.0
max_torque = 80.0

# Apply torques (must lock)
def apply_joint_torques(torques):
    with mujoco_lock:
        ctrl = np.zeros(model.nu, dtype=float)
        for i, aid in enumerate(arm_actuator_indices):
            if aid < model.nu:
                ctrl[aid] = np.clip(torques[i], -max_torque, max_torque)
        # merge with existing ctrl to preserve other actuator targets
        existing = data.ctrl.copy()
        for i in range(model.nu):
            if ctrl[i] == 0.0 and existing[i] != 0.0:
                ctrl[i] = existing[i]
        data.ctrl[:] = ctrl

# Finger qpos addresses (read once)
finger_j1_qposadr = None
finger_j2_qposadr = None
for j in range(model.njnt):
    jname = id2name(mjtObj.mjOBJ_JOINT, j)
    if jname == "finger_joint1":
        finger_j1_qposadr = model.jnt_qposadr[j]
    if jname == "finger_joint2":
        finger_j2_qposadr = model.jnt_qposadr[j]
print("Finger qpos addresses:", finger_j1_qposadr, finger_j2_qposadr)

# Direct qpos gripper control (locked)
def set_gripper_qpos(left_q, right_q):
    with mujoco_lock:
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
        # step a few frames so viewer shows the change
        for _ in range(8):
            mujoco.mj_step(model, data)

def open_gripper():
    set_gripper_qpos(0.0, 0.0)

def close_gripper():
    set_gripper_qpos(0.04, 0.04)

# Nullspace-regularized IK move (uses locked helpers)
def move_ee_to_jointspace_ns(target_pos, duration=1.0):
    steps = max(1, int(duration / model.opt.timestep))
    dt = model.opt.timestep
    n_arm = len(arm_joint_indices)
    for _ in range(steps):
        cur_pos = ee_pos()
        pos_err = target_pos - cur_pos
        v_des = cartesian_vel_gain * pos_err
        J = compute_arm_jacobian()   # (3, n_arm)
        lam = 1e-3
        JT = J.T
        JJt = J.dot(JT) + lam * np.eye(3)
        try:
            J_pinv = JT.dot(np.linalg.inv(JJt))
        except np.linalg.LinAlgError:
            J_pinv = np.linalg.pinv(J)
        qdot_task = J_pinv.dot(v_des)
        JJ = J_pinv.dot(J)
        P_ns = np.eye(n_arm) - JJ
        # read qpos and qvel under lock
        with mujoco_lock:
            qcur_vec = np.array([data.qpos[model.jnt_qposadr[j]] for j in arm_joint_indices])
            qvel_vec = np.array([data.qvel[int(model.jnt_dofadr[j])] if int(model.jnt_dofadr[j]) < model.nv else 0.0 for j in arm_joint_indices])
        qdot_ns = alpha_ns * (q_nom - qcur_vec)
        qdot = qdot_task + P_ns.dot(qdot_ns)
        q_target = qcur_vec + qdot * dt
        torque = Kp_joint * (q_target - qcur_vec) - Kd_joint * qvel_vec
        torque = np.clip(torque, -max_torque, max_torque)
        apply_joint_torques(torque)
        step(1)

# Manual attach helpers (use lock when writing mujoco data)
attached = False
rel_pos_hand_to_cup = np.zeros(3)
rel_quat_hand_to_cup = np.array([1.0, 0.0, 0.0, 0.0])

def attach_cup_to_hand():
    global attached, rel_pos_hand_to_cup, rel_quat_hand_to_cup
    with mujoco_lock:
        hand_p = data.xpos[hand_bid].copy()
        hand_q = data.xquat[hand_bid].copy()
        cup_p = data.xpos[cup_bid].copy()
        cup_q = data.xquat[cup_bid].copy()
    rel_pos_hand_to_cup = cup_p - hand_p
    rel_quat_hand_to_cup = quat_mul(quat_inv(hand_q), cup_q)
    attached = True
    print("[ATTACH] Cup attached to hand (manual)")

def detach_cup():
    global attached
    attached = False
    print("[DETACH] Cup detached")

def update_attached_cup():
    if not attached:
        return
    with mujoco_lock:
        hand_p = data.xpos[hand_bid].copy()
        hand_q = data.xquat[hand_bid].copy()
    rel_world = quat_rotate(rel_pos_hand_to_cup, hand_q)
    new_p = hand_p + rel_world
    new_q = quat_mul(hand_q, rel_quat_hand_to_cup)
    # write into qpos for free joint if present
    cup_joint_idx = None
    for j in range(model.njnt):
        if model.jnt_bodyid[j] == cup_bid:
            cup_joint_idx = j
            break
    with mujoco_lock:
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

# Compute grasp frames (reads cup under lock)
def compute_grasp_frames():
    cp = cup_pos()
    cq = cup_quat()
    approach = quat_rotate(np.array([0.0, 0.0, -1.0]), cq)
    approach = approach / (np.linalg.norm(approach) + 1e-9)
    p_grasp = cp + approach * 0.02
    pre_grasp = cp + approach * 0.12
    lift_pose = cp + np.array([0.0, 0.0, 0.20])
    desired_quat = ee_quat()
    return pre_grasp, p_grasp, lift_pose, desired_quat

# Initialize comfortable pose (locked)
with mujoco_lock:
    for i, q in enumerate(q_nom):
        if i < model.njnt:
            data.qpos[model.jnt_qposadr[i]] = float(q)
    for j in range(model.njnt):
        dof = int(model.jnt_dofadr[j])
        if 0 <= dof < model.nv:
            data.qvel[dof] = 0.0
    mujoco.mj_forward(model, data)

# Small settle
for _ in range(40):
    step(1)

# Ensure gripper open
open_gripper()

# Compute frames
pre_grasp, grasp_pose, lift_pose, _ = compute_grasp_frames()

# Demo sequence runs in background thread
def demo_sequence():
    try:
        print("Starting pick sequence (background thread)...")
        move_ee_to_jointspace_ns(pre_grasp, duration=1.0)
        move_ee_to_jointspace_ns(grasp_pose, duration=0.9)
        close_gripper()
        attach_cup_to_hand()
        # lift while attached (incremental)
        for _ in range(int(1.0 / model.opt.timestep)):
            move_ee_to_jointspace_ns(lift_pose, duration=0.02)
            update_attached_cup()
        print("Pick finished (cup attached).")
        # Transport and place
        place_target = cup_pos() + np.array([0.0, -0.25, 0.20])
        move_ee_to_jointspace_ns(place_target, duration=1.2)
        lower_target = place_target + np.array([0.0, 0.0, -0.12])
        move_ee_to_jointspace_ns(lower_target, duration=0.8)
        open_gripper()
        detach_cup()
        print("Place finished (cup detached).")
    except Exception:
        print("Exception in demo thread:")
        traceback.print_exc()

# Launch viewer in main thread and start demo thread
print("Launching viewer. Demo will run in background; viewer will remain open until you close it.")
try:
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # initial camera alignment (locked)
        with mujoco_lock:
            try:
                cam_target = data.xpos[cup_bid] if np.isfinite(data.xpos[cup_bid]).all() else data.xpos[ee_bid]
                viewer.cam.lookat[:] = cam_target
                viewer.cam.distance = 0.8
                viewer.cam.elevation = -20.0
                viewer.cam.azimuth = 90.0
            except Exception:
                pass

        demo_thread = threading.Thread(target=demo_sequence, daemon=True)
        demo_thread.start()

        # main viewer loop: keep stepping and syncing; update attached cup each frame
        try:
            while True:
                if viewer.is_running():
                    if attached:
                        update_attached_cup()
                    # step and sync under lock to avoid concurrent mjData access
                    with mujoco_lock:
                        mujoco.mj_step(model, data)
                        viewer.sync()
                    time.sleep(0.001)
                else:
                    if attached:
                        update_attached_cup()
                    with mujoco_lock:
                        mujoco.mj_step(model, data)
                    time.sleep(0.05)
        except KeyboardInterrupt:
            print("\nInterrupted by user (Ctrl+C). Exiting.")
except Exception as e:
    print("Viewer failed to launch or closed unexpectedly:", e)
    traceback.print_exc()

print("Program exiting. Close the viewer window if it is still open.")
