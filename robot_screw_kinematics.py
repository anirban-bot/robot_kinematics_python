from kinematic_utilities import *


# Joint angles
th_ang = [0.1, -0.5, 0.1, 0.1, 0.1, 0.5, 0.1]
print("\nCurrent joint configuration of Baxter robot:")
print(th_ang)
print("=================================\n")

# Link lengths
L0, L1, L2, L3 = 0.27035, 0.36435, 0.37429, 0.22952
o1, o2, o3 = 0.069, 0.069, 0.010
gripper_link = 0.025 + 0.1372

# Base transformation
base_mat = np.array([[0.7071, -0.7071, 0.0, 0.0648],
                     [0.7071, 0.7071, 0.0, 0.2584],
                     [0.0, 0.0, 1.0, 0.1190],
                     [0, 0, 0, 1.0]])

# Define axis of rotations
w1, w2, w3 = [0, 0, 1], [0, 1, 0], [1, 0, 0]
w4 = w6 = w2
w5 = w7 = w3
wr = np.array([w1, w2, w3, w4, w5, w6, w7])

# Frame origins
q1, q2, q4, q6 = [0, 0, 0], [o1, 0, L0], [o1+L1, 0, L0-o2], [o1+L1+L2, 0, L0-o2-o3]
q3, q5, q7 = q2, q4, q6
qr = np.array([q1, q2, q3, q4, q5, q6, q7])

# Define g_st0
g_st0 = np.array([[0, 0, 1, 1.194935],
                  [0.0, 1.0, 0.0, 0.0],
                  [-1.0, 0.0, 0.0, 0.19135],
                  [0.0, 0.0, 0.0, 1.0]])

# Solve direct position kinematics
gst, transform_upto_joint = exp_direct_kin(g_st0, wr, qr, th_ang)
g_ee = np.dot(base_mat, gst)
print("baseTend_effector:")
print(g_ee)
print("=================================\n")

# Compute spatial jacobian
spatial_jac = velocity_direct_kin(g_st0, wr, qr, th_ang)
print("Spatial Jacobian of Baxter Robot:")
print(spatial_jac)
print("=================================\n")
