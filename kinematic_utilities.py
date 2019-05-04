import numpy as np


# Finds exponential of twist vector
def exp_twist(xi, theta):
    omega = xi[3:]
    v = xi[:3]

    omega_hat = np.array([[0, -omega[2], omega[1]],
                          [omega[2], 0, -omega[0]],
                          [-omega[1], omega[0], 0]])

    omega_hat_theta = np.identity(3) + np.sin(theta)*omega_hat + (1-np.cos(theta))*np.dot(omega_hat, omega_hat)
    p_val = np.dot((np.identity(3) - omega_hat_theta), np.cross(omega, v)) + np.dot(np.outer(omega, omega), v)*theta
    gab = np.vstack((np.hstack((omega_hat_theta, p_val.reshape(3, 1))), np.array([[0, 0, 0, 1]])))
    return gab


# Finds forward kinematics using Exponential coordinates
def exp_direct_kin(gst0, joint_axes, q_axes, th):
    dim = 3
    num_of_joints = len(th)
    gst_temp = np.identity(dim+1)
    tran_upto_joint = np.zeros((num_of_joints+1, dim+1, dim+1))

    for i in range(num_of_joints):
        tran_upto_joint[i, :, :] = gst_temp
        omega = joint_axes[i, :]
        q = q_axes[i, :]
        xi = np.hstack((np.cross(-omega, q), omega))
        gst_joint_i = exp_twist(xi, th[i])
        gst_temp = np.dot(gst_temp, gst_joint_i)

    tran_upto_joint[num_of_joints, :, :] = gst_temp
    gst = np.dot(gst_temp, gst0)
    return gst, tran_upto_joint


# Finds spatial Jacobian using Exponential coordinates
def velocity_direct_kin(gst0, joint_axes, q_axes, th):
    dim = 3
    num_of_joints = len(th)
    spatial_jac = np.zeros((dim+3, num_of_joints))

    gst, transform_upto_joint = exp_direct_kin(gst0, joint_axes, q_axes, th)
    for i in range(num_of_joints):
        if i > 0:
            g = transform_upto_joint[i, :, :]
            R = g[:3, :3]
            p = g[:3, 3]
            p_hat = np.array([[0, -p[2], p[1]],
                              [p[2], 0, -p[0]],
                              [-p[1], p[0], 0]])
            temp1 = np.dot(p_hat, R)
            Ad_g = np.hstack((R, temp1))
            temp2 = np.hstack((np.zeros((3, 3)), R))
            Ad_g = np.vstack((Ad_g, temp2))

        omega = joint_axes[i, :]
        q = q_axes[i, :]
        xi = np.hstack((np.cross(-omega, q), omega))

        if i > 0:
            xi_prime = np.dot(Ad_g, xi)
        else:
            xi_prime = xi

        spatial_jac[:, i] = xi_prime
    return spatial_jac
