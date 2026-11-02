import numpy as np
from numba import jit
from scipy.spatial.transform import Rotation

def compute_euler_from_matrix(R, axes='xyz', extrinsic=True):
    rot = Rotation.from_matrix(R)
    return rot.as_euler(axes.lower(), degrees=False).reshape(1, 3)

def compute_q_from_matrix(R):
    return Rotation.from_matrix(R).as_quat()

def hat(v):
    v = np.squeeze(v)
    R = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return R

@jit(nopython=True, parallel=False, cache=True)
def mat_exp(omega):
    def hat_local(v):
        v = v.flatten()
        return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    angle = np.linalg.norm(omega)
    if angle < 1e-10:
        return np.identity(3) + hat_local(omega)
    axis = omega / angle
    s = np.sin(angle)
    c = np.cos(angle)
    return c * np.identity(3) + (1 - c) * np.outer(axis, axis) + s * hat_local(axis)

mat_exp_vec = np.vectorize(mat_exp, signature="(3)->(3,3)")

def mat_log(R):
    q = compute_q_from_matrix(R)
    w = q[3]
    vec = q[0:3]
    n = np.linalg.norm(vec)
    epsilon = 1e-7
    if n < epsilon:
        w2 = w * w
        n2 = n * n
        atn = 2.0 / w - (2.0 * n2) / (w * w2)
    else:
        if np.absolute(w) < epsilon:
            if w > 0: atn = np.pi / n
            else: atn = -np.pi / n
        else:
            atn = 2.0 * np.arctan2(n, w) / n
    tangent = atn * vec
    return tangent.reshape(-1, 1)

@jit(nopython=True, parallel=False, cache=True)
def Jr_exp(phi):
    def hat_local(v):
        v = v.flatten()
        return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    theta = np.linalg.norm(phi)
    if theta < 1e-3:
        J = np.eye(3) - 0.5 * hat_local(phi) + 1.0 / 6.0 * (hat_local(phi) @ hat_local(phi))
    else:
        J = (np.eye(3) - (1 - np.cos(theta)) / np.power(theta, 2.0) * hat_local(phi)
            + (theta - np.sin(theta)) / np.power(theta, 3.0) * (hat_local(phi) @ hat_local(phi)))
    return J
