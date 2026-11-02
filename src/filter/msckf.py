import numpy as np
from numba import jit
from .math_utils import compute_euler_from_matrix, mat_exp_vec, mat_exp, hat, Jr_exp

class State(object):
    def __init__(self):
        self.s_R = None
        self.s_v = None
        self.s_p = None
        self.s_ba = None
        self.s_bg = None
        self.s_timestamp_us = -1
        self.N = 0  
        self.si_Rs = [] 
        self.si_ps = []
        self.si_timestamps_us = []

    def initialize_state(self, t_us, R, v, p, ba_init, bg_init):
        self.s_R = R
        self.s_v = v.reshape(3,1)
        self.s_p = p.reshape(3,1)
        self.s_bg = bg_init.reshape(3,1)
        self.s_ba = ba_init.reshape(3,1)
        self.s_timestamp_us = t_us
        self.si_Rs = []
        self.si_ps = []
        self.si_timestamps_us = []

    def apply_correction(self, dX):
        dX_past = dX[:-15]
        dX_evol = dX[-15:]
        if self.N > 0:
            temp = dX_past.reshape((self.N, 6))
            dps = np.expand_dims(temp[:, 3:6], axis=2)
            dthetas = temp[:, 0:3]
            dRs = mat_exp_vec(dthetas)
            Rs_past = np.stack(self.si_Rs, axis=0)
            ps_past = np.stack(self.si_ps, axis=0)
            Rs_past_new = np.matmul(dRs, Rs_past)
            ps_past_new = ps_past + dps
            self.si_Rs = list(Rs_past_new)
            self.si_ps = list(ps_past_new)

        dtheta = dX_evol[:3].reshape(3)
        dv = dX_evol[3:6]
        dp = dX_evol[6:9]
        dbg = dX_evol[9:12]
        dba = dX_evol[12:15]

        dR = mat_exp(dtheta)
        self.s_R = dR.dot(self.s_R)
        self.s_v = self.s_v + dv
        self.s_p = self.s_p + dp
        self.s_bg = self.s_bg + dbg
        self.s_ba = self.s_ba + dba

@jit(nopython=True, parallel=False, cache=True)
def propagate_rvt_and_jac(R_k, v_k, p_k, b_gk, b_ak, gyr, acc, g, dt):
    def hat_local(v):
        v = v.flatten()
        return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

    dtheta = (gyr - b_gk) * dt
    dRd = mat_exp(dtheta.flatten())
    Rd = R_k @ dRd
    dv_w = R_k @ (acc - b_ak) * dt
    dp_w = 0.5 * dv_w * dt
    gdt = g * dt
    gdt22 = 0.5 * gdt * dt
    vd = v_k + dv_w + gdt
    pd = p_k + v_k * dt + dp_w + gdt22

    A = np.eye(15)
    A[3:6, 0:3] = -hat_local(dv_w)
    A[6:9, 0:3] = -hat_local(dp_w)
    A[6:9, 3:6] = np.eye(3) * dt
    A[0:3, 9:12] = -Rd @ Jr_exp(dtheta.flatten()) * dt
    A[3:6, 12:15] = -R_k * dt
    A[6:9, 12:15] = -0.5 * R_k * dt * dt
    return Rd, vd, pd, A

@jit(nopython=True, parallel=False, cache=True)
def propagate_covariance(A_aug, B_aug, dt, Sigma, W, Q):
    dim_new_state = A_aug.shape[0] - A_aug.shape[1]
    A = A_aug[-15 - dim_new_state :, -15:]
    AT = A_aug[-15 - dim_new_state :, -15:].T
    ret = np.zeros((A_aug.shape[0], A_aug.shape[0]))
    ret[: -15 - dim_new_state, : -15 - dim_new_state] = Sigma[:-15, :-15]
    ret[: -15 - dim_new_state, -15 - dim_new_state :] = Sigma[:-15, -15:] @ AT
    ret[-15 - dim_new_state :, : -15 - dim_new_state] = A @ Sigma[-15:, :-15]
    ret[-15 - dim_new_state :, -15 - dim_new_state :] = A @ Sigma[-15:, -15:] @ AT
    ret[-15 - dim_new_state :, -15 - dim_new_state :] += B_aug @ W @ B_aug.T
    ret[-6:, -6:] += dt * Q
    return ret

class FilterConfig:
    def __init__(self, **kwargs):
        self.g_norm = kwargs.get("g_norm", 9.81)
        self.sigma_na = kwargs.get("sigma_na", np.sqrt(1e-3))
        self.sigma_ng = kwargs.get("sigma_ng", np.sqrt(1e-4))
        self.ita_ba = kwargs.get("ita_ba", 5e-4)
        self.ita_bg = kwargs.get("ita_bg", 1e-5)
        
        self.init_attitude_sigma = kwargs.get("init_attitude_sigma", 1.0 / 180.0 * np.pi)
        self.init_yaw_sigma = kwargs.get("init_yaw_sigma", 0.1 / 180.0 * np.pi)
        self.init_vel_sigma = kwargs.get("init_vel_sigma", 1.0)
        self.init_pos_sigma = kwargs.get("init_pos_sigma", 0.001)
        self.init_bg_sigma = kwargs.get("init_bg_sigma", 0.0001)
        self.init_ba_sigma = kwargs.get("init_ba_sigma", 0.2)


CARRIER_FILTER_PRESETS = {
    'weight_robot': {
        'sigma_na': 0.1,
        'sigma_ng': 0.1,
        'ita_ba': 1.0e-4,
        'init_yaw_sigma': 5.0 / 180.0 * np.pi,
        'init_pos_sigma': 0.01,
    },
}


class ImuMSCKF:
    def __init__(self, config=FilterConfig()):
        g_norm = getattr(config, "g_norm", 9.81)
        self.g = np.array([0, 0, -g_norm]).reshape((3, 1))

        self.sigma_na = config.sigma_na; self.sigma_ng = config.sigma_ng
        self.ita_ba = config.ita_ba;     self.ita_bg = config.ita_bg
        self.init_attitude_sigma = config.init_attitude_sigma
        self.init_yaw_sigma = config.init_yaw_sigma
        self.init_vel_sigma = config.init_vel_sigma
        self.init_pos_sigma = config.init_pos_sigma
        self.init_bg_sigma = config.init_bg_sigma
        self.init_ba_sigma = config.init_ba_sigma

        self.state = State()
        self.initialized = False

    def prepare_filter(self):
        var_a = np.power(self.sigma_na, 2.0)
        var_g = np.power(self.sigma_ng, 2.0)
        var_ba = np.power(self.ita_ba, 2.0)
        var_bg = np.power(self.ita_bg, 2.0)
        self.W = np.diag(np.array([var_g, var_g, var_g, var_a, var_a, var_a]))
        self.Q = np.diag(np.array([var_bg, var_bg, var_bg, var_ba, var_ba, var_ba]))
        self.reset_covariance()

    def reset_covariance(self):
        var_atti = np.power(self.init_attitude_sigma, 2.0)
        var_yaw = np.power(self.init_yaw_sigma, 2.0)
        var_vel = np.power(self.init_vel_sigma, 2.0)
        var_pos = np.power(self.init_pos_sigma, 2.0)
        var_bg_init = np.power(self.init_bg_sigma, 2.0)
        var_ba_init = np.power(self.init_ba_sigma, 2.0)

        Cov = np.zeros((15 + 6 * self.state.N, 15 + 6 * self.state.N))
        for i in range(self.state.N):
            Cov[6 * i :, 6 * i :][0:3, 0:3] = np.diag([var_atti, var_atti, var_yaw])
            Cov[6 * i :, 6 * i :][3:6, 3:6] = np.diag([var_pos, var_pos, var_pos])

        Cov[-15:, -15:][:3, :3] = np.diag(np.array([var_atti, var_atti, var_yaw]))
        Cov[-15:, -15:][3:6, 3:6] = np.diag(np.array([var_vel, var_vel, var_vel]))
        Cov[-15:, -15:][6:9, 6:9] = np.diag(np.array([var_pos, var_pos, var_pos]))
        Cov[-15:, -15:][9:12, 9:12] = np.diag(np.array([var_bg_init, var_bg_init, var_bg_init]))
        Cov[-15:, -15:][12:15, 12:15] = np.diag(np.array([var_ba_init, var_ba_init, var_ba_init]))
        self.Sigma = Cov

    def initialize_with_state(self, t_us, R, v, p, ba_init, bg_init, verbose=True):
        self.prepare_filter()
        self.initialized = True
        self.state.initialize_state(t_us, R, v, p, ba_init, bg_init)
        if verbose:
            print("Filter initialized with full state!")

    def propagate(self, acc, gyr, t_us, t_augmentation_us=None):
        acc = acc.reshape(3,1)
        gyr = gyr.reshape(3,1)
        R_k, v_k, p_k, b_ak, b_gk = self.state.s_R, self.state.s_v, self.state.s_p, self.state.s_ba, self.state.s_bg
        N = self.state.N
        dt_us = t_us - self.state.s_timestamp_us
        
        R_kp1, v_kp1, p_kp1, Akp1 = propagate_rvt_and_jac(R_k, v_k, p_k, b_gk, b_ak, gyr, acc, self.g, dt_us * 1e-6)

        B = np.zeros((15, 6))
        B[0:3, 0:3] = -Akp1[0:3, 9:12]
        B[3:6, 3:6] = -Akp1[3:6, 12:15]
        B[6:9, 3:6] = -Akp1[6:9, 12:15]

        if t_augmentation_us: 
            dtd_us = t_augmentation_us - self.state.s_timestamp_us
            Rd, vd, pd, Ad = propagate_rvt_and_jac(R_k, v_k, p_k, b_gk, b_ak, gyr, acc, self.g, dtd_us * 1e-6)
            
            JA = np.zeros((6, 15))
            JA[0:3, :] = Ad[0:3, :]
            JA[3:6, :] = Ad[6:9, :]
            
            A_aug = np.zeros(((15 + 6 * (N + 1)), (15 + 6 * N)))
            A_aug[0 : 6 * N, 0 : 6 * N] = np.eye(6 * N)
            A_aug[-15 - 6 : -15, -15:] = JA
            A_aug[-15:, -15:] = Akp1
            
            BJ = np.zeros((6, 6))
            BJ[0:3, 0:3] = -Ad[0:3, 9:12]
            BJ[3:6, 3:6] = -Ad[6:9, 12:15]
            B_aug = np.zeros(((15 + 6), 6))
            B_aug[-15 - 6 : -15, :] = BJ
            B_aug[-15:, :] = B

            self.state.si_Rs.append(Rd)
            self.state.si_ps.append(pd)
            self.state.si_timestamps_us.append(t_augmentation_us)
            self.state.N += 1
        else:
            A_aug = np.eye((15 + 6 * N))
            A_aug[-15:, -15:] = Akp1
            B_aug = np.zeros((15, 6))
            B_aug[-15:, :] = B

        self.Sigma = propagate_covariance(A_aug, B_aug, dt_us * 1e-6, self.Sigma, self.W, self.Q)
        
        self.state.s_R = R_kp1
        self.state.s_v = v_kp1
        self.state.s_p = p_kp1
        self.state.s_timestamp_us = t_us

    def velocity_innovation(self, t_begin_us, t_end_us):
        begin_idx = self.state.si_timestamps_us.index(t_begin_us)
        end_idx = self.state.si_timestamps_us.index(t_end_us)
        dt = (t_end_us - t_begin_us) * 1e-6

        Ri = self.state.si_Rs[begin_idx]
        ri_eul = compute_euler_from_matrix(Ri, "xyz", extrinsic=True)
        ri_y = ri_eul[0, 1]
        ri_z = ri_eul[0, 2]
        Ri_z = np.array([
            [np.cos(ri_z), -(np.sin(ri_z)), 0],
            [np.sin(ri_z),  np.cos(ri_z), 0],
            [           0,             0, 1],
        ])

        dp = self.state.si_ps[end_idx] - self.state.si_ps[begin_idx]
        pred_v_avg = Ri_z.T.dot(dp) / dt

        H = np.zeros((3, 15 + 6 * self.state.N))
        H[:, (6 * begin_idx + 3):(6 * begin_idx + 6)] = -Ri_z.T / dt
        H[:, (6 * end_idx + 3):(6 * end_idx + 6)] = Ri_z.T / dt

        Hz = np.array([
            [0, 0, 0], [0, 0, 0],
            [np.cos(ri_z) * np.tan(ri_y), np.sin(ri_z) * np.tan(ri_y), 1],
        ])
        H[:, (6 * begin_idx):(6 * begin_idx + 3)] = np.linalg.multi_dot(
            [Ri_z.T, hat(dp / dt), Hz]
        )
        hph = np.linalg.multi_dot([H, self.Sigma, H.T])
        return {
            'pred_v_avg': pred_v_avg,
            'H': H,
            'HPH': hph,
            'dt': dt,
        }

    def apply_velocity_update(self, meas_v_avg, meas_cov, stats):
        self.R = meas_cov
        H = stats['H']
        pred_v_avg = stats['pred_v_avg']
        S = stats['HPH'] + self.R
        Sinv = np.linalg.inv(S)
        K = np.linalg.multi_dot([self.Sigma, H.T, Sinv])
        delta_X = K.dot(meas_v_avg.reshape(3, 1) - pred_v_avg)
        self.state.apply_correction(delta_X)
        Sigma_up = self.Sigma - np.linalg.multi_dot([K, H, self.Sigma])
        self.Sigma = 0.5 * (Sigma_up + Sigma_up.T)

    def update_velocity_with_history(self, meas_v_avg, meas_cov, t_begin_us, t_end_us):
        stats = self.velocity_innovation(t_begin_us, t_end_us)
        self.apply_velocity_update(meas_v_avg, meas_cov, stats)
        return stats

    def marginalize(self, cut_idx):
        self.state.si_Rs = self.state.si_Rs[cut_idx + 1 :]
        self.state.si_ps = self.state.si_ps[cut_idx + 1 :]
        self.state.si_timestamps_us = self.state.si_timestamps_us[cut_idx + 1 :]
        self.Sigma = self.Sigma[6 * (cut_idx + 1) :, 6 * (cut_idx + 1) :]
        self.state.N = self.state.N - (cut_idx + 1)
