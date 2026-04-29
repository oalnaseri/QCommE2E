"""
Academic References:
  - Mandic, D.P. et al., The Quaternion LMS Algorithm for Adaptive Filtering
    of Hypercomplex Processes, IEEE Transactions on Signal Processing, 57(4), 2009.
  - Shen, Y. et al., Kalman filtering for polarization mode dispersion tracking
    in coherent optical communication systems, Journal of Lightwave Technology, 2010.
"""

import numpy as np
from quantum_comm_sim.channels.quaternion_pmd_channel import (
    quaternion_product,
    quaternion_conjugate,
    quaternion_inverse,
    quaternion_to_rotation_vector,
    quaternion_boxplus,
    quaternion_boxminus,
)


def compute_spectral_fidelity(psi_in, psi_out):
    inner_product = np.dot(psi_in, psi_out)
    return np.abs(inner_product)**2


def compute_broadband_fidelity(psi_in, psi_out_freq, spectral_weights):
    num_freq = psi_out_freq.shape[1]
    fidelities = np.zeros(num_freq)
    for i in range(num_freq):
        fidelities[i] = compute_spectral_fidelity(psi_in, psi_out_freq[:, i])
    return np.sum(spectral_weights * fidelities)


class QuaternionNLMS:
    def __init__(self, stepsize=0.5, numfreq=5, eps=1e-12, clamp_delta=0.5,
                 sign_invariant=True, update_sign=-1.0):
        self.mu = float(stepsize)
        self.numfreq = int(numfreq)
        self.eps = float(eps)
        self.clamp_delta = float(clamp_delta)
        self.sign_invariant = bool(sign_invariant)
        self.update_sign = float(update_sign)
        self.qcomp = np.tile(np.array([1.0, 0.0, 0.0, 0.0])[:, None], (1, self.numfreq))
        self.w = self.qcomp.reshape(-1, 1)
        self.mse_history = []
        self.w_history = []
        self.spectral_fidelity_history = []
        self.error_norm_history = []

    def _normalize_target(self, target):
        if target.ndim == 1:
            return np.tile(target.reshape(4, 1), (1, self.numfreq))
        return target

    def adapt(self, x, target, spectralweights=None):
        qtarget_all = self._normalize_target(target)
        total_mse = 0.0
        total_errnorm = 0.0
        for f in range(self.numfreq):
            qc = self.qcomp[:, f].copy()
            qch = x[:, f].copy()
            qtgt = qtarget_all[:, f].copy()
            qcconj = quaternion_conjugate(qc)
            qout = quaternion_product(qcconj, qch)
            qout = qout / (np.linalg.norm(qout) + self.eps)
            if self.sign_invariant and (np.dot(qtgt, qout) < 0.0):
                qtgt = -qtgt
            nu = quaternion_boxminus(qtgt, qout)
            p = float(np.dot(qch, qch)) + self.eps
            delta = (self.mu / p) * nu
            dn = np.linalg.norm(delta)
            if dn > self.clamp_delta:
                delta = delta * (self.clamp_delta / (dn + self.eps))
            self.qcomp[:, f] = quaternion_boxplus(qc, self.update_sign * delta)
            self.qcomp[:, f] = self.qcomp[:, f] / (np.linalg.norm(self.qcomp[:, f]) + self.eps)
            total_errnorm += float(np.linalg.norm(nu))
            total_mse += float(np.dot(nu, nu))
        avg_mse = total_mse / self.numfreq
        avg_err = total_errnorm / self.numfreq
        self.mse_history.append(avg_mse)
        self.error_norm_history.append(avg_err)
        self.w = self.qcomp.reshape(-1, 1)
        self.w_history.append(self.qcomp.flatten())
        if spectralweights is not None:
            compensated = np.zeros((4, self.numfreq))
            for f in range(self.numfreq):
                qcconj = quaternion_conjugate(self.qcomp[:, f])
                compensated[:, f] = quaternion_product(qcconj, x[:, f])
                compensated[:, f] /= (np.linalg.norm(compensated[:, f]) + self.eps)
            qtgt0 = qtarget_all[:, 0]
            fid = compute_broadband_fidelity(qtgt0, compensated, spectralweights)
            self.spectral_fidelity_history.append(fid)
        else:
            self.spectral_fidelity_history.append(1.0 - 0.5 * avg_mse)


class QuaternionKalmanBaseline:
    def __init__(self, processnoise=1e-6, observationnoise=1e-2, numfreq=5,
                 initialPscale=1e-2, clamp_delta=0.5, sign_invariant=True,
                 update_sign=-1.0):
        self.numfreq = int(numfreq)
        self.qcomp = np.tile(np.array([1.0, 0.0, 0.0, 0.0])[:, None], (1, self.numfreq))
        self.P = [initialPscale * np.eye(3) for _ in range(self.numfreq)]
        self.Q = [float(processnoise) * np.eye(3) for _ in range(self.numfreq)]
        self.R = float(observationnoise)
        self.clamp_delta = float(clamp_delta)
        self.sign_invariant = bool(sign_invariant)
        self.update_sign = float(update_sign)
        self.w = self.qcomp.reshape(-1, 1)
        self.mse_history = []
        self.w_history = []
        self.spectral_fidelity_history = []
        self.error_norm_history = []

    def _R3(self):
        return self.R * np.eye(3)

    def _normalize_target(self, target):
        if target.ndim == 1:
            return np.tile(target.reshape(4, 1), (1, self.numfreq))
        return target

    def adapt(self, x, target, spectralweights=None):
        qtarget_all = self._normalize_target(target)
        total_mse = 0.0
        total_errnorm = 0.0
        for f in range(self.numfreq):
            qc = self.qcomp[:, f].copy()
            qch = x[:, f].copy()
            qtgt = qtarget_all[:, f].copy()
            Ppred = self.P[f] + self.Q[f]
            qcconj = quaternion_conjugate(qc)
            qout = quaternion_product(qcconj, qch)
            qout = qout / (np.linalg.norm(qout) + 1e-12)
            if self.sign_invariant and (np.dot(qtgt, qout) < 0.0):
                qtgt = -qtgt
            nu = quaternion_boxminus(qtgt, qout).reshape(3, 1)
            H = np.eye(3)
            R3 = self._R3()
            S = H @ Ppred @ H.T + R3
            try:
                Sinv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                Sinv = np.linalg.pinv(S)
            K = Ppred @ H.T @ Sinv
            delta = (K @ nu).flatten()
            dn = np.linalg.norm(delta)
            if dn > self.clamp_delta:
                delta = delta * (self.clamp_delta / (dn + 1e-12))
            I = np.eye(3)
            self.P[f] = (I - K @ H) @ Ppred @ (I - K @ H).T + K @ R3 @ K.T
            self.P[f] = 0.5 * (self.P[f] + self.P[f].T)
            self.qcomp[:, f] = quaternion_boxplus(qc, self.update_sign * delta)
            self.qcomp[:, f] = self.qcomp[:, f] / (np.linalg.norm(self.qcomp[:, f]) + 1e-12)
            total_errnorm += float(np.linalg.norm(nu))
            total_mse += float((nu.T @ nu).item())
        avg_mse = total_mse / self.numfreq
        avg_err = total_errnorm / self.numfreq
        self.mse_history.append(avg_mse)
        self.error_norm_history.append(avg_err)
        self.w = self.qcomp.reshape(-1, 1)
        self.w_history.append(self.qcomp.flatten())
        if spectralweights is not None:
            compensated = np.zeros((4, self.numfreq))
            for f in range(self.numfreq):
                qcconj = quaternion_conjugate(self.qcomp[:, f])
                compensated[:, f] = quaternion_product(qcconj, x[:, f])
                compensated[:, f] /= (np.linalg.norm(compensated[:, f]) + 1e-12)
            qtgt0 = qtarget_all[:, 0]
            fid = compute_broadband_fidelity(qtgt0, compensated, spectralweights)
            self.spectral_fidelity_history.append(fid)
        else:
            self.spectral_fidelity_history.append(1.0 - 0.5 * avg_mse)


class TrueQuaternionLMS:
    """Proper Quaternion LMS using HR-calculus.

    Based on: Mandic et al., "The Quaternion LMS Algorithm for Adaptive
    Filtering of Hypercomplex Processes", IEEE Trans. Signal Process., 2009.

    Update rule: q_comp ← q_comp + μ × gradient where the gradient respects
    quaternion non-commutativity.
    """

    def __init__(self, step_size=0.01, num_freq=5):
        self.mu = float(step_size)
        self.num_freq = int(num_freq)
        self.q_comp = np.tile(np.array([[1.0], [0.0], [0.0], [0.0]]), (1, self.num_freq))
        self.mse_history = []
        self.spectral_fidelity_history = []
        self.error_norm_history = []
        self.w_history = []

    def adapt(self, q_channel, q_target, spectral_weights=None):
        total_error_norm = 0.0
        total_mse = 0.0
        if q_target.ndim > 1:
            q_tgt_single = q_target[:, 0]
        else:
            q_tgt_single = q_target.flatten()
        for f in range(self.num_freq):
            q_c = self.q_comp[:, f]
            q_ch = q_channel[:, f]
            q_tgt = q_tgt_single
            q_c_conj = quaternion_conjugate(q_c)
            q_out = quaternion_product(q_c_conj, q_ch)
            q_err_quat = quaternion_product(quaternion_inverse(q_tgt), q_out)
            e_vec = quaternion_to_rotation_vector(q_err_quat)
            error_norm = np.linalg.norm(e_vec)
            total_error_norm += error_norm
            err1 = np.linalg.norm(q_tgt - q_out)
            err2 = np.linalg.norm(q_tgt + q_out)
            error_dist = min(err1, err2)
            total_mse += error_dist**2
            q_ch_conj = quaternion_conjugate(q_ch)
            e_quat = np.array([0.0, e_vec[0], e_vec[1], e_vec[2]])
            grad_direction = quaternion_product(e_quat, q_ch_conj)
            delta_theta = self.mu * grad_direction[1:4]
            self.q_comp[:, f] = quaternion_boxplus(q_c, -delta_theta)
        avg_error_norm = total_error_norm / self.num_freq
        avg_mse = total_mse / self.num_freq
        self.error_norm_history.append(avg_error_norm)
        self.mse_history.append(avg_mse)
        self.w_history.append(self.q_comp.flatten())
        if spectral_weights is not None:
            compensated = np.zeros((4, self.num_freq))
            for f in range(self.num_freq):
                q_c_conj = quaternion_conjugate(self.q_comp[:, f])
                compensated[:, f] = quaternion_product(q_c_conj, q_channel[:, f])
            compensated /= (np.linalg.norm(compensated, axis=0, keepdims=True) + 1e-12)
            qtgt0 = q_tgt_single
            fid = compute_broadband_fidelity(qtgt0, compensated, spectral_weights)
            self.spectral_fidelity_history.append(fid)
        else:
            self.spectral_fidelity_history.append(1.0 - 0.5 * avg_mse)


class TrueQuaternionRLS:
    """Proper Quaternion RLS using HR-calculus style updates.

    Uses per-frequency 3x3 inverse correlation matrices in the tangent space.
    """

    def __init__(self, forgetting_factor=0.99, delta=1e-2, num_freq=5):
        self.lam = float(forgetting_factor)
        self.num_freq = int(num_freq)
        self.P = [np.eye(3) / float(delta) for _ in range(self.num_freq)]
        self.q_comp = np.tile(np.array([[1.0], [0.0], [0.0], [0.0]]), (1, self.num_freq))
        self.mse_history = []
        self.spectral_fidelity_history = []
        self.error_norm_history = []
        self.w_history = []

    def adapt(self, q_channel, q_target, spectral_weights=None):
        total_error_norm = 0.0
        total_mse = 0.0
        if q_target.ndim > 1:
            q_tgt_single = q_target[:, 0]
        else:
            q_tgt_single = q_target.flatten()
        for f in range(self.num_freq):
            q_c = self.q_comp[:, f]
            q_ch = q_channel[:, f]
            q_tgt = q_tgt_single
            q_c_conj = quaternion_conjugate(q_c)
            q_out = quaternion_product(q_c_conj, q_ch)
            q_err_quat = quaternion_product(quaternion_inverse(q_tgt), q_out)
            e_vec = quaternion_to_rotation_vector(q_err_quat)
            error_norm = np.linalg.norm(e_vec)
            total_error_norm += error_norm
            err1 = np.linalg.norm(q_tgt - q_out)
            err2 = np.linalg.norm(q_tgt + q_out)
            error_dist = min(err1, err2)
            total_mse += error_dist**2
            u = e_vec.reshape(3, 1)
            P_f = self.P[f]
            denom = self.lam + float((u.T @ P_f @ u).item())
            k = (P_f @ u) / denom
            P_new = (P_f - k @ u.T @ P_f) / self.lam
            self.P[f] = 0.5 * (P_new + P_new.T)
            delta_theta = k.flatten()
            self.q_comp[:, f] = quaternion_boxplus(q_c, -delta_theta)
        avg_error_norm = total_error_norm / self.num_freq
        avg_mse = total_mse / self.num_freq
        self.error_norm_history.append(avg_error_norm)
        self.mse_history.append(avg_mse)
        self.w_history.append(self.q_comp.flatten())
        if spectral_weights is not None:
            compensated = np.zeros((4, self.num_freq))
            for f in range(self.num_freq):
                q_c_conj = quaternion_conjugate(self.q_comp[:, f])
                compensated[:, f] = quaternion_product(q_c_conj, q_channel[:, f])
            compensated /= (np.linalg.norm(compensated, axis=0, keepdims=True) + 1e-12)
            qtgt0 = q_tgt_single
            fid = compute_broadband_fidelity(qtgt0, compensated, spectral_weights)
            self.spectral_fidelity_history.append(fid)
        else:
            self.spectral_fidelity_history.append(1.0 - 0.5 * avg_mse)


class QuaternionMEKF:
    """Multiplicative Extended Kalman Filter (MEKF) on quaternion manifold.

    This is a frequency-wise MEKF that tracks the compensation quaternion
    using a 3-DoF error state in the tangent space, similar to attitude
    estimation filters.
    """

    def __init__(self, processnoise=1e-5, observationnoise=1e-3, numfreq=5,
                 initialPscale=1e-2, clamp_delta=0.5, sign_invariant=True,
                 update_sign=-1.0):
        self.numfreq = int(numfreq)
        self.qcomp = np.tile(np.array([1.0, 0.0, 0.0, 0.0])[:, None], (1, self.numfreq))
        self.P = [initialPscale * np.eye(3) for _ in range(self.numfreq)]
        self.Q = [float(processnoise) * np.eye(3) for _ in range(self.numfreq)]
        self.R = float(observationnoise)
        self.clamp_delta = float(clamp_delta)
        self.sign_invariant = bool(sign_invariant)
        self.update_sign = float(update_sign)
        self.w = self.qcomp.reshape(-1, 1)
        self.mse_history = []
        self.w_history = []
        self.spectral_fidelity_history = []
        self.error_norm_history = []

    def _R3(self):
        return self.R * np.eye(3)

    def _normalize_target(self, target):
        if target.ndim == 1:
            return np.tile(target.reshape(4, 1), (1, self.numfreq))
        return target

    def adapt(self, x, target, spectralweights=None):
        qtarget_all = self._normalize_target(target)
        total_mse = 0.0
        total_errnorm = 0.0
        for f in range(self.numfreq):
            qc = self.qcomp[:, f].copy()
            qch = x[:, f].copy()
            qtgt = qtarget_all[:, f].copy()
            Ppred = self.P[f] + self.Q[f]
            qcconj = quaternion_conjugate(qc)
            qout = quaternion_product(qcconj, qch)
            qout = qout / (np.linalg.norm(qout) + 1e-12)
            if self.sign_invariant and (np.dot(qtgt, qout) < 0.0):
                qtgt = -qtgt
            nu = quaternion_boxminus(qtgt, qout).reshape(3, 1)
            H = np.eye(3)
            R3 = self._R3()
            S = H @ Ppred @ H.T + R3
            try:
                Sinv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                Sinv = np.linalg.pinv(S)
            K = Ppred @ H.T @ Sinv
            delta = (K @ nu).flatten()
            dn = np.linalg.norm(delta)
            if dn > self.clamp_delta:
                delta = delta * (self.clamp_delta / (dn + 1e-12))
            I = np.eye(3)
            self.P[f] = (I - K @ H) @ Ppred @ (I - K @ H).T + K @ R3 @ K.T
            self.P[f] = 0.5 * (self.P[f] + self.P[f].T)
            self.qcomp[:, f] = quaternion_boxplus(qc, self.update_sign * delta)
            self.qcomp[:, f] = self.qcomp[:, f] / (np.linalg.norm(self.qcomp[:, f]) + 1e-12)
            total_errnorm += float(np.linalg.norm(nu))
            total_mse += float((nu.T @ nu).item())
        avg_mse = total_mse / self.numfreq
        avg_err = total_errnorm / self.numfreq
        self.mse_history.append(avg_mse)
        self.error_norm_history.append(avg_err)
        self.w = self.qcomp.reshape(-1, 1)
        self.w_history.append(self.qcomp.flatten())
        if spectralweights is not None:
            compensated = np.zeros((4, self.numfreq))
            for f in range(self.numfreq):
                qcconj = quaternion_conjugate(self.qcomp[:, f])
                compensated[:, f] = quaternion_product(qcconj, x[:, f])
                compensated[:, f] /= (np.linalg.norm(compensated[:, f]) + 1e-12)
            qtgt0 = qtarget_all[:, 0]
            fid = compute_broadband_fidelity(qtgt0, compensated, spectralweights)
            self.spectral_fidelity_history.append(fid)
        else:
            self.spectral_fidelity_history.append(1.0 - 0.5 * avg_mse)


class FMQuaternionMEKF(QuaternionMEKF):
    """Frequency-multiplexed MEKF variant.

    Wraps QuaternionMEKF but allows more aggressive process noise on
    selected frequency bins to better track rapidly varying PMD in some
    parts of the spectrum.
    """

    def __init__(self, fast_bins=None, fast_factor=10.0, **kwargs):
        super().__init__(**kwargs)
        if fast_bins is None:
            self.fast_bins = []
        else:
            self.fast_bins = list(fast_bins)
        for f in self.fast_bins:
            if 0 <= f < self.numfreq:
                self.Q[f] = self.Q[f] * float(fast_factor)
