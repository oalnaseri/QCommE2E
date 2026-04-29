"""
Academic References:
  - Mandic, D.P. et al., The Quaternion LMS Algorithm for Adaptive Filtering
    of Hypercomplex Processes, IEEE Transactions on Signal Processing, 57(4), 2009.
  - Shen, Y. et al., Kalman filtering for polarization mode dispersion tracking
    in coherent optical communication systems, Journal of Lightwave Technology, 2010.
"""

import numpy as np
from .quaternion_pmd_channel import (
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
            total_mse += float(nu.T @ nu)
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
