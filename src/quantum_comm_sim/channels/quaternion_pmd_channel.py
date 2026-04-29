"""
Academic References:
  - Gordon, J.P. & Kogelnik, H., PMD fundamentals: polarization mode dispersion in optical fibers,
    Proceedings of the National Academy of Sciences, 97(9), 4541-4550, 2000.
  - Poole, C.D. & Wagner, R.E., Phenomenological approach to polarization dispersion
    in long single-mode fibers, Electronics Letters, 22(19), 1029-1030, 1986.
  - Wai, P.K.A. & Menyuk, C.R., Polarization mode dispersion, decorrelation, and fiber modeling,
    Journal of Lightwave Technology, 14(2), 148-157, 1996.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class Quaternion:
    """Quaternion representation for polarization states."""
    w: float
    x: float
    y: float
    z: float

    def normalize(self):
        norm = np.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)
        if norm > 0:
            self.w /= norm
            self.x /= norm
            self.y /= norm
            self.z /= norm
        return self

    def to_array(self):
        return np.array([self.w, self.x, self.y, self.z])

    @classmethod
    def from_pmd_vector(cls, beta_vec):
        beta_norm = np.linalg.norm(beta_vec)
        if beta_norm < 1e-10:
            return cls(1.0, 0.0, 0.0, 0.0)
        half_beta = beta_norm / 2.0
        w = np.cos(half_beta)
        xyz = np.sin(half_beta) * beta_vec / beta_norm
        return cls(w, xyz[0], xyz[1], xyz[2])


def quaternion_product(p, q):
    pw, px, py, pz = p[0], p[1], p[2], p[3]
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    w = pw*qw - px*qx - py*qy - pz*qz
    x = pw*qx + px*qw + py*qz - pz*qy
    y = pw*qy - px*qz + py*qw + pz*qx
    z = pw*qz + px*qy - py*qx + pz*qw
    return np.array([w, x, y, z])


def quaternion_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quaternion_inverse(q):
    q_conj = quaternion_conjugate(q)
    norm_sq = np.sum(q**2)
    return q_conj / (norm_sq + 1e-12)


def quaternion_to_rotation_vector(q):
    qw, qv = q[0], q[1:4]
    qv_norm = np.linalg.norm(qv)
    if qv_norm < 1e-8:
        return 2.0 * qv
    theta = 2.0 * np.arctan2(qv_norm, qw)
    if np.abs(theta) < 1e-6:
        return 2.0 * qv
    return (theta / qv_norm) * qv


def rotation_vector_to_quaternion(theta_v):
    theta = np.linalg.norm(theta_v)
    if theta < 1e-8:
        return np.array([1.0, 0.5*theta_v[0], 0.5*theta_v[1], 0.5*theta_v[2]])
    half_theta = 0.5 * theta
    s = np.sin(half_theta) / theta
    return np.array([np.cos(half_theta), s*theta_v[0], s*theta_v[1], s*theta_v[2]])


def quaternion_boxplus(q, delta_theta):
    delta_q = rotation_vector_to_quaternion(delta_theta)
    result = quaternion_product(q, delta_q)
    return result / (np.linalg.norm(result) + 1e-12)


def quaternion_boxminus(q1, q2):
    q2_inv = quaternion_inverse(q2)
    q_diff = quaternion_product(q2_inv, q1)
    return quaternion_to_rotation_vector(q_diff)


class QuaternionPMDChannel:
    """Frequency-dependent PMD channel using quaternion representation."""

    def __init__(self, fiber_length_km=50.0, beta_0=0.1, beta_1=0.05,
                 time_scale=1000, drift_gamma=0.01, seed=None,
                 beta_1_boost=5.0, beta_1_step_start=300, beta_1_step_len=50):
        self.fiber_length = fiber_length_km
        self.beta_0 = beta_0
        self.beta_1 = beta_1
        self.time_scale = time_scale
        self.current_iteration = 0
        self.drift_gamma = drift_gamma
        self.beta_1_base = float(beta_1)
        self.beta_1 = float(beta_1)
        self.beta_1_boost = float(beta_1_boost)
        self.beta_1_step_start = int(beta_1_step_start)
        self.beta_1_step_len = int(beta_1_step_len)
        if seed is not None:
            np.random.seed(seed)
        self.chrom_dir = np.random.randn(3)
        self.chrom_dir /= np.linalg.norm(self.chrom_dir)
        self.chrom_dir_gamma = 0.01
        self.chrom_dir_jump_p = 0.002
        self.base_pmd = np.random.randn(3)
        self.base_pmd = self.base_pmd / np.linalg.norm(self.base_pmd) * beta_0
        self.dgd_std = 0.085e-12 * np.sqrt(fiber_length_km)

    def update(self, iteration):
        self.current_iteration = iteration
        if np.random.rand() < self.chrom_dir_jump_p:
            d = np.random.randn(3)
        else:
            d = (1 - self.chrom_dir_gamma) * self.chrom_dir + self.chrom_dir_gamma * np.random.randn(3)
        self.chrom_dir = d / (np.linalg.norm(d) + 1e-12)
        drift_noise = np.random.randn(3)
        self.base_pmd = (1 - self.drift_gamma) * self.base_pmd + self.drift_gamma * drift_noise
        self.base_pmd = self.base_pmd / np.linalg.norm(self.base_pmd) * self.beta_0
        if self.beta_1_step_start <= iteration < (self.beta_1_step_start + self.beta_1_step_len):
            self.beta_1 = self.beta_1_base * self.beta_1_boost
        else:
            self.beta_1 = self.beta_1_base
        if np.random.rand() < 0.002:
            self.base_pmd = np.random.randn(3)
            self.base_pmd = self.base_pmd / np.linalg.norm(self.base_pmd) * self.beta_0

    def get_pmd_vector(self, omega):
        omega_0 = 2 * np.pi * 193.1e12
        delta_omega = (omega - omega_0) / omega_0
        time_factor = 1.0 + 0.1 * np.sin(2 * np.pi * self.current_iteration / self.time_scale)
        drift_term = np.random.randn(3) * self.dgd_std * 0.1
        if np.random.rand() < 0.01:
            drift_term += np.random.randn(3) * 5 * self.dgd_std
        beta = self.base_pmd * time_factor + drift_term
        chromatic_rotation = self.beta_1 * delta_omega * self.chrom_dir
        beta += chromatic_rotation
        beta /= np.linalg.norm(beta)
        return beta

    def get_state(self, omega):
        beta_vec = self.get_pmd_vector(omega)
        return Quaternion.from_pmd_vector(beta_vec)
