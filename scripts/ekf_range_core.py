#!/usr/bin/env python3
"""
ekf_range_core.py

Pure EKF math for 2D range-only multilateration (NO ROS).

State:
    x = [px, py, vx, vy]^T

Process (constant velocity):
    px_{k+1} = px_k + vx_k * dt
    py_{k+1} = py_k + vy_k * dt
    vx_{k+1} = vx_k
    vy_{k+1} = vy_k

Measurement (ranges to fixed sensors):
    z_i = sqrt((px - sx_i)^2 + (py - sy_i)^2) + v_i
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np


__all__ = [
    "ls_init_xy_from_ranges",
    "ConstantVelocityModel",
    "RangeMeasurementModel",
    "RangeEKF",
]


def _as_1d_float(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float).reshape((-1,))
    if z.ndim != 1:
        raise ValueError("z must be 1D")
    if not np.all(np.isfinite(z)):
        raise ValueError("z contains NaN/inf")
    return z


def ls_init_xy_from_ranges(
    sensors_xy: Sequence[Tuple[float, float]],
    z: np.ndarray,
    *,
    min_range: float = 1e-3,
) -> Tuple[float, float]:
    """
    Linear least-squares (LS) multilateration init in 2D.

    We subtract the equation of a reference sensor (x0,y0,z0) from others:
        2*(xi-x0)*x + 2*(yi-y0)*y = (xi^2-x0^2) + (yi^2-y0^2) - (zi^2-z0^2)

    Reference sensor is selected as the smallest range for stability.
    Requires N>=3 and non-degenerate sensor geometry (rank(A)=2).
    """
    sensors = list(sensors_xy)
    N = len(sensors)
    z = _as_1d_float(z)

    if z.shape[0] != N:
        raise ValueError(f"z length {z.shape[0]} != N {N}")
    if N < 3:
        raise ValueError(f"Need at least 3 sensors for 2D LS init, got N={N}")
    if min_range <= 0:
        raise ValueError("min_range must be > 0")

    # Physical constraint: ranges should be positive
    z = np.maximum(z, float(min_range))

    i0 = int(np.argmin(z))
    x0, y0 = sensors[i0]
    z0 = float(z[i0])

    A = []
    b = []
    for i, (xi, yi) in enumerate(sensors):
        if i == i0:
            continue
        zi = float(z[i])
        A.append([2.0 * (xi - x0), 2.0 * (yi - y0)])
        b.append((xi * xi - x0 * x0) + (yi * yi - y0 * y0) - (zi * zi - z0 * z0))

    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    if np.linalg.matrix_rank(A) < 2:
        raise ValueError("Degenerate sensor geometry (rank < 2).")

    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    x_est, y_est = float(sol[0]), float(sol[1])

    if not (math.isfinite(x_est) and math.isfinite(y_est)):
        raise ValueError("LS returned non-finite solution")

    return x_est, y_est


class ConstantVelocityModel:
    """Discrete constant-velocity model for [x,y,vx,vy] with dt and process scale tau."""

    def __init__(self, dt: float, tau: float):
        dt = float(dt)
        tau = float(tau)
        if dt <= 0.0:
            raise ValueError("dt must be > 0")

        self.F = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

        # Standard CWNA discretization (tau scales the process noise)
        d = dt
        self.Q = tau * np.array(
            [
                [d**3 / 3.0, 0.0, d**2 / 2.0, 0.0],
                [0.0, d**3 / 3.0, 0.0, d**2 / 2.0],
                [d**2 / 2.0, 0.0, d, 0.0],
                [0.0, d**2 / 2.0, 0.0, d],
            ],
            dtype=float,
        )


class RangeMeasurementModel:
    """Range measurement model to N fixed sensors/anchors in 2D."""

    def __init__(self, sensors_xy: Sequence[Tuple[float, float]]):
        sensors_list = list(sensors_xy)
        sensors = np.asarray(sensors_list, dtype=float)
        if sensors.ndim != 2 or sensors.shape[1] != 2:
            raise ValueError("sensors_xy must be Nx2")

        if not np.all(np.isfinite(sensors)):
            raise ValueError("sensors_xy contains NaN/inf")

        self.sensors_xy = sensors_list
        self.sensors = sensors
        self.N = int(sensors.shape[0])
        self._sx = sensors[:, 0]
        self._sy = sensors[:, 1]

    def predict(self, x: float, y: float) -> np.ndarray:
        dx = x - self._sx
        dy = y - self._sy
        return np.sqrt(dx * dx + dy * dy)

    def jacobian(self, x: float, y: float) -> np.ndarray:
        dx = x - self._sx
        dy = y - self._sy
        d = np.sqrt(dx * dx + dy * dy)
        d = np.maximum(d, 1e-9)

        H = np.zeros((self.N, 4), dtype=float)
        H[:, 0] = dx / d
        H[:, 1] = dy / d
        return H


class RangeEKF:
    """
    Pure EKF engine (NO ROS):
      - holds x, P
      - initialize/reset
      - step(z, skip_predict)
    """

    def __init__(
        self,
        process: ConstantVelocityModel,
        meas: RangeMeasurementModel,
        R: np.ndarray,
        P0: np.ndarray,
        *,
        eps_S: float = 1e-9,
    ):
        self.proc = process
        self.meas = meas

        self.R = np.asarray(R, dtype=float)
        if self.R.shape != (self.meas.N, self.meas.N):
            raise ValueError(f"R shape {self.R.shape} != ({self.meas.N},{self.meas.N})")

        self.P0 = np.asarray(P0, dtype=float)
        if self.P0.shape != (4, 4):
            raise ValueError("P0 must be 4x4")

        self._eps_S = float(eps_S)

        self.x = np.zeros((4, 1), dtype=float)
        self.P = self.P0.copy()
        self.initialized = False

    def reset(self) -> None:
        self.x[:] = 0.0
        self.P = self.P0.copy()
        self.initialized = False

    def initialize(self, x0: Sequence[float]) -> None:
        x0 = np.asarray(x0, dtype=float).reshape((4, 1))
        if not np.all(np.isfinite(x0)):
            raise ValueError("x0 contains NaN/inf")
        self.x = x0
        self.P = self.P0.copy()
        self.initialized = True

    def step(self, z: np.ndarray, *, skip_predict: bool = False) -> None:
        z = _as_1d_float(z)
        if z.shape[0] != self.meas.N:
            raise ValueError(f"z length {z.shape[0]} != N {self.meas.N}")

        # ---- Predict ----
        if skip_predict:
            x_pred = self.x.copy()
            P_pred = self.P.copy()
        else:
            x_pred = self.proc.F @ self.x
            P_pred = self.proc.F @ self.P @ self.proc.F.T + self.proc.Q

        # ---- Update ----
        px = float(x_pred[0, 0])
        py = float(x_pred[1, 0])

        z_pred = self.meas.predict(px, py)
        H = self.meas.jacobian(px, py)

        innov = (z - z_pred).reshape((self.meas.N, 1))  # (N,1)
        S = H @ P_pred @ H.T + self.R                   # (N,N)
        S = 0.5 * (S + S.T)                             # enforce symmetry

        if self._eps_S > 0:
            S = S + self._eps_S * np.eye(self.meas.N, dtype=float)

        PHt = P_pred @ H.T                              # (4,N)

        # K = PHt * inv(S)  -> solve(S * K^T = PHt^T)
        try:
            K = np.linalg.solve(S, PHt.T).T             # (4,N)
        except np.linalg.LinAlgError:
            # Stronger regularization fallback
            S = S + 1e-6 * np.eye(self.meas.N, dtype=float)
            K = np.linalg.solve(S, PHt.T).T

        self.x = x_pred + K @ innov

        # Joseph form (keeps P symmetric/PSD-ish)
        I = np.eye(4, dtype=float)
        IKH = I - K @ H
        self.P = IKH @ P_pred @ IKH.T + K @ self.R @ K.T
        self.P = 0.5 * (self.P + self.P.T)
