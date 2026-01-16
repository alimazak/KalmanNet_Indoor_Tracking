#!/usr/bin/env python3
from __future__ import annotations

import sys
import math
import time
import select
import termios
import tty
from dataclasses import dataclass

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.parameter_client import AsyncParametersClient
from rclpy.utilities import remove_ros_args

from rcl_interfaces.msg import ParameterType


def _pv_to_py(v):
    """rcl_interfaces/ParameterValue -> python value"""
    t = v.type
    if t == ParameterType.PARAMETER_BOOL:
        return bool(v.bool_value)
    if t == ParameterType.PARAMETER_INTEGER:
        return int(v.integer_value)
    if t == ParameterType.PARAMETER_DOUBLE:
        return float(v.double_value)
    if t == ParameterType.PARAMETER_STRING:
        return str(v.string_value)
    # arrays not needed here
    return None


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


@dataclass
class Tunables:
    sigma_meas: float
    sigma_ekf: float
    tau: float
    delta: float
    rate: float | None = None  # optional


class EKFParamTeleop(Node):
    """
    Keyboard teleop that sets ROS2 parameters on:
      - /tracking/ekf
      - /tracking/range_measurement_generator

    Defaults assume your launch names:
      ekf node name: /tracking/ekf
      range node:    /tracking/range_measurement_generator
    """

    def __init__(self, ekf_node: str, range_node: str):
        super().__init__("ekf_param_teleop")

        self.ekf_node = ekf_node
        self.range_node = range_node

        self.ekf = AsyncParametersClient(self, self.ekf_node)
        self.rng = AsyncParametersClient(self, self.range_node)

        self._wait_for_services()

        # read current params as baseline
        self.base = self._read_current()
        self.cur = Tunables(**vars(self.base))  # copy

        # steps
        self.mult_step = 1.25   # multiplicative step for sigma/tau
        self.delta_step = 0.01  # additive step for delta

        # presets (your controlled experiments)
        self.presets: dict[str, Tunables] = {
            "1": Tunables(sigma_meas=0.50, sigma_ekf=0.05, tau=1.0, delta=0.1, rate=10.0),
            "2": Tunables(sigma_meas=0.05, sigma_ekf=0.50, tau=1.0, delta=0.1, rate=10.0),
            "3": Tunables(sigma_meas=0.10, sigma_ekf=0.10, tau=1.0, delta=0.2, rate=10.0),
            # "4" is same as 3 in your message; keep it for convenience:
            "4": Tunables(sigma_meas=0.10, sigma_ekf=0.10, tau=1.0, delta=0.2, rate=10.0),
        }

        self._print_help()
        self._print_state(prefix="[START] ")

    def _wait_for_services(self):
        self.get_logger().info(f"Waiting param services: ekf='{self.ekf_node}', range='{self.range_node}'")
        for _ in range(200):
            if self.ekf.service_is_ready() and self.rng.service_is_ready():
                self.get_logger().info("Parameter services ready ✅")
                return
            rclpy.spin_once(self, timeout_sec=0.05)
        raise RuntimeError("Parameter services not ready. Check node names / namespace.")

    def _get(self, client: AsyncParametersClient, names: list[str], timeout: float = 1.0) -> dict[str, object]:
        fut = client.get_parameters(names)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=timeout)
        if fut.result() is None:
            raise RuntimeError(f"get_parameters timeout for {client.remote_node_name}")
        resp = fut.result()
        out = {}
        for name, pv in zip(names, resp.values):
            out[name] = _pv_to_py(pv)
        return out

    def _set(self, client: AsyncParametersClient, params: list[Parameter], timeout: float = 1.0) -> bool:
        fut = client.set_parameters(params)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=timeout)
        if fut.result() is None:
            self.get_logger().warn(f"set_parameters timeout for {client.remote_node_name}")
            return False
        resp = fut.result()
        ok = all(r.successful for r in resp.results)
        if not ok:
            reasons = [r.reason for r in resp.results if not r.successful]
            self.get_logger().warn(f"set_parameters failed: {reasons}")
        return ok

    def _read_current(self) -> Tunables:
        ekf_vals = self._get(self.ekf, ["sigma", "tau", "delta"])
        rng_vals = self._get(self.rng, ["sigma"])  # meas sigma
        # rate is optional; try, but don't die if not declared
        rate = None
        try:
            rate_vals = self._get(self.rng, ["rate"])
            rate = float(rate_vals["rate"])
        except Exception:
            rate = None

        return Tunables(
            sigma_meas=float(rng_vals["sigma"]),
            sigma_ekf=float(ekf_vals["sigma"]),
            tau=float(ekf_vals["tau"]),
            delta=float(ekf_vals["delta"]),
            rate=rate,
        )

    def _apply(self):
        # EKF params
        ok1 = self._set(self.ekf, [
            Parameter("sigma", Parameter.Type.DOUBLE, float(self.cur.sigma_ekf)),
            Parameter("tau",   Parameter.Type.DOUBLE, float(self.cur.tau)),
            Parameter("delta", Parameter.Type.DOUBLE, float(self.cur.delta)),
        ])

        # Range generator params (true measurement noise)
        ok2 = self._set(self.rng, [
            Parameter("sigma", Parameter.Type.DOUBLE, float(self.cur.sigma_meas)),
        ])

        # Rate (optional)
        ok3 = True
        if self.cur.rate is not None:
            ok3 = self._set(self.rng, [
                Parameter("rate", Parameter.Type.DOUBLE, float(self.cur.rate)),
            ])

        self._print_state(prefix="[APPLY] " if (ok1 and ok2 and ok3) else "[WARN] ")

    def _print_state(self, prefix: str = ""):
        r = f"{self.cur.rate:.2f}" if self.cur.rate is not None else "n/a"
        print(
            f"{prefix}"
            f"EKF(sigma={self.cur.sigma_ekf:.4f}, tau={self.cur.tau:.4f}, delta={self.cur.delta:.3f}) | "
            f"MEAS(sigma_meas={self.cur.sigma_meas:.4f}, rate={r})"
        )

    def _print_help(self):
        print("\nEKF Param Teleop 🎛️")
        print("  h/? : help")
        print("  p   : print current (reads from nodes)")
        print("  0   : reset to baseline (values at script start)")
        print("  1/2/3/4 : apply experiment presets")
        print("")
        print("  [EKF assumed measurement noise]")
        print("   q : sigma_ekf *= 1.25      a : sigma_ekf /= 1.25")
        print("")
        print("  [EKF process noise / aggressiveness]")
        print("   w : tau *= 1.25           s : tau /= 1.25")
        print("")
        print("  [EKF dt mismatch]")
        print("   e : delta += 0.01         d : delta -= 0.01")
        print("")
        print("  [True measurement noise (range generator)]")
        print("   r : sigma_meas *= 1.25    f : sigma_meas /= 1.25")
        print("")
        print("CTRL-C to quit\n")

    def handle_key(self, k: str):
        k = k.strip()
        if not k:
            return

        if k in ("h", "?"):
            self._print_help()
            return

        if k == "p":
            self.cur = self._read_current()
            self._print_state(prefix="[READ] ")
            return

        if k == "0":
            self.cur = Tunables(**vars(self.base))
            self._apply()
            return

        if k in self.presets:
            self.cur = Tunables(**vars(self.presets[k]))
            # clamp sanity
            self.cur.sigma_meas = _clamp(self.cur.sigma_meas, 1e-4, 5.0)
            self.cur.sigma_ekf  = _clamp(self.cur.sigma_ekf,  1e-4, 5.0)
            self.cur.tau        = _clamp(self.cur.tau,        1e-4, 50.0)
            self.cur.delta      = _clamp(self.cur.delta,      0.01, 0.5)
            if self.cur.rate is not None:
                self.cur.rate = _clamp(self.cur.rate, 1.0, 100.0)
            self._apply()
            return

        # step changes
        if k == "q":
            self.cur.sigma_ekf *= self.mult_step
        elif k == "a":
            self.cur.sigma_ekf /= self.mult_step
        elif k == "w":
            self.cur.tau *= self.mult_step
        elif k == "s":
            self.cur.tau /= self.mult_step
        elif k == "e":
            self.cur.delta += self.delta_step
        elif k == "d":
            self.cur.delta -= self.delta_step
        elif k == "r":
            self.cur.sigma_meas *= self.mult_step
        elif k == "f":
            self.cur.sigma_meas /= self.mult_step
        else:
            return

        # clamp sanity
        self.cur.sigma_meas = _clamp(self.cur.sigma_meas, 1e-4, 5.0)
        self.cur.sigma_ekf  = _clamp(self.cur.sigma_ekf,  1e-4, 5.0)
        self.cur.tau        = _clamp(self.cur.tau,        1e-4, 50.0)
        self.cur.delta      = _clamp(self.cur.delta,      0.01, 0.5)

        self._apply()


def main():
    argv = sys.argv
    # allow normal args before --ros-args if you want later
    _ = remove_ros_args(argv)

    rclpy.init(args=argv)

    # these are your default node names
    ekf_node = "/tracking/ekf"
    range_node = "/tracking/range_measurement_generator"

    node = EKFParamTeleop(ekf_node=ekf_node, range_node=range_node)

    # raw keyboard
    if not sys.stdin.isatty():
        raise RuntimeError("This tool needs a TTY. Run from a normal terminal.")

    settings = termios.tcgetattr(sys.stdin)
    tty.setraw(sys.stdin.fileno())

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.0)
            r, _, _ = select.select([sys.stdin], [], [], 0.1)
            if r:
                ch = sys.stdin.read(1)
                node.handle_key(ch)
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
