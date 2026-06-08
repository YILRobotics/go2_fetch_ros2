#!/usr/bin/env python3
"""State-machine node orchestrating standup, policy, and search behavior."""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String


class StateMachineNode(Node):
    """Finite state machine for behavior arbitration."""

    ST_STANDUP = 'standup'
    ST_POLICY = 'policy'
    ST_SEARCH = 'search'

    def __init__(self) -> None:
        super().__init__('state_machine_node')
        self._declare_parameters()

        self.cube_visible_topic = self.get_parameter('cube_visible_topic').value
        self.mode_topic = self.get_parameter('mode_topic').value
        self.state_topic = self.get_parameter('state_topic').value
        self.tracking_lost_topic = self.get_parameter('tracking_lost_topic').value

        self.tick_rate_hz = float(self.get_parameter('tick_rate_hz').value)
        self.standup_duration_s = float(self.get_parameter('standup_duration_s').value)
        self.cube_lost_timeout_s = float(self.get_parameter('cube_lost_timeout_s').value)
        self.cube_reacquire_hold_s = float(self.get_parameter('cube_reacquire_hold_s').value)

        now_s = self.get_clock().now().nanoseconds * 1e-9
        self.state = self.ST_STANDUP
        self._state_enter_time_s = now_s

        self._cube_visible = False
        self._last_visible_time_s = -1e9
        self._last_invisible_time_s = now_s

        self._mode_pub = self.create_publisher(String, self.mode_topic, 10)
        self._state_pub = self.create_publisher(String, self.state_topic, 10)
        self._tracking_lost_pub = self.create_publisher(Bool, self.tracking_lost_topic, 10)

        self.create_subscription(Bool, self.cube_visible_topic, self._cube_visible_cb, 20)
        self._timer = self.create_timer(1.0 / self.tick_rate_hz, self._tick)

        self.get_logger().info('State machine started in STANDUP mode.')

    def _declare_parameters(self) -> None:
        self.declare_parameter('cube_visible_topic', '/go2_fetch/cube_visible')
        self.declare_parameter('mode_topic', '/go2_fetch/mode')
        self.declare_parameter('state_topic', '/go2_fetch/state')
        self.declare_parameter('tracking_lost_topic', '/go2_fetch/cube_tracking_lost')

        self.declare_parameter('tick_rate_hz', 10.0)
        self.declare_parameter('standup_duration_s', 3.0)
        self.declare_parameter('cube_lost_timeout_s', 0.5)
        self.declare_parameter('cube_reacquire_hold_s', 0.25)

    def _cube_visible_cb(self, msg: Bool) -> None:
        now_s = self.get_clock().now().nanoseconds * 1e-9
        self._cube_visible = bool(msg.data)
        if self._cube_visible:
            self._last_visible_time_s = now_s
        else:
            self._last_invisible_time_s = now_s

    # Loop function
    def _tick(self) -> None:
        now_s = self.get_clock().now().nanoseconds * 1e-9

        if self.state == self.ST_STANDUP:
            if now_s - self._state_enter_time_s >= self.standup_duration_s:
                if self._is_cube_recently_visible(now_s):
                    self._transition_to(self.ST_POLICY, now_s)
                else:
                    self._transition_to(self.ST_SEARCH, now_s)

        elif self.state == self.ST_POLICY:
            if not self._is_cube_recently_visible(now_s):
                self._transition_to(self.ST_SEARCH, now_s)

        elif self.state == self.ST_SEARCH:
            if self._cube_visible and (now_s - self._last_visible_time_s) <= self.cube_reacquire_hold_s:
                # wait for the hold window to confirm stable detection
                pass
            elif self._cube_visible and (now_s - self._last_invisible_time_s) >= self.cube_reacquire_hold_s:
                self._transition_to(self.ST_POLICY, now_s)

        self._publish_outputs(now_s)

    def _is_cube_recently_visible(self, now_s: float) -> bool:
        return (now_s - self._last_visible_time_s) <= self.cube_lost_timeout_s

    def _transition_to(self, next_state: str, now_s: float) -> None:
        if next_state == self.state:
            return
        self.state = next_state
        self._state_enter_time_s = now_s
        self.get_logger().info(f'Transitioned to: {self.state}')

    def _publish_outputs(self, now_s: float) -> None:
        del now_s

        mode_msg = String()
        mode_msg.data = self.state
        self._mode_pub.publish(mode_msg)

        state_msg = String()
        state_msg.data = self.state
        self._state_pub.publish(state_msg)

        lost_msg = Bool()
        lost_msg.data = (self.state == self.ST_SEARCH)
        self._tracking_lost_pub.publish(lost_msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = StateMachineNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
