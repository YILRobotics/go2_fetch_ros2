import fcntl
import socket
import struct
import time

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

#   1. Receives camera packets from the Go2 at 230.1.1.1:1720. 
#      230.1.1.1:1720 is the network destination used by the Go2 camera
#      1720 is the UDP port carrying the video packets.
#   2. GStreamer joins those RTP packets together.
#   3. It decodes the H.264 video into normal BGR images.
#   4. Each image is converted into a ROS 2 Image message.
#   5. It publishes the result on:


class Go2SocketCameraNode(Node):
    """Decode the Go2 RTP/H.264 multicast stream and publish ROS images."""

    def __init__(self):
        super().__init__('go2_socket_camera_node')

        self.declare_parameter('multicast_group', '230.1.1.1')
        self.declare_parameter('port', 1720)
        self.declare_parameter('interface_ip', '192.168.123.18')
        self.declare_parameter('network_interface', '')
        self.declare_parameter('image_topic', '/go2/camera/image_raw')
        self.declare_parameter('frame_id', 'go2_camera_optical_frame')
        self.declare_parameter('latency_ms', 100)

        group = self.get_parameter('multicast_group').value
        port = self.get_parameter('port').value
        interface_ip = self.get_parameter('interface_ip').value
        interface = self.get_parameter('network_interface').value
        image_topic = self.get_parameter('image_topic').value
        self.frame_id = self.get_parameter('frame_id').value
        latency_ms = self.get_parameter('latency_ms').value

        if not interface and interface_ip:
            interface = self._interface_name_for_ip(interface_ip)
            if not interface:
                raise RuntimeError(
                    f'No network interface owns interface_ip={interface_ip}. '
                    'Set the network_interface parameter to the interface connected to the Go2.'
                )

        self.image_pub = self.create_publisher(
            Image, image_topic, qos_profile_sensor_data
        )
        self.bridge = CvBridge()
        self.frame_count = 0
        self.started_at = time.monotonic()
        self.warned_no_frames = False

        # The Go2 stream is RTP containing H.264 NAL units. It is not an MJPEG
        # byte stream, so JPEG marker splitting and cv2.imdecode cannot decode it.
        source_options = (
            f'address={group} port={port} auto-multicast=true '
            'buffer-size=2097152'
        )
        if interface:
            source_options += f' multicast-iface={interface}'

        pipeline_description = (
            f'udpsrc {source_options} '
            '! application/x-rtp,media=video,encoding-name=H264,clock-rate=90000 '
            f'! rtpjitterbuffer latency={latency_ms} drop-on-latency=true '
            '! rtph264depay '
            '! h264parse '
            '! avdec_h264 '
            '! videoconvert '
            '! video/x-raw,format=BGR '
            '! appsink name=camera_sink emit-signals=true sync=false '
            'max-buffers=1 drop=true'
        )

        Gst.init(None)
        try:
            self.pipeline = Gst.parse_launch(pipeline_description)
        except Exception as exc:
            raise RuntimeError(f'Failed to create camera pipeline: {exc}') from exc

        self.sink = self.pipeline.get_by_name('camera_sink')
        self.sink.connect('new-sample', self._on_sample)
        self.bus = self.pipeline.get_bus()

        state_result = self.pipeline.set_state(Gst.State.PLAYING)
        if state_result == Gst.StateChangeReturn.FAILURE:
            self.pipeline.set_state(Gst.State.NULL)
            raise RuntimeError('GStreamer rejected the camera pipeline')

        interface_text = interface or 'the system multicast route'
        self.get_logger().info(
            f'Receiving RTP/H.264 from {group}:{port} on {interface_text}; '
            f'publishing {image_topic}'
        )
        self.monitor_timer = self.create_timer(1.0, self._monitor_pipeline)

    @staticmethod
    def _interface_name_for_ip(target_ip):
        """Return the interface name owning an IPv4 address, if present."""
        for _, name in socket.if_nameindex():
            request = struct.pack('256s', name[:15].encode('utf-8'))
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                    response = fcntl.ioctl(sock.fileno(), 0x8915, request)  # SIOCGIFADDR
                if socket.inet_ntoa(response[20:24]) == target_ip:
                    return name
            except OSError:
                continue
        return None

    def _on_sample(self, sink):
        sample = sink.emit('pull-sample')
        if sample is None:
            return Gst.FlowReturn.ERROR

        caps = sample.get_caps().get_structure(0)
        width = caps.get_value('width')
        height = caps.get_value('height')
        buffer = sample.get_buffer()
        mapped, map_info = buffer.map(Gst.MapFlags.READ)
        if not mapped:
            return Gst.FlowReturn.ERROR

        try:
            expected_size = width * height * 3
            pixels = np.frombuffer(map_info.data, dtype=np.uint8)
            if pixels.size < expected_size:
                self.get_logger().error(
                    f'Decoded frame is too small: {pixels.size} < {expected_size}'
                )
                return Gst.FlowReturn.ERROR
            frame = pixels[:expected_size].reshape((height, width, 3)).copy()
        finally:
            buffer.unmap(map_info)

        image_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        image_msg.header.stamp = self.get_clock().now().to_msg()
        image_msg.header.frame_id = self.frame_id
        self.image_pub.publish(image_msg)
        self.frame_count += 1

        if self.frame_count == 1:
            self.get_logger().info(f'Publishing decoded {width}x{height} camera frames')
        return Gst.FlowReturn.OK

    def _monitor_pipeline(self):
        while True:
            message = self.bus.pop_filtered(
                Gst.MessageType.ERROR | Gst.MessageType.WARNING
            )
            if message is None:
                break
            if message.type == Gst.MessageType.ERROR:
                error, debug = message.parse_error()
                self.get_logger().error(
                    f'GStreamer camera error: {error.message}; {debug or "no details"}'
                )
            else:
                warning, debug = message.parse_warning()
                self.get_logger().warning(
                    f'GStreamer camera warning: {warning.message}; {debug or "no details"}'
                )

        if (
            self.frame_count == 0
            and not self.warned_no_frames
            and time.monotonic() - self.started_at >= 5.0
        ):
            self.warned_no_frames = True
            self.get_logger().warning(
                'No camera frames received after 5 seconds. Verify that the Go2 '
                'camera stream is enabled and that network_interface selects the '
                'robot-facing interface.'
            )

    def destroy_node(self):
        if hasattr(self, 'pipeline'):
            self.pipeline.set_state(Gst.State.NULL)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = Go2SocketCameraNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        if node is not None:
            node.get_logger().fatal(str(exc))
        else:
            print(f'Failed to start Go2 camera node: {exc}')
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
