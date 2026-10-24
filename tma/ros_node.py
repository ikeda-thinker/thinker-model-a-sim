"""Sim を ROS2 ノードとして動かす (WSL2 / Ubuntu 26.04 + ROS2 Lyrical)。

    source /opt/ros/lyrical/setup.bash
    uv run tma/ros_node.py --viewer null            # ノードだけ
    ros2 launch launch/sim.launch.py                # ノード + robot_state_publisher + RViz

トピック:
    /clock          rosgraph_msgs/Clock      シミュレーション時刻 (他のノードは use_sim_time:=true で使う)
    /joint_states   sensor_msgs/JointState   全可動関節の位置 [rad, m]・速度。ネイルの x/y/z もここに入る (z は押し込み量 [m])
    /joint_command  sensor_msgs/JointState   関節目標 (name と position のみ使う)。アーム・Hand-E の指・ネイルの平衡位置を指定できる。
                                             Hand-E は左右どちらの指の名前でも両指に同じ目標を入れる (URDF の mimic)
    /tf             base_link から、ロボット以外の物体 (箱など) への変換
ロボットの TF と /robot_description は robot_state_publisher が出す (launch/sim.launch.py)。
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # リポジトリ直下 (tma パッケージ)

import argparse
import re
import signal

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster

import newton
import newton.examples
from tma import sim

ROOT_FRAME = "base_link"  # tma.urdf のルート。Newton のワールド原点と一致する


def time_msg(t):
    from builtin_interfaces.msg import Time

    sec = int(t)
    return Time(sec=sec, nanosec=int((t - sec) * 1e9))


class SimNode(Node):
    def __init__(self, s: sim.Sim, args):
        super().__init__("tma_sim")
        self.sim = s
        self.realtime = not args.fast
        model = s.model
        labels = list(model.joint_label)
        jtype = model.joint_type.numpy()
        q_start, qd_start = model.joint_q_start.numpy(), model.joint_qd_start.numpy()
        # 可動関節 (いずれも 1 自由度) の URDF 名と q / dof インデックス。Newton のラベルは "<モデル名>/<関節名>" なので末尾を取る
        self.joints = [
            (l.rsplit("/", 1)[-1], int(q_start[j]), int(qd_start[j]))
            for j, l in enumerate(labels)
            if jtype[j] in (int(newton.JointType.REVOLUTE), int(newton.JointType.PRISMATIC))
        ]
        self.dof_of = {name: d for name, _, d in self.joints}
        # ロボット以外の body (URDF のリンクに無いもの) は TF で出す
        urdf_links = set(re.findall(r'<link name="([^"]+)"', (sim.ROOT / args.urdf).read_text(encoding="utf-8")))
        self.objects = [(b, l.rsplit("/", 1)[-1]) for b, l in enumerate(model.body_label) if l.rsplit("/", 1)[-1] not in urdf_links]

        self.pub_clock = self.create_publisher(Clock, "clock", 10)
        self.pub_js = self.create_publisher(JointState, "joint_states", 10)
        self.tf = TransformBroadcaster(self)
        self.create_subscription(JointState, "joint_command", self.on_command, 10)
        self.get_logger().info(f"joints: {[n for n, _, _ in self.joints]}")
        self.get_logger().info(f"objects: {[n for _, n in self.objects]}")

    def on_command(self, msg: JointState):
        tgt = self.sim.control.joint_target_q.numpy()
        for name, pos in zip(msg.name, msg.position):
            if name in self.sim.robot.gripper_joints:
                # 左右の目標を揃えないと mimic 拘束と綱引きになる (Sim.set_gripper)
                tgt[self.sim.gripper_dof] = tgt[self.sim.gripper_r_dof] = np.clip(pos, 0.0, self.sim.robot.stroke)
            elif name in self.dof_of:
                tgt[self.dof_of[name]] = pos
            else:
                self.get_logger().warn(f"unknown joint: {name}", throttle_duration_sec=5.0)
        # assign で中身だけ書き換える (CUDA グラフが配列を指しているため)
        self.sim.control.joint_target_q.assign(tgt)

    def publish(self):
        stamp = time_msg(self.sim.sim_time)
        self.pub_clock.publish(Clock(clock=stamp))
        q, qd = self.sim.state_0.joint_q.numpy(), self.sim.state_0.joint_qd.numpy()
        js = JointState()
        js.header.stamp = stamp
        js.name = [n for n, _, _ in self.joints]
        js.position = [float(q[i]) for _, i, _ in self.joints]
        js.velocity = [float(qd[d]) for _, _, d in self.joints]
        self.pub_js.publish(js)
        if self.objects:
            body_q = self.sim.state_0.body_q.numpy()
            tfs = []
            for b, name in self.objects:
                t = TransformStamped()
                t.header.stamp, t.header.frame_id, t.child_frame_id = stamp, ROOT_FRAME, name
                p, r = body_q[b, :3], body_q[b, 3:]  # クォータニオンは xyzw
                t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = map(float, p)
                t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = map(float, r)
                tfs.append(t)
            self.tf.sendTransform(tfs)

    def run(self, num_frames):
        start = time.perf_counter()
        frame = 0
        while rclpy.ok() and self.sim.viewer.is_running() and frame < num_frames:
            rclpy.spin_once(self, timeout_sec=0.0)
            self.sim.step()
            self.publish()
            self.sim.render()
            frame += 1
            if self.realtime:  # 実時間より速く進めない (遅い分は追いつかない)
                lag = start + self.sim.sim_time - time.perf_counter()
                if lag > 0:
                    time.sleep(lag)


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--urdf", default="tma.urdf")
    parser.add_argument("--box", action=argparse.BooleanOptionalAction, default=True, help="把持対象の箱を置く")
    # newton の --realtime はプロセス優先度の指定で別物
    parser.add_argument("--fast", action="store_true", help="実時間に合わせず、計算できる速さで進める")
    parser.set_defaults(num_frames=10**12)
    # --ros-args 以降 (ros2 launch が付けるリマップなど) は rclpy に渡し、newton の引数解析からは外す
    ros_args = sys.argv[sys.argv.index("--ros-args") :] if "--ros-args" in sys.argv else []
    sys.argv = sys.argv[: len(sys.argv) - len(ros_args)]
    viewer, args = newton.examples.init(parser)
    # rclpy のシグナルハンドラは使わない (Ctrl+C で先にコンテキストが閉じ、ループ中の publish が例外になる)。
    # KeyboardInterrupt で抜けて下で後始末する。SIGINT を無視する設定で起動されても (バックグラウンドのジョブなど) 止められるよう明示する
    signal.signal(signal.SIGINT, signal.default_int_handler)
    rclpy.init(args=[sys.argv[0], *ros_args], signal_handler_options=SignalHandlerOptions.NO)
    node = SimNode(sim.Sim(viewer, args), args)
    try:
        node.run(args.num_frames)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
        viewer.close()
