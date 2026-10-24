"""シミュレータのノード (tma/ros_node.py) + robot_state_publisher + RViz。

    source /opt/ros/lyrical/setup.bash
    ros2 launch launch/sim.launch.py
    ros2 launch launch/sim.launch.py rviz:=false viewer:=gl   # RViz なしで Newton の OpenGL ビューア
    ros2 launch launch/sim.launch.py urdf:=cobotta.urdf       # COBOTTA 版

シミュレータのノードは uv の環境 (newton・warp) で、それ以外は ROS の Python で動かす。
"""
import os
import re
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

ROOT = Path(__file__).resolve().parent.parent
PKGS = ROOT / "packages"


def robot_description(urdf_path: Path) -> str:
    """package:// を file:// の絶対パスにする (ROS のパッケージとして入れなくても RViz がメッシュを読めるように)。
    無い .dae は .stl に読み替える (tma/sim.py の resolve_meshes と同じ)。"""

    def sub(m):
        p = PKGS / m.group(1)
        if not p.exists():
            p = p.with_suffix(".stl")
        return f'filename="file://{p.as_posix()}"'

    return re.sub(r'filename="package://([^"]+)"', sub, urdf_path.read_text(encoding="utf-8"))


def state_publisher(context):
    urdf = ROOT / LaunchConfiguration("urdf").perform(context)
    return [
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            parameters=[{"robot_description": robot_description(urdf), "use_sim_time": True}],
        )
    ]


def generate_launch_description():
    sim_time = {"use_sim_time": True}
    return LaunchDescription([
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("viewer", default_value="null", description="Newton のビューア (null / gl / viser)"),
        DeclareLaunchArgument("urdf", default_value="tma.urdf", description="リポジトリ直下の URDF (tma.urdf / cobotta.urdf)"),
        ExecuteProcess(
            cmd=["uv", "run", "tma/ros_node.py", "--viewer", LaunchConfiguration("viewer"), "--urdf", LaunchConfiguration("urdf")],
            cwd=str(ROOT),
            # リポジトリの .venv は Windows 側の環境なので、WSL では別の場所に作る
            additional_env={"UV_PROJECT_ENVIRONMENT": os.environ.get("UV_PROJECT_ENVIRONMENT", str(Path.home() / ".venvs" / "tma"))},
            output="screen",
        ),
        OpaqueFunction(function=state_publisher),
        Node(
            package="rviz2",
            executable="rviz2",
            arguments=["-d", str(ROOT / "launch" / "tma.rviz")],
            parameters=[sim_time],
            # WSLg では Qt が Wayland で動き、Ogre の GLX の描画ウィンドウが作れずに落ちる。X11 (XWayland) にする
            additional_env={"QT_QPA_PLATFORM": "xcb"},
            condition=IfCondition(LaunchConfiguration("rviz")),
        ),
    ])
