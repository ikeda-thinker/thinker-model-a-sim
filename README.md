# TMA sim

Thinker Model A (THF + 爪, Robotiq Hand-E, RealSense D405, UR3e) の URDF と、Newton Physics での把持・ピン挿入のシミュレーション・学習。
URDF は ROS2 の think_hand_f_description の `robots/thinker-model-a` を xacro で展開したものを元に、手で直している。

## 構成

```
tma.urdf              ロボットの URDF
tma/                  シミュレータ本体と環境 (import して使う)
  sim.py              URDF を Newton に読み込んでアーム・Hand-E・THF を動かす土台 (単体でも実行できる)
  grasp.py            把持の RL 環境 (1 環境・ワーク 1 個)、アームの IK
  grasp_pile.py       ばら積み版 (1 環境)
  grasp_pile_vec.py   ばら積み版の並列版 (学習に使う)
  peg.py              ピン挿入の RL 環境 (並列版)
  ros_node.py         Sim を ROS2 ノードとして動かす
launch/               ROS2 の launch (ノード + robot_state_publisher + RViz) と RViz の設定
scripts/
  view.py             URDF をブラウザで見る
  train_ppo.py        PPO で学習・評価
  fetch_packages.sh   UR3e / Hand-E の公式メッシュを packages/ に取得
  convert_meshes.py   cad/ の STL を packages/ のメッシュに変換
cad/                  CAD から書き出したメッシュの元データ (mm)
packages/             メッシュ (package:// の解決先)
ckpt/                 学習のチェックポイント (gitignore)
```

各ファイルの先頭の docstring に実行例がある (`uv run tma/peg.py ...`、`uv run scripts/train_ppo.py ...` など)。

## セットアップ

```sh
sh scripts/fetch_packages.sh   # UR3e / Hand-E の公式メッシュを packages/ に取得
uv run scripts/view.py       # http://localhost:8080 を開く
```

別の URDF を見る場合は `uv run scripts/view.py <file.urdf>`。

## ROS2 (WSL2)

WSL2 の Ubuntu 26.04 + ROS2 Lyrical (Python 3.14 でこのプロジェクトと揃う) で動かす。

```sh
sudo apt install ros-lyrical-desktop   # apt のリポジトリ追加は ROS2 公式の手順どおり
# リポジトリの .venv は Windows 側の環境なので、WSL では別の場所に作る。
# ROS の Python パッケージ (rclpy が使う em など) は apt でシステムに入るので、システムの site-packages も見えるようにする
export UV_PROJECT_ENVIRONMENT=$HOME/.venvs/tma   # ~/.bashrc に書いておく
uv venv --python /usr/bin/python3 --system-site-packages $UV_PROJECT_ENVIRONMENT
uv sync

source /opt/ros/lyrical/setup.bash
ros2 launch launch/sim.launch.py               # ノード + robot_state_publisher + RViz
uv run tma/ros_node.py --viewer null           # ノードだけ
```

トピックは `/clock`・`/joint_states`・`/joint_command` (関節目標、JointState)・`/tf` (箱などロボット以外の物体)。詳しくは `tma/ros_node.py` の docstring。

```sh
ros2 topic pub --once /joint_command sensor_msgs/msg/JointState \
  "{name: [thinker_arm_shoulder_pan_joint, robotiq_hande_left_finger_joint], position: [0.5, 0.02]}"
```

## メッシュ

`package://<name>/...` は `packages/<name>/...` として解決される。

| パッケージ | 状態 |
|---|---|
| `ur_description` | 公式。`fetch_packages.sh` で取得 (gitignore) |
| `robotiq_hande_description` | 非公式 (macmacal)。`fetch_packages.sh` で取得 (gitignore) |
| `realsense2_description` | realsense-ros 由来の D405 (`cad/d405.stl`) を前面と筐体に分けたものをコミット済み |
| `think_hand_f_description` | 爪 (`pile-nail`)・THF (`thf`)・Hand-E とのアダプタ (`hande-adapter`) は CAD から変換。残りは**仮の箱型メッシュ (STL)** で形状は推定。本物の `.dae` を同じパスに置けば優先される |

CAD のメッシュは `cad/*.stl` (Fusion 360 から書き出した mm 単位) から `uv run scripts/convert_meshes.py` で再生成できる。仮の箱型メッシュはコミット済みのものをそのまま使う。
