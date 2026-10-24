"""Run tma.urdf in Newton Physics.

    uv run tma/sim.py                 # OpenGL viewer
    uv run tma/sim.py --viewer viser  # browser viewer (http://localhost:8080)
    uv run tma/sim.py --viewer null --num-frames 500   # headless smoke test
    uv run tma/sim.py --urdf cobotta.urdf             # COBOTTA 版 (scripts/make_cobotta_urdf.py)
"""
import argparse
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton import JointTargetMode
from newton._src.solvers.mujoco.constants import SOLREF_MODE_FORCE_SPACE

NAIL_PUSH_MAX_MM = 3.9  # 最大押し込み量。z_joint の値そのまま 0〜3.9mm

# ネイルは目標 0 のバネ (受動) として扱い、変位が接触力を反映する触覚センサになるようにする。
# 剛性・減衰は仮の値 (実機の値が分かれば差し替える)
# x,y 関節 [N*m/rad]。爪先は回転軸から 15mm ほど中央側にずれているので、柔らかいと上から押したとき
# 爪が回って逃げ、押し込み (z) に力が乗らない。1.0 では逃げたので 10 にした
NAIL_ROT_KE = 10.0
NAIL_ROT_KD = 0.1
# z 関節 [N/m]。全押し込み 3.9mm で約 7.8N。
# 2000 N/m (= 2.0 N/mm) は think_hand_f_transient_sim/sim_main.py の SPRING_K_LEFT/RIGHT に合わせた
NAIL_Z_KE = 2000.0
# 把持時の目標押し込み量 [mm]。押し込みは正。同 sim_main.py の DELTA_REF の 0.5mm (約 1N)
NAIL_PUSH_REF_MM = 0.5
NAIL_Z_KD = 5.0

# 把持対象の箱。アーム初期姿勢 (全関節 0) での爪の間に置く。座標は実測 (FK + 爪の衝突メッシュ) から決めた
BOX_SIZE = 0.05  # 一辺 [m]
BOX_MASS = 0.1  # [kg]
BOX_POS = (0.4607, 0.405, 0.0665)  # 爪の x 中点、爪先の y、グリッパの z
# 接触の剛性・減衰 [N/m], [N*s/m]。Newton の force-space 指定 (質量に依存しない)。
# MuJoCo 既定の接触は剛さが加速度の単位で決まり、軽い爪 (10-30g) では数百 N/m 相当と柔らかく、
# 爪のバネ (2000 N/m) より柔らかいため変位の大半がめり込みに回っていた
CONTACT_KE = 5e4
CONTACT_KD = 200.0
MIMIC_TIMECONST = 0.004  # 指の連動拘束の時定数 [s]。MuJoCo の制約は 2*sim_dt (1/1200 s) 以上でないと不安定
# 接触を拾う距離 [m]。既定は 0.1 で、小物の山だと触れていないペアまで接触に数えて重くなる (30 個で約 1400 点)
CONTACT_GAP = 0.003
HANDE_STROKE = 0.025 # Hande の片指ストローク [m]。関節値 0 が閉、0.025 が全開
# 爪の接触。爪は金属で、先端に滑り止めのゴムが貼ってある (当たり判定は先端の区間が別の凸包)。
# MuJoCo の摩擦はペアの大きい方を使うので、金属側の値は物体の摩擦 (grasp_pile.MU) より小さければ物体側の値になる。
# 既定の condim=3 はねじり摩擦が無く、2 点の挟み込みだと物体が指先を結ぶ軸まわりに回って滑るので、滑り止めは condim=4
NAIL_MU = 0.4  # 金属 (ステンレス)
# 滑り止め。実機は NITTO の黒テープ (薄く剥がれにくい)。以前の 3M の滑り止めテープは摩擦が大きすぎて
# 部品の間に指先を差し込みにくかったので変えた、とのこと。正確な値は不明で目安
NAIL_PAD_MU = 1.0
NAIL_PAD_MU_TORSIONAL = 0.005  # [m] 接触面の半径くらい
NAIL_PAD_CONDIM = 4

ROOT = Path(__file__).resolve().parent.parent
PKGS = ROOT / "packages"


@dataclass(frozen=True)
class Robot:
    """URDF ごとに違う所。アームの関節は "thinker_arm_" で始まる名前で探す (どの URDF も同じ)。"""

    gripper_joints: tuple  # (左指, 右指)。どちらも 0 が閉で正に開き、右指は URDF の mimic で左指に追従する
    stroke: float  # 片指のストローク [m]
    arm_q_deg: tuple | None  # 初期姿勢 [deg]。None なら全関節 0
    box_pos: tuple  # 把持対象の箱の位置 (爪の間)
    box_size: float = BOX_SIZE  # 箱の一辺 [m]。密度は BOX_MASS / BOX_SIZE^3 のまま


ROBOTS = {
    "tma": Robot(("robotiq_hande_left_finger_joint", "robotiq_hande_right_finger_joint"), HANDE_STROKE, None, BOX_POS),
    # COBOTTA + 純正グリッパ (scripts/make_cobotta_urdf.py)。全閉 0、片指 15mm で全開 (爪先のすきま約 34mm)。
    # 姿勢は前腕を水平に、手先を真下に向けたもの。箱はこの姿勢の FK で爪先の中点 (0.1775, -0.0445, 0.1393) に中心を置く
    "cobotta": Robot(
        ("cobotta_gripper_left_finger_joint", "cobotta_gripper_right_finger_joint"),
        0.015,
        (0.0, 0.0, 90.0, 0.0, 90.0, 0.0),
        (0.1775, -0.0445, 0.1393),
        box_size=0.02,
    ),
}


def resolve_meshes(urdf_path: Path) -> str:
    """Rewrite package:// URIs to absolute paths (falling back to placeholder .stl)."""

    def sub(m):
        p = PKGS / m.group(1)
        if not p.exists():
            p = p.with_suffix(".stl")
        return f'filename="{p.as_posix()}"'

    return re.sub(r'filename="package://([^"]+)"', sub, urdf_path.read_text(encoding="utf-8"))


def build_robot(urdf: str) -> newton.ModelBuilder:
    robot = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(robot)
    # newton の URDF パーサは絶対パスのみ確実に解決できるので、一時ファイルに書き出して渡す
    with tempfile.NamedTemporaryFile("w", suffix=".urdf", delete=False, encoding="utf-8") as f:
        f.write(resolve_meshes(ROOT / urdf))
    robot.add_urdf(
        f.name,
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        floating=False,  # UR3e のベースは固定
        enable_self_collisions=False,
        joint_damping_mode="passive",
    )
    Path(f.name).unlink(missing_ok=True)
    # 色指定のないパーツには newton が形状ごとにパレットの派手な色を振るので、灰色に差し替える (material 指定のあるパーツはそのまま)
    for s, c in enumerate(robot.shape_color):
        if tuple(c) == tuple(newton.ModelBuilder._shape_palette_color(s)):
            robot.shape_color[s] = (0.6, 0.6, 0.6)
    # 自己衝突は切っているが、左右の爪同士だけは当てる (実機では全閉の手前で爪先がぶつかる)
    nail = {
        f: {s for s, b in enumerate(robot.shape_body) if b >= 0 and robot.body_label[b].endswith(f"{f}_thf_nail_link")}
        for f in ("left", "right")
    }
    robot.shape_collision_filter_pairs = [
        (a, b) for a, b in robot.shape_collision_filter_pairs
        if not ((a in nail["left"] and b in nail["right"]) or (a in nail["right"] and b in nail["left"]))
    ]
    return robot


class Sim:
    # アームの初期姿勢・目標 [deg] (shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3)。None なら全関節 0
    ARM_Q_DEG = None
    use_graph = True
    # SolverMuJoCo への追加引数 (iterations など)。impratio は摩擦方向の拘束を法線方向より硬くする比で、既定の 1 だと
    # 握って持ち上げた物がじわじわ滑り落ちる。バラ積みのスクリプト方策の成功率は 1 で 0.20、5 で 0.52、10 で 0.59、20 で 0.49、30 で 0.39
    SOLVER_KW = dict(impratio=10.0)
    # Newton の接触バッファ (1 環境あたり)。既定の見積もりは形状のペア数ベースで約 11000 点 (環境数に比例して GPU メモリを食う) だが、
    # バラ積みでの実測は 1 環境あたり最大 166 点。None なら既定の見積もり
    RIGID_CONTACTS_PER_ENV = None
    NJMAX, NCONMAX = 512, 256  # MuJoCo の拘束・接触バッファ。物体が多いときはサブクラスで増やす
    CAMERA_TARGET = None  # None なら箱の位置 (Robot.box_pos)

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.robot = ROBOTS[Path(args.urdf).stem]
        if self.CAMERA_TARGET is None:
            self.CAMERA_TARGET = self.robot.box_pos
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.substeps = 20  # 接触を硬くしたので細かく (MuJoCo は時定数を 2*dt 未満にできない)
        self.sim_dt = self.frame_dt / self.substeps
        self.sim_time = 0.0

        builder = build_robot(args.urdf)
        # 全関節を位置制御 (PD) にする。ゲイン値は仮 (ネイルは下でバネとして上書き)
        for i in range(len(builder.joint_target_ke)):
            builder.joint_target_ke[i] = 500.0
            builder.joint_target_kd[i] = 50.0
            builder.joint_target_mode[i] = int(JointTargetMode.POSITION)
        for j, label in enumerate(builder.joint_label):
            if builder.joint_type[j] != int(newton.JointType.FIXED) and (
                "thinker_arm_" in label or label.endswith(self.robot.gripper_joints)
            ):
                d = builder.joint_qd_start[j]
                # アームは剛に固定、グリッパは力で閉じる
                arm = "thinker_arm_" in label
                builder.joint_target_ke[d] = 20000.0 if arm else 2000.0
                builder.joint_target_kd[d] = 2000.0 if arm else 100.0
            m = re.search(r"(?:left|right)_thf_([xyz])_joint$", label)
            if m:
                d = builder.joint_qd_start[j]
                rot = m.group(1) != "z"
                builder.joint_target_ke[d] = NAIL_ROT_KE if rot else NAIL_Z_KE
                builder.joint_target_kd[d] = NAIL_ROT_KD if rot else NAIL_Z_KD
        self.arm_q_index = [
            builder.joint_q_start[j]
            for j, label in enumerate(builder.joint_label)
            if "thinker_arm_" in label and builder.joint_type[j] != int(newton.JointType.FIXED)
        ]
        arm_q_deg = self.ARM_Q_DEG if self.ARM_Q_DEG is not None else self.robot.arm_q_deg
        if arm_q_deg is not None:
            for k, deg in zip(self.arm_q_index, arm_q_deg):
                builder.joint_q[k] = np.radians(deg)
        builder.add_ground_plane()
        self.add_scene(builder)
        # 環境数 (--num-envs)。1 環境ぶんを組んだ builder を複製して、独立したワールドにする。
        # 以降のインデックス (nail_q_index など) はワールド 0 のもの。ワールド w のぶんは w * (1 ワールドの q 数 / dof 数 / body 数) を足す
        self.num_envs = int(getattr(args, "num_envs", 1))
        if self.num_envs > 1:
            multi = newton.ModelBuilder()
            newton.solvers.SolverMuJoCo.register_custom_attributes(multi)
            multi.replicate(builder, self.num_envs)
            builder = multi

        self.model = builder.finalize()
        # ネイル関節 (left/right の x,y,z) の q / dof インデックス。全て 1 自由度なので同じ並び
        labels = list(self.model.joint_label)
        q_start = self.model.joint_q_start.numpy()
        qd_start = self.model.joint_qd_start.numpy()
        self.nail_q_index, self.nail_dof_index = {}, {}
        for f in ("left", "right"):
            for a in ("x", "y", "z"):
                j = next(i for i, l in enumerate(labels) if l.endswith(f"{f}_thf_{a}_joint"))
                self.nail_q_index[f"{f}_{a}"] = int(q_start[j])
                self.nail_dof_index[f"{f}_{a}"] = int(qd_start[j])

        gj = next(i for i, l in enumerate(labels) if l.endswith(self.robot.gripper_joints[0]))
        self.gripper_q, self.gripper_dof = int(q_start[gj]), int(qd_start[gj])
        gr = next(i for i, l in enumerate(labels) if l.endswith(self.robot.gripper_joints[1]))
        self.gripper_r_dof = int(qd_start[gr])  # 右指も左指と同じ目標にする (下の set_gripper)
        self.gripper_rq = int(q_start[gr])
        # グリッパは全開から始める。閉じた (関節値 0) まま始めると、爪の間に置いた箱に爪がめり込んで弾き飛ばす
        q = self.model.joint_q.numpy()
        per_world = len(q) // self.num_envs
        for w in range(self.num_envs):
            q[[w * per_world + self.gripper_q, w * per_world + self.gripper_rq]] = self.robot.stroke
        self.model.joint_q.assign(q)
        self.box_body = next((i for i, l in enumerate(self.model.body_label) if l.endswith("box")), None)

        self.model.mujoco.solref_mode.fill_(SOLREF_MODE_FORCE_SPACE)
        self.model.shape_material_ke.fill_(CONTACT_KE)
        self.model.shape_material_kd.fill_(CONTACT_KD)
        self.model.shape_gap.fill_(CONTACT_GAP)
        body_label = list(self.model.body_label)
        shape_body = self.model.shape_body.numpy()
        collide = self.model.shape_flags.numpy() & int(newton.ShapeFlags.COLLIDE_SHAPES)
        nail_shapes = [s for s, b in enumerate(shape_body) if b >= 0 and collide[s] and body_label[b].endswith("_nail_link")]
        # 滑り止めは爪ごとに最も先端側 (nail_link の z が大きい) の当たり判定
        tf = self.model.shape_transform.numpy()

        def max_z(s):  # 形状の頂点を body 座標に直したときの z の最大 (クォータニオンは xyzw)
            v = np.asarray(self.model.shape_source[s].vertices)
            q, w = tf[s, 3:6], tf[s, 6]
            t = 2.0 * np.cross(q, v)
            return float((v + w * t + np.cross(q, t))[:, 2].max() + tf[s, 2])

        tip_z = {s: max_z(s) for s in nail_shapes}
        pad_shapes = [max((s for s in nail_shapes if shape_body[s] == b), key=tip_z.get) for b in {shape_body[s] for s in nail_shapes}]
        for arr, shapes, v in (
            (self.model.shape_material_mu, nail_shapes, NAIL_MU),
            (self.model.shape_material_mu, pad_shapes, NAIL_PAD_MU),
            (self.model.shape_material_mu_torsional, pad_shapes, NAIL_PAD_MU_TORSIONAL),
            (self.model.mujoco.condim, pad_shapes, NAIL_PAD_CONDIM),
        ):
            a = arr.numpy()
            a[shapes] = v
            arr.assign(a)
        # force-space の接触剛性は Newton の接触 (model.collide) を使うときだけ効く
        self.solver = newton.solvers.SolverMuJoCo(
            self.model, use_mujoco_contacts=False, njmax=self.NJMAX, nconmax=self.NCONMAX, **self.SOLVER_KW
        )
        # Hande の左右の指は URDF の mimic で連動させている。MuJoCo の等式拘束の既定 (時定数 0.02 s) だと柔らかすぎて、
        # 把持物を持ち上げる加速で左右差が ±14mm まで揺れた。固くすると ±1.6mm
        eq_solref = self.solver.mjw_model.eq_solref.numpy()
        eq_solref[..., 0] = MIMIC_TIMECONST
        self.solver.mjw_model.eq_solref.assign(eq_solref)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        # アームは初期姿勢を、グリッパは全開を目標にして保持する (q と qd の並びはアームとグリッパでは同じ)
        tgt = self.control.joint_target_q.numpy()
        q = self.model.joint_q.numpy()
        for k in self.arm_q_index:
            tgt[k] = q[k]
        per_world = len(tgt) // self.num_envs
        for w in range(self.num_envs):
            tgt[[w * per_world + self.gripper_dof, w * per_world + self.gripper_r_dof]] = self.robot.stroke
        self.control.joint_target_q.assign(tgt)
        if self.RIGID_CONTACTS_PER_ENV is not None:
            self.model.rigid_contact_max = self.RIGID_CONTACTS_PER_ENV * self.num_envs
        self.contacts = self.model.contacts()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.graph = None
        self.q0 = self.model.joint_q.numpy().copy()
        print(f"bodies={self.model.body_count} joints={self.model.joint_count} dofs={self.model.joint_dof_count}")

        self.viewer.set_model(self.model)
        self.look_at_nails()

    def add_scene(self, builder):
        """ロボット以外の物体を置く。サブクラスで差し替える。"""
        if self.args.box:
            h = self.robot.box_size / 2
            p = self.robot.box_pos
            # 箱の下に静止した台を置く (持ち上げは手順の後)
            builder.add_shape_box(
                -1, xform=wp.transform(wp.vec3(p[0], p[1], p[2] - h - 0.01), wp.quat_identity()), hx=0.1, hy=0.1, hz=0.01
            )
            box = builder.add_body(xform=wp.transform(wp.vec3(*p), wp.quat_identity()), label="box")
            cfg = newton.ModelBuilder.ShapeConfig(density=BOX_MASS / BOX_SIZE**3, mu=1.0)
            builder.add_shape_box(box, hx=h, hy=h, hz=h, cfg=cfg)

    def look_at_nails(self, dist=0.12, elevation_deg=25.0, azimuth_deg=60.0):
        """カメラを爪先 (CAMERA_TARGET) に向ける。azimuth は +y から x 方向へ振る角度。"""
        target = np.array(self.CAMERA_TARGET)
        az, el = np.radians(azimuth_deg), np.radians(elevation_deg)
        offset = dist * np.array([np.sin(az) * np.cos(el), np.cos(az) * np.cos(el), np.sin(el)])
        pos = target + offset
        front = target - pos
        front /= np.linalg.norm(front)
        if hasattr(self.viewer, "set_camera"):  # ViewerGL
            self.viewer.set_camera(
                wp.vec3(*pos), pitch=float(np.degrees(np.arcsin(front[2]))), yaw=float(np.degrees(np.arctan2(front[1], front[0])))
            )
        elif hasattr(self.viewer, "_server"):  # ViewerViser

            def place(client):
                client.camera.position = tuple(pos)
                client.camera.look_at = tuple(target)

            self.viewer._server.on_client_connect(place)
            for client in self.viewer._server.get_clients().values():
                place(client)

    def set_gripper(self, opening_mm):
        """グリッパの片指の開き [mm] (0=閉、Hand-E は 25、COBOTTA は 15 で全開)。右指は URDF の mimic で追従する。"""
        tgt = self.control.joint_target_q.numpy()
        # 右指の目標を放っておくと URDF 取り込み時の値 (リミットの中間 13.4mm) のまま左指と綱引きになり、
        # 全閉にしても片指 6.6mm で止まる。mimic 拘束と矛盾しないよう同じ値を入れる
        tgt[self.gripper_dof] = tgt[self.gripper_r_dof] = np.clip(opening_mm * 1e-3, 0.0, self.robot.stroke)
        self.control.joint_target_q.assign(tgt)

    def gripper_opening_mm(self):
        return float(self.state_0.joint_q.numpy()[self.gripper_q] * 1e3)

    def box_pos(self):
        return self.state_0.body_q.numpy()[self.box_body][:3]

    def nail_state(self):
        """各ネイルの {関節名: 値} を返す。x/y は回転 [deg] (±7)、z は押し込み量 [mm] (0=デフォルト、3.9=最大押し込み。関節値そのまま)。"""
        q = self.state_0.joint_q.numpy()
        out = {}
        for key, idx in self.nail_q_index.items():
            out[key] = float(np.degrees(q[idx])) if key.endswith(("_x", "_y")) else float(q[idx] * 1e3)
        return out

    def set_nail_target(self, finger, x_deg=0.0, y_deg=0.0, push_mm=0.0):
        """finger は "left" か "right"。バネの平衡位置をリミットでクランプして変える (センサ用途では呼ばない)。"""
        vals = {
            "x": np.radians(np.clip(x_deg, -7.0, 7.0)),
            "y": np.radians(np.clip(y_deg, -7.0, 7.0)),
            "z": np.clip(push_mm, 0.0, NAIL_PUSH_MAX_MM) * 1e-3,
        }
        tgt = self.control.joint_target_q.numpy()
        for axis, v in vals.items():
            tgt[self.nail_dof_index[f"{finger}_{axis}"]] = v
        self.control.joint_target_q.assign(tgt)

    def _simulate_frame(self):
        for _ in range(self.substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        # 目標は control の既定値 (全関節 0) のまま。ここで self.control の目標を書き換えて制御する
        # 1 フレーム (substeps 回) を CUDA グラフにして起動する。カーネル起動のオーバーヘッドが支配的なため。
        # substeps が偶数なので state_0 / state_1 は元の向きに戻り、グラフが指す配列は変わらない。
        # 目標 (control) と状態 (state) は assign() で中身だけ書き換えること (配列を差し替えるとグラフに反映されない)
        if self.use_graph and self.graph is None:
            self._simulate_frame()  # ウォームアップ (カーネルのコンパイル・メモリ確保) 兼 1 フレーム目
            with wp.ScopedCapture() as capture:
                self._simulate_frame()
            self.graph = capture.graph
        elif self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self._simulate_frame()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        q = self.state_0.joint_q.numpy()
        if not np.isfinite(q).all():
            raise ValueError("joint_q has non-finite values")
        print("nail state:", self.nail_state())


if __name__ == "__main__":
    import newton.examples

    parser = newton.examples.create_parser()
    parser.add_argument("--urdf", default="tma.urdf")
    parser.add_argument("--box", action=argparse.BooleanOptionalAction, default=True, help="把持対象の箱を置く")
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Sim(viewer, args), args)
