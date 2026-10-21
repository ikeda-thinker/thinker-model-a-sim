"""ピン挿入 (ペグインホール) の RL 環境 (並列版)。THF 爪でつかんだピンを、位置がずれて・揺れる治具の穴に挿す。

- 開始: ピンはつかんだ状態 (起動時に全環境でつかませてライブラリに溜める。つかむ深さ・ずれ・傾きはばらつく)。
  爪先は穴の推定位置の上空、横に数 mm ずらした所から始める
- 外乱:
  - 穴の推定位置 = 真値 + エピソードごとの固定バイアス (カメラの較正誤差相当) + 更新ごとのジッタ。EST_PERIOD ステップごとに更新
  - 治具は x/y のプリズム関節 (PD) に載っていて、"shake" のエピソードでは途中でランダムに目標位置が跳ぶ
- 行動 (3 次元, [-1, 1]): 爪先中点 (TCP) の目標位置の差分 (×ACT_SCALE m)。姿勢は下向き固定で、IK で関節目標にする
- 観測: TCP と推定穴の相対位置、TCP の指令と実位置の差 (アームの PD が押し返されている量 = 接触力の目安)、TCP 速度、
  ネイル 6 値の履歴、平滑化後の行動、経過時間、最初に当たってからの経過時間と沈んだ量 (爪の押し込みで当たりを検出。
  スパイラル探索のような「当たった高さを覚えて探る」動きを、記憶なしの方策でも真似できるように)
- 報酬: 穴との水平距離・穴上面への接近・挿入深さのポテンシャル差分 + 成功ボーナス - ずれ (握り直し) - 押し込みすぎ - 行動の変化量

    uv run tma/envs/peg.py --viewer null --num-envs 64 --episodes 200      # スパイラル探索 (古典手法) の成功率
    uv run tma/envs/peg.py --viewer null --num-envs 64 --policy straight   # 推定位置に真っすぐ下ろすだけ
    uv run tma/envs/peg.py --num-envs 4                                    # GL で見る
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # リポジトリ直下 (tma パッケージ)

import numpy as np
import warp as wp

import newton
import newton.examples
from newton import JointTargetMode
from tma.envs import grasp as ge
from tma import sim

# ピン (アルミ、Ø8 x 40mm)。鋼 (16g) だと pyramidal 錐でも楕円錐でも握ったまま 10mm/s ほどでずり落ちる (MuJoCo の摩擦のクリープ)
PEG_R = 0.004
PEG_HALF = 0.020
PEG_DENSITY = 2700.0
# 握り方。開度とパッド間のすきまの関係は「すきま ≈ 2 x 片指の開度 - 16mm」で、Ø8 には開度 12mm で当たる。
# 一気に閉じる (3 フレームで全閉) とピンを弾き飛ばすので、GRIP_OPEN_MM から 1 フレーム GRIP_RAMP_MM ずつ目標を下げる
GRIP_OPEN_MM = 13.0
GRIP_RAMP_MM = 0.25
GRIP_TARGET_MM = 9.0  # 目標開度。当たる開度との差 3mm x 2000N/m で片指約 6N
GRIP_DEPTH = (0.006, 0.012)  # ピン上端が爪先中点からどれだけ上か [m] (つかむ深さのばらつき)
GRIP_XY = 0.0005  # つかむ位置の水平ずれ (±) [m]
GRIP_TILT_DEG = 2.0  # つかんだピンの傾き (±) [deg]

# 治具 (穴の上面中心が治具 body の原点)。穴は多角形の板を並べて作る (内接円の半径が HOLE_R)
CLEARANCE = 0.0004  # 片側のすきま [m]
HOLE_DEPTH = 0.025
HOLE_SEGS = 16
FIX_HALF = 0.04  # 治具の上面の半寸法 [m]
FIX_XY = ge.OBJ_XY  # 治具の基準位置 (把持環境のワークと同じ所)
FIX_TOP = 0.05  # 穴の上面の高さ [m]。底板が地面の接触距離 (CONTACT_GAP) に入らないように浮かせる
FIX_RANGE = 0.015  # エピソードごとの治具の置き位置のばらつき (±) [m]
FIX_KE, FIX_KD = 2e4, 400.0  # 治具を保持する PD [N/m], [N*s/m]
MU = 0.3  # ピン・治具の摩擦

SHAKE_PROB = 0.5  # 揺れるエピソードの割合
SHAKE_RATE = 0.03  # 揺れるエピソードで、1 ステップあたりに治具の目標が跳ぶ確率
SHAKE_MM = 3.0  # 跳ぶ量 (各軸 ±) [mm]。累積は FIX_RANGE + SHAKE_MM 以内に収める
EST_BIAS_MM = 1.5  # 推定位置の固定バイアス (標準偏差, 各軸) [mm]。片側すきまより大きいので真っすぐ下ろすと当たる
EST_JITTER_MM = 0.3  # 推定の更新ごとのばらつき [mm]
EST_PERIOD = 10  # 推定を更新する間隔 [step] (0.5 s)

CONTACT_MM = 1.0  # 爪の押し込みがこれを超えたら当たったとみなす (リセット直後の揺れで 1mm 近く動くことがある)
ACT_SCALE = 0.002  # 1 ステップあたりの TCP 目標の最大変化 [m] (40 mm/s)
CMD_LEAD = 0.005  # TCP の指令は実位置からこれ以上離さない [m] (押し付けたまま指令だけ沈み続けないように)
START_XY = 0.008  # 開始時の TCP の水平ずれ (推定穴から ±) [m]
START_Z = (0.005, 0.025)  # 開始時のピン下端の高さ (穴の上面から) [m]
EP_STEPS = 200  # 10 s
SUCCESS_DEPTH = 0.015  # 挿入深さ [m]
HOLD_STEPS = 3
SLIP_POS = 0.003  # つかんだピンが爪先に対してこれ以上ずれたら失敗 [m]
SLIP_DEG = 10.0
PUSH_SOFT_MM = 2.0  # 爪の押し込みがこれを超えた分を罰する (約 4N)
IK_ITERS = 8


def quat_mul(a, b):
    """xyzw の四元数の積 (バッチ)"""
    x1, y1, z1, w1 = np.moveaxis(a, -1, 0)
    x2, y2, z2, w2 = np.moveaxis(b, -1, 0)
    return np.stack(
        [w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
         w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2], axis=-1
    )


def quat_conj(q):
    return q * np.array([-1.0, -1.0, -1.0, 1.0])


def quat_rot(q, v):
    """q (xyzw) で v を回す (バッチ)"""
    u, w = q[..., :3], q[..., 3:]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def axis_angle_quat(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis, axis=-1, keepdims=True)
    return np.concatenate([axis * np.sin(angle / 2)[..., None], np.cos(angle / 2)[..., None]], axis=-1)


def potential(dxy, height, depth):
    """報酬整形のポテンシャル。dxy はピン下端と穴の軸の水平距離、height は下端の穴上面からの高さ、depth は挿入深さ"""
    return (
        0.5 * (1.0 - np.tanh(dxy / 0.003))
        + 0.3 * (1.0 - np.tanh(np.maximum(height, 0.0) / 0.01))
        + 3.0 * np.clip(depth / SUCCESS_DEPTH, 0.0, 1.0)
    )


class PegEnv(sim.Sim):
    NJMAX, NCONMAX = 400, 100
    RIGID_CONTACTS_PER_ENV = 200
    SOLVER_KW = {**sim.Sim.SOLVER_KW, "cone": "elliptic"}  # pyramidal 錐だとアルミのピンでも握ったままずり落ちる
    ARM_Q_DEG = ge.ARM_Q_DEG
    CAMERA_TARGET = (*FIX_XY, FIX_TOP)
    act_dim = 3
    LIBRARY_ROUNDS = 2
    # 学習スクリプトが集計に使う結果の種類 (info のキー, 表示名)
    OUTCOMES = (("success", "成功"), ("slipped", "ずれ"), ("timeout", "時間切れ"))

    def add_scene(self, builder):
        # アームのベースのメッシュは地面にめり込んでいて、常に 1 環境 20 点ほどの接触を作るので当たり判定を外す
        for s, b in enumerate(builder.shape_body):
            if b >= 0 and builder.body_label[b].endswith("base_link_inertia"):
                builder.shape_flags[s] &= ~int(newton.ShapeFlags.COLLIDE_SHAPES)
        hole_r = PEG_R + CLEARANCE
        # 治具: world -> (x プリズム) -> スライダ -> (y プリズム) -> 治具
        base = wp.transform(wp.vec3(*FIX_XY, FIX_TOP), wp.quat_identity())
        inertia = wp.mat33(np.eye(3) * 1e-3)
        slider = builder.add_link(xform=base, mass=0.5, inertia=inertia, label="fixture_slider")
        fixture = builder.add_link(xform=base, mass=1.0, inertia=inertia, label="fixture", lock_inertia=True)
        drive = dict(target_ke=FIX_KE, target_kd=FIX_KD, limit_lower=-0.05, limit_upper=0.05, actuator_mode=JointTargetMode.POSITION)
        jx = builder.add_joint_prismatic(-1, slider, parent_xform=base, axis=newton.Axis.X, label="fixture_x", **drive)
        jy = builder.add_joint_prismatic(slider, fixture, axis=newton.Axis.Y, label="fixture_y", **drive)
        builder.add_articulation([jx, jy], label="fixture")
        cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=MU)
        hz = HOLE_DEPTH / 2
        # 穴の壁: 内接円の半径 hole_r の正多角形の各辺に、半径方向の半厚 4mm の板
        t = 0.004
        half_side = hole_r * np.tan(np.pi / HOLE_SEGS) + 0.0005
        for i in range(HOLE_SEGS):
            th = 2 * np.pi * i / HOLE_SEGS
            xf = wp.transform(wp.vec3((hole_r + t) * np.cos(th), (hole_r + t) * np.sin(th), -hz), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(th)))
            builder.add_shape_box(fixture, xform=xf, hx=t, hy=half_side, hz=hz, cfg=cfg)
        # 外枠 (上面): 内側の半寸法 a は壁の板の外側に重なるように
        a, b = hole_r + 0.004, FIX_HALF
        for sx, sy, hx, hy in ((0, 1, b, (b - a) / 2), (0, -1, b, (b - a) / 2), (1, 0, (b - a) / 2, a), (-1, 0, (b - a) / 2, a)):
            xf = wp.transform(wp.vec3(sx * (a + b) / 2, sy * (a + b) / 2, -hz), wp.quat_identity())
            builder.add_shape_box(fixture, xform=xf, hx=hx, hy=hy, hz=hz, cfg=cfg)
        builder.add_shape_box(fixture, xform=wp.transform(wp.vec3(0.0, 0.0, -HOLE_DEPTH - 0.002), wp.quat_identity()), hx=b, hy=b, hz=0.002, cfg=cfg)
        # ピン (円柱の軸は z)
        peg = builder.add_body(xform=wp.transform(wp.vec3(FIX_XY[0], FIX_XY[1], 0.3), wp.quat_identity()), label="peg")
        builder.add_shape_cylinder(peg, radius=PEG_R, half_height=PEG_HALF, cfg=newton.ModelBuilder.ShapeConfig(density=PEG_DENSITY, mu=MU))

    def __init__(self, viewer, args, seed=0, scripted=False):
        super().__init__(viewer, args)
        self.rng = np.random.default_rng(seed)
        self.scripted = scripted
        N, m = self.num_envs, self.model
        self.C, self.D, self.B = m.joint_coord_count // N, m.joint_dof_count // N, m.body_count // N
        self.T = self.control.joint_target_q.shape[0] // N
        self.arm_idx = np.array(self.arm_q_index)
        labels, q_start = list(m.joint_label), m.joint_q_start.numpy()

        def joint_q(suffix):
            return int(q_start[next(i for i, l in enumerate(labels) if l.endswith(suffix))])

        self.gripper_rq = joint_q("robotiq_hande_right_finger_joint")
        self.fix_idx = np.array([joint_q("fixture_x"), joint_q("fixture_y")])  # q と target の並びは同じ (自由関節より前)
        self.peg_slot = joint_q("peg_free_joint") + np.arange(7)
        self.peg_dof = int(m.joint_qd_start.numpy()[next(i for i, l in enumerate(labels) if l.endswith("peg_free_joint"))]) + np.arange(6)
        body = list(m.body_label)

        def body_of(suffix):
            return next(i for i, l in enumerate(body) if l.endswith(suffix))

        self.tips = np.array([body_of(f"{f}_thf_tip_link") for f in ("left", "right")])
        self.tool_body, self.peg_body, self.fix_body = body_of("thinker_THF1_tool_link"), body_of("peg"), body_of("fixture")
        self.q_init = m.joint_q.numpy().reshape(N, self.C)[0].copy()
        self.tgt_init = self.control.joint_target_q.numpy().reshape(N, self.T)[0].copy()
        self.ik = ge.ArmIK(args.urdf, np.radians(ge.ARM_Q_DEG), n=N)

        self.t = np.zeros(N, dtype=int)
        self.hold = np.zeros(N, dtype=int)
        self.shake = np.zeros(N, dtype=bool)
        self.fix_base = np.zeros((N, 2))  # 治具のエピソードの基準位置 (関節値)
        self.fix_tgt = np.zeros((N, 2))
        self.est_bias = np.zeros((N, 3))
        self.est = np.zeros((N, 3))  # 穴の上面中心の推定位置
        self.tcp_cmd = np.zeros((N, 3))
        self.q_cmd = np.zeros((N, 6))
        self.rel_pos = np.zeros((N, 3))  # つかんだピンの、ツールリンク座標での位置・姿勢 (ライブラリの値)
        self.rel_quat = np.zeros((N, 4))
        self.protrude = np.zeros(N)  # 爪先中点からピン下端までの高さ
        self.act_filt = np.zeros((N, self.act_dim))
        self.act_prev = np.zeros((N, self.act_dim))
        self.nail_hist = np.zeros((N, ge.NAIL_HIST, 6))
        self.tip_prev = np.zeros((N, 3))
        self.phi_prev = np.zeros(N)
        self.t_contact = np.full(N, -1)  # 最初に当たったステップ (-1 はまだ)
        self.z_contact = np.zeros(N)  # そのときの爪先の高さ
        # スクリプト方策 (スパイラル探索) の状態
        self.phase = np.zeros(N, dtype=int)
        self.theta = np.zeros(N)
        self.spiral = True
        self.disp = None
        self.watch = 0

        t0 = time.time()
        self.build_library()
        print(f"把持のライブラリ: {len(self.lib_q)} 個 ({time.time() - t0:.1f} s)")
        self.reset_envs(np.arange(N))
        self.obs = self.observe()

    @property
    def obs_dim(self):
        return self.obs.shape[1]

    # ---- 状態の読み書き ----
    def read(self):
        N = self.num_envs
        q = self.state_0.joint_q.numpy().reshape(N, self.C)
        qd = self.state_0.joint_qd.numpy().reshape(N, self.D)
        bq = self.state_0.body_q.numpy().reshape(N, self.B, 7)
        return q, qd, bq

    def write(self, q, qd):
        for s in (self.state_0, self.state_1):
            s.joint_q.assign(q.reshape(-1))
            s.joint_qd.assign(qd.reshape(-1))
            newton.eval_fk(self.model, s.joint_q, s.joint_qd, s)

    def tip_mid(self, bq):
        return (bq[:, self.tips[0], :3] + bq[:, self.tips[1], :3]) / 2

    def peg_rel(self, bq):
        """ピンの姿勢を、位置は爪先中点から・向きはツールリンクの座標で表したもの。
        爪の押し込み・傾きでピンはツールに対して数 mm 動くので、位置は爪と一緒に動く爪先から測る"""
        tq = bq[:, self.tool_body, 3:]
        pos = quat_rot(quat_conj(tq), bq[:, self.peg_body, :3] - self.tip_mid(bq))
        return pos, quat_mul(quat_conj(tq), bq[:, self.peg_body, 3:])

    def peg_bottom(self, bq):
        return bq[:, self.peg_body, :3] - quat_rot(bq[:, self.peg_body, 3:], np.array([0.0, 0.0, PEG_HALF]))

    def nail_vec(self, q):
        out = np.empty((self.num_envs, 6))
        for j, key in enumerate(ge.NAIL_KEYS):
            v = q[:, self.nail_q_index[key]]
            out[:, j] = v * 1e3 if key.endswith("z") else np.degrees(v)
        return out

    def solve_ik(self, tip_pos, q_init=None, iters=IK_ITERS):
        return self.ik.solve_batch(tip_pos, np.zeros(self.num_envs), q_init, iters=iters)

    # ---- 把持のライブラリ ----
    def build_library(self):
        """全環境で、上空で開いた爪の間にピンを置いて閉じ、つかんだ状態 (指・ネイルの関節値と、ツールに対するピンの姿勢) を溜める"""
        N, rng = self.num_envs, self.rng
        q_lib_arm = self.solve_ik(np.tile([*FIX_XY, 0.15], (N, 1)), iters=48)
        lib_q, lib_rel = [], []
        opening = GRIP_OPEN_MM * 1e-3
        for _ in range(self.LIBRARY_ROUNDS):
            q = np.tile(self.q_init, (N, 1))
            q[:, self.arm_idx] = q_lib_arm
            q[:, [self.gripper_q, self.gripper_rq]] = opening
            self.write(q, np.zeros((N, self.D)))
            tip = self.tip_mid(self.read()[2])
            depth = rng.uniform(*GRIP_DEPTH, N)
            pos = tip + np.column_stack([rng.uniform(-GRIP_XY, GRIP_XY, (N, 2)), depth - PEG_HALF])
            tilt_axis = np.column_stack([rng.normal(size=(N, 2)), np.zeros(N)])
            quat = axis_angle_quat(tilt_axis, np.radians(rng.uniform(-GRIP_TILT_DEG, GRIP_TILT_DEG, N)))
            q[:, self.peg_slot[:3]], q[:, self.peg_slot[3:]] = pos, quat
            tgt = np.tile(self.tgt_init, (N, 1))
            tgt[:, self.arm_idx] = q_lib_arm
            self.write(q, np.zeros((N, self.D)))
            ramp = int(np.ceil((GRIP_OPEN_MM - GRIP_TARGET_MM) / GRIP_RAMP_MM))
            for f in range(ramp + 40):
                tgt[:, [self.gripper_dof, self.gripper_r_dof]] = max(GRIP_TARGET_MM, GRIP_OPEN_MM - GRIP_RAMP_MM * f) * 1e-3
                self.control.joint_target_q.assign(tgt.reshape(-1))
                if f < ramp:  # 指が当たるまではピンをその場に留める (重力で落ちないように)
                    cur, qd, _ = self.read()
                    cur[:, self.peg_slot], qd[:, self.peg_dof] = q[:, self.peg_slot], 0.0
                    self.write(cur, qd)
                sim.Sim.step(self)
            cur, _, bq = self.read()
            rel_p, rel_q = self.peg_rel(bq)
            bottom = self.peg_bottom(bq)
            held = np.isfinite(cur).all(axis=1) & (np.abs(bq[:, self.peg_body, 2] - pos[:, 2]) < 0.003)
            for w in np.nonzero(held)[0]:
                lib_q.append(cur[w].copy())
                lib_rel.append(np.concatenate([rel_p[w], rel_q[w], [tip[w, 2] - bottom[w, 2]]]))
        if not lib_q:
            raise RuntimeError("ピンを 1 本もつかめなかった")
        self.lib_q, self.lib_rel = np.stack(lib_q), np.stack(lib_rel)
        self.tgt_grip = tgt[0].copy()  # 閉じた指の目標 (アーム以外)

    # ---- リセット ----
    def reset_envs(self, idx):
        idx = np.asarray(idx)
        n, N, rng = len(idx), self.num_envs, self.rng
        if n == 0:
            return
        pick = rng.integers(len(self.lib_q), size=n)
        self.rel_pos[idx], self.rel_quat[idx], self.protrude[idx] = self.lib_rel[pick, :3], self.lib_rel[pick, 3:7], self.lib_rel[pick, 7]
        self.fix_base[idx] = rng.uniform(-FIX_RANGE, FIX_RANGE, (n, 2))
        self.fix_tgt[idx] = self.fix_base[idx]
        self.shake[idx] = rng.random(n) < SHAKE_PROB
        self.est_bias[idx] = np.column_stack([rng.normal(0.0, EST_BIAS_MM * 1e-3, (n, 2)), rng.normal(0.0, 0.5e-3, n)])
        hole = np.column_stack([np.array(FIX_XY) + self.fix_base[idx], np.full(n, FIX_TOP)])
        self.est[idx] = hole + self.est_bias[idx] + np.column_stack([rng.normal(0.0, EST_JITTER_MM * 1e-3, (n, 2)), np.zeros(n)])
        start = self.est[idx] + np.column_stack(
            [rng.uniform(-START_XY, START_XY, (n, 2)), self.protrude[idx] + rng.uniform(*START_Z, n)]
        )
        # IK は全環境ぶんの問題を一度に解くので、idx 以外は idx[0] の問題で埋める
        pos = np.tile(start[0], (N, 1))
        pos[:n] = start
        q_start = self.solve_ik(pos, iters=48)[:n]

        q, qd, _ = self.read()
        for i, w in enumerate(idx):
            q[w] = self.lib_q[pick[i]]
            q[w, self.arm_idx] = q_start[i]
            q[w, self.fix_idx] = self.fix_base[w]
            qd[w] = 0.0
        self.write(q, qd)
        # ツールの姿勢が決まったので、つかんだピンをそれに合わせて置き直す
        bq = self.read()[2]
        tq = bq[idx, self.tool_body, 3:]
        q[idx[:, None], self.peg_slot[None, :3]] = self.tip_mid(bq)[idx] + quat_rot(tq, self.rel_pos[idx])
        q[idx[:, None], self.peg_slot[None, 3:]] = quat_mul(tq, self.rel_quat[idx])
        self.write(q, qd)
        tgt = self.control.joint_target_q.numpy().reshape(N, self.T)
        tgt[idx] = self.tgt_grip
        tgt[idx[:, None], self.arm_idx[None]] = q_start
        tgt[idx[:, None], self.fix_idx[None]] = self.fix_base[idx]
        self.control.joint_target_q.assign(tgt.reshape(-1))

        bq = self.read()[2]
        self.q_cmd[idx] = q_start
        self.tcp_cmd[idx] = self.tip_mid(bq)[idx]
        self.tip_prev[idx] = self.tcp_cmd[idx]
        self.act_filt[idx] = 0.0
        self.act_prev[idx] = 0.0
        self.nail_hist[idx] = 0.0
        self.t[idx], self.hold[idx] = 0, 0
        self.phase[idx], self.theta[idx] = 0, 0.0
        self.t_contact[idx], self.z_contact[idx] = -1, 0.0
        dxy, height, depth = self.measure(bq)
        self.phi_prev[idx] = potential(dxy, height, depth)[idx]

    # ---- 計測 ----
    def hole(self, bq):
        return bq[:, self.fix_body, :3]

    def measure(self, bq):
        """ピン下端と穴の軸の水平距離、下端の穴上面からの高さ、挿入深さ"""
        bottom, hole = self.peg_bottom(bq), self.hole(bq)
        dxy = np.hypot(*(bottom[:, :2] - hole[:, :2]).T)
        height = bottom[:, 2] - hole[:, 2]
        depth = np.where(dxy < PEG_R + CLEARANCE + 0.001, np.maximum(-height, 0.0), 0.0)
        return dxy, height, depth

    def observe(self):
        q, qd, bq = self.read()
        tip = self.tip_mid(bq)
        rel = self.est - tip
        rel[:, 2] += self.protrude  # z は「ピン下端の目安」と穴上面の差にする
        vel = (tip - self.tip_prev) / (ge.CTRL_FRAMES / self.fps)
        scale = np.tile([7.0, 7.0, sim.NAIL_PUSH_MAX_MM] * 2, ge.NAIL_HIST)
        hit = self.t_contact >= 0
        return np.concatenate(
            [
                rel / 0.02,
                (self.tcp_cmd - tip) / CMD_LEAD,
                vel / 0.05,
                self.nail_hist.reshape(self.num_envs, -1) / scale,
                self.act_filt,
                (self.t / EP_STEPS)[:, None],
                hit[:, None].astype(float),
                np.where(hit, (self.t - self.t_contact) / EP_STEPS, 0.0)[:, None],
                np.where(hit, (self.z_contact - tip[:, 2]) / 0.005, 0.0)[:, None],
            ],
            axis=1,
        ).astype(np.float32)

    # ---- ステップ ----
    def step_env(self, action):
        N, rng = self.num_envs, self.rng
        a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        self.act_filt = ge.ACT_EMA * a + (1.0 - ge.ACT_EMA) * self.act_filt
        _, _, bq = self.read()
        tip = self.tip_mid(bq)
        self.tip_prev = tip
        self.tcp_cmd = np.clip(self.tcp_cmd + self.act_filt * ACT_SCALE, tip - CMD_LEAD, tip + CMD_LEAD)
        self.q_cmd = self.solve_ik(self.tcp_cmd, self.q_cmd)
        tgt = self.control.joint_target_q.numpy().reshape(N, self.T)
        tgt[:, self.arm_idx] = self.q_cmd
        # 治具の揺れ: 揺れるエピソードでは確率 SHAKE_RATE で目標位置が跳ぶ
        jump = self.shake & (rng.random(N) < SHAKE_RATE)
        lim = FIX_RANGE + SHAKE_MM * 1e-3
        self.fix_tgt = np.where(
            jump[:, None], np.clip(self.fix_tgt + rng.uniform(-SHAKE_MM, SHAKE_MM, (N, 2)) * 1e-3, -lim, lim), self.fix_tgt
        )
        tgt[:, self.fix_idx] = self.fix_tgt
        self.control.joint_target_q.assign(tgt.reshape(-1))
        for _ in range(ge.CTRL_FRAMES):
            sim.Sim.step(self)
        self.t += 1

        q, qd, bq = self.read()
        self.nail_hist = np.concatenate([self.nail_hist[:, 1:], self.nail_vec(q)[:, None]], axis=1)
        # 推定位置の更新 (固定バイアス + ジッタ)
        upd = self.t % EST_PERIOD == 0
        if upd.any():
            fresh = self.hole(bq) + self.est_bias + np.column_stack([rng.normal(0.0, EST_JITTER_MM * 1e-3, (N, 2)), np.zeros(N)])
            self.est = np.where(upd[:, None], fresh, self.est)
        push = self.nail_hist[:, -1, [2, 5]]
        first = (self.t_contact < 0) & (push.max(axis=1) > CONTACT_MM)
        self.t_contact = np.where(first, self.t, self.t_contact)
        self.z_contact = np.where(first, self.tip_mid(bq)[:, 2], self.z_contact)
        dxy, height, depth = self.measure(bq)
        rel_p, rel_q = self.peg_rel(bq)
        slip_pos = np.linalg.norm(rel_p - self.rel_pos, axis=1)
        slip_ang = np.degrees(2 * np.arccos(np.clip(np.abs((rel_q * self.rel_quat).sum(axis=1)), 0.0, 1.0)))
        slipped = (slip_pos > SLIP_POS) | (slip_ang > SLIP_DEG) | ~np.isfinite(bq).all(axis=(1, 2))
        self.hold = np.where(depth >= SUCCESS_DEPTH, self.hold + 1, 0)
        success = (self.hold >= HOLD_STEPS) & ~slipped
        over = (np.maximum(push - PUSH_SOFT_MM, 0.0) / (sim.NAIL_PUSH_MAX_MM - PUSH_SOFT_MM)) ** 2
        phi = potential(dxy, height, depth)
        reward = (
            np.where(slipped, -self.phi_prev, phi - self.phi_prev)  # ずれたら終端のポテンシャルを 0 とする (深さを稼いでからずれる抜け道を塞ぐ)
            + 20.0 * success
            - 5.0 * slipped
            - 0.1 * over.sum(axis=1)
            - 0.002 * ((a - self.act_prev) ** 2).sum(axis=1)
        )
        reward = np.nan_to_num(reward)
        self.act_prev = a
        self.phi_prev = phi
        done = success | slipped | (self.t >= EP_STEPS)
        self.disp = dict(
            t=self.t.copy(), phase=self.phase.copy(), nail=self.nail_hist[:, -1].copy(), dxy=dxy, height=height, depth=depth,
            shake=self.shake.copy(),
        )
        info = dict(
            success=success & done, slipped=slipped & done, timeout=done & ~(success | slipped), depth=depth, dxy=dxy,
            slip_mm=slip_pos * 1e3, slip_deg=slip_ang, t=self.t.copy(), push=push.max(axis=1),
            type=["shake" if s else "still" for s in self.shake],
        )
        if info["timeout"].any():
            info["final_obs"] = self.observe()
        self.reset_envs(np.nonzero(done)[0])
        self.obs = self.observe()
        return self.obs, reward.astype(np.float32), done, info

    # ---- 古典手法: スパイラル探索 ----
    MAX_PUSH_MM = 2.0  # 押し込みがこれを超えたら下降を止める
    SPIRAL_PITCH = 0.0008  # スパイラルの 1 周あたりの半径の増分 [m] (すきまの 2 倍以下)
    SPIRAL_STEP = 0.0008  # 1 ステップに進む弧長 [m]
    PRESS = 0.0015  # 探索中に当たった高さから押し付ける量 [m]
    FALL = 0.001  # 当たった高さからこれだけ下がったら穴に入ったとみなす [m]。入りかけで横に引きずると、ピンが爪の間で回ってしまう

    def scripted_action(self):
        """0: 推定位置へ下ろす → 1: 当たったらその高さで軽く押し付けながら推定位置のまわりをスパイラル → 2: 落ちたら真下へ"""
        N = self.num_envs
        _, _, bq = self.read()
        tip = self.tip_mid(bq)
        push = self.nail_hist[:, -1, [2, 5]].max(axis=1)
        hit = (self.phase == 0) & (self.t_contact >= 0)
        self.phase = np.where(hit, 1 if self.spiral else 2, self.phase)
        fell = (self.phase == 1) & (tip[:, 2] < self.z_contact - self.FALL)
        self.phase = np.where(fell, 2, self.phase)

        goal = self.tcp_cmd.copy()
        goal[:, :2] = self.est[:, :2]
        goal[:, 2] = np.where(push > self.MAX_PUSH_MM, tip[:, 2], tip[:, 2] - 0.01)  # 下降・挿入は一定速度で、押し込みすぎたら止める
        s = self.phase == 1
        r = self.SPIRAL_PITCH * self.theta / (2 * np.pi)
        self.theta = np.where(s, self.theta + self.SPIRAL_STEP / np.maximum(r, self.SPIRAL_PITCH), self.theta)
        r = self.SPIRAL_PITCH * self.theta / (2 * np.pi)
        spiral = self.est[:, :2] + r[:, None] * np.column_stack([np.cos(self.theta), np.sin(self.theta)])
        goal[s, :2] = spiral[s]
        goal[s, 2] = self.z_contact[s] - self.PRESS
        ins = self.phase == 2
        goal[ins, :2] = self.tcp_cmd[ins, :2]
        da = np.clip((goal - self.tcp_cmd) / ACT_SCALE, -1.0, 1.0)
        e = ge.ACT_EMA
        return np.clip((da - (1.0 - e) * self.act_filt) / e, -1.0, 1.0)

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        if self.disp is not None:
            w = self.watch
            self.viewer.log_scalar("peg/dxy_mm", float(self.disp["dxy"][w] * 1e3))
            self.viewer.log_scalar("peg/depth_mm", float(self.disp["depth"][w] * 1e3))
            for j, key in enumerate(ge.NAIL_KEYS):
                self.viewer.log_scalar(f"nail/{key}", float(self.disp["nail"][w, j]))
        self.viewer.end_frame()


def add_env_args(parser):
    parser.add_argument("--clearance-mm", type=float, default=CLEARANCE * 1e3, help="穴の片側のすきま [mm]")
    parser.add_argument("--est-bias-mm", type=float, default=EST_BIAS_MM, help="穴の推定位置の固定バイアス (標準偏差) [mm]")
    parser.add_argument("--shake-mm", type=float, default=SHAKE_MM, help="治具が跳ぶ量 (各軸 ±) [mm]")
    parser.add_argument("--shake-prob", type=float, default=SHAKE_PROB, help="治具が揺れるエピソードの割合")


def apply_env_args(args):
    global CLEARANCE, EST_BIAS_MM, SHAKE_MM, SHAKE_PROB
    CLEARANCE = args.clearance_mm * 1e-3
    EST_BIAS_MM, SHAKE_MM, SHAKE_PROB = args.est_bias_mm, args.shake_mm, args.shake_prob


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--urdf", default="tma.urdf")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy", choices=("spiral", "straight"), default="spiral", help="straight は当たっても探さずに押し込む")
    parser.add_argument("--watch", type=int, default=0)
    parser.add_argument("--print-every", type=int, default=0)
    add_env_args(parser)
    parser.set_defaults(num_frames=10**9)
    viewer, args = newton.examples.init(parser)
    apply_env_args(args)
    env = PegEnv(viewer, args, seed=args.seed, scripted=True)
    env.spiral = args.policy == "spiral"
    env.watch = args.watch
    if hasattr(viewer, "hide_loading_splash"):
        viewer.hide_loading_splash()
    print(f"num_envs={env.num_envs} obs_dim={env.obs_dim} act_dim={env.act_dim} 突き出し {env.lib_rel[:, 7].min() * 1e3:.1f}〜{env.lib_rel[:, 7].max() * 1e3:.1f}mm")
    stats, finished, steps = {}, 0, 0
    t0 = time.time()
    while finished < args.episodes and viewer.is_running():
        obs, rew, done, info = env.step_env(env.scripted_action())
        env.render()
        steps += env.num_envs
        w = env.watch
        if args.print_every and (done[w] or env.disp["t"][w] % args.print_every == 0):
            d = env.disp
            print(
                f"env{w} t={d['t'][w]:3d} phase={d['phase'][w]} dxy={d['dxy'][w] * 1e3:4.1f}mm h={d['height'][w] * 1e3:+5.1f}mm "
                f"depth={d['depth'][w] * 1e3:4.1f}mm push=({d['nail'][w, 2]:.2f},{d['nail'][w, 5]:.2f})mm r={rew[w]:+.2f}"
            )
        for w in np.nonzero(done)[0]:
            s = stats.setdefault(info["type"][w], [0] * (1 + len(env.OUTCOMES)))
            s[0] += 1
            for j, (k, _) in enumerate(env.OUTCOMES):
                s[1 + j] += bool(info[k][w])
            finished += 1
    dt = time.time() - t0
    print(f"{steps} env-steps, {dt:.1f} s -> {steps / dt:.0f} env-steps/s, {finished} episodes")
    tot = np.sum(list(stats.values()), axis=0)
    print(f"  {'':8s} 試行 " + " ".join(name for _, name in env.OUTCOMES))
    for name in sorted(stats):
        print(f"  {name:8s} " + " ".join(f"{v:4d}" for v in stats[name]))
    print(f"  {'合計':8s} " + " ".join(f"{v:4d}" for v in tot) + f"   成功率 {tot[1] / tot[0]:.2f}")
    viewer.close()
