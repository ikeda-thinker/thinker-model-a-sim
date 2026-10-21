"""把持の RL 環境 (1 環境版)。ワークの姿勢推定 (SAM3 / FoundationPose 相当) はノイズ付きで与える前提。

- reset: ワークをランダムに置き、ノイズ付き推定姿勢の上空 PRE_HEIGHT へ IK でアームを置く (q_pre)。IK はここだけ
- 行動 (6 次元, [-1, 1]): shoulder_pan / shoulder_lift / elbow / wrist_1 / wrist_3 の目標角の差分 (×ACT_SCALE rad) と
  Hande の開度 (絶対値、-1=閉 / +1=GRIP_OPEN_MAX_MM)。wrist_2 は q_pre のまま固定 (下向き把持では IK でもほぼ -90 度のまま)
- 観測: アーム q, dq、開度、ネイル 6 値の履歴、q - q_pre、ワークの推定位置 (爪先基準) と推定 yaw のずれ
- 報酬: 接近 + ネイルの反応 + 持ち上げ高さ + 成功ボーナス - 行動の変化量

    uv run tma/envs/grasp.py --viewer null              # スクリプト方策で 1 エピソード (持ち上げ判定の確認)
    uv run tma/envs/grasp.py --viewer null --episodes 5 # 位置・yaw を変えて繰り返す
    uv run tma/envs/grasp.py --viewer viser             # ブラウザで見る (http://localhost:8080)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # リポジトリ直下 (tma パッケージ)

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.ik as ik
from tma import sim

ARM_Q_DEG = (0.0, -90.0, 90.0, -90.0, -90.0, 0.0)  # 下向きの基準姿勢 (check_press と同じ)
# 行動で動かすアーム関節 (shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_3)。
# wrist_1 を固定すると上下動で手先が傾いて横にずれるので外せない
ACT_JOINTS = (0, 1, 2, 3, 5)
ACT_SCALE = 0.03  # 1 制御ステップあたりの関節目標の最大変化 [rad]
ACT_EMA = 0.5  # 行動の平滑化 (1 で平滑化なし)
GRIP_OPEN_MAX_MM = 25.0  # 開度行動 +1 に対応する片指の開き [mm]。バラ積み爪は先端が内側に寄るので全ストロークまで使う
CTRL_FRAMES = 3  # 1 制御ステップの物理フレーム数 (60Hz / 3 = 20Hz)
EP_STEPS = 160  # 8 s

# ワーク (今は立方体 1 個。後でばら積みの山に置き換える)
OBJ_SIZE = 0.02  # 一辺 [m]
OBJ_MASS = 0.02  # [kg]
OBJ_XY = (0.2986, 0.135)  # 基準姿勢の爪先の真下
OBJ_XY_RANGE = 0.02  # 置き位置のばらつき (±) [m]
# 推定姿勢のノイズ (FoundationPose 相当より大きめに)
EST_POS_STD = 0.003  # [m]
EST_YAW_STD = np.radians(5.0)

PRE_HEIGHT = 0.04  # q_pre での爪先の高さ (ワーク上面から) [m]
MISS_RISE = 0.02  # 爪先が最低点からこれだけ上がっても物体が浮いていなければ失敗 [m]
LIFT_SUCCESS = 0.03  # 持ち上げ成功とみなす高さ [m]
HOLD_STEPS = 20  # 成功に必要な保持ステップ数 (1 s)
NAIL_ACTIVE_MM = 0.2  # 押し込みがこれを超えたら接触とみなす [mm]
NAIL_ACTIVE_DEG = 0.5  # 回転がこれを超えたら接触とみなす [deg]
NAIL_HIST = 3  # 観測に入れるネイル値の履歴長

NAIL_KEYS = ("left_x", "left_y", "left_z", "right_x", "right_y", "right_z")


def potential(d, n_touch, lift):
    """報酬整形のポテンシャル Φ。報酬には差分 Φ(s') - Φ(s) だけを入れるので、
    近くで粘っても報酬は貯まらず、エピソード全体の整形分は Φ の終値 - 初期値で頭打ちになる"""
    return 0.2 * (1.0 - np.tanh(d / 0.02)) + 0.2 * n_touch + 2.0 * np.clip(lift / LIFT_SUCCESS, 0.0, 1.0)


def yaw_quat(yaw):
    return wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(yaw))


def wrap_period(a, period):
    """対称性の周期で角度差を ±period/2 に畳む。"""
    return (a + period / 2) % period - period / 2


def wrap_quarter(a):
    """立方体は 90 度対称なので yaw 差を ±45 度に畳む。"""
    return wrap_period(a, np.pi / 2)


class ArmIK:
    """ロボット単体のモデルで、爪先中点 (TCP) の位置と yaw から関節角を解く。"""

    def __init__(self, urdf, q_seed, n=1):
        """n は同時に解く問題数 (solve_batch 用)。"""
        self.n = n
        builder = sim.build_robot(urdf)
        self.model = builder.finalize()
        labels = list(self.model.body_label)
        tool = next(i for i, l in enumerate(labels) if l.endswith("thinker_THF1_tool_link"))
        tips = [next(i for i, l in enumerate(labels) if l.endswith(f"{f}_thf_tip_link")) for f in ("left", "right")]
        q = self.model.joint_q.numpy()
        q[: len(q_seed)] = q_seed
        state = self.model.state()
        newton.eval_fk(self.model, wp.array(q, dtype=float), self.model.joint_qd, state)
        bq = state.body_q.numpy()
        tool_tf = wp.transform(*bq[tool])
        tip_mid = (bq[tips[0]][:3] + bq[tips[1]][:3]) / 2
        # 爪先中点をツールリンク座標で表したもの (アーム姿勢によらず一定)
        offset = wp.transform_point(wp.transform_inverse(tool_tf), wp.vec3(*tip_mid))
        self.down = wp.transform_get_rotation(tool_tf)  # 基準姿勢 (yaw=0) の手先の向き
        self.q_seed = q

        self.pos_obj = ik.IKObjectivePosition(
            link_index=tool, link_offset=offset, target_positions=wp.array([wp.vec3(*tip_mid)] * n, dtype=wp.vec3)
        )
        self.rot_obj = ik.IKObjectiveRotation(
            link_index=tool,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array([wp.vec4(*self.down)] * n, dtype=wp.vec4),
        )
        limits = ik.IKObjectiveJointLimit(
            joint_limit_lower=self.model.joint_limit_lower, joint_limit_upper=self.model.joint_limit_upper, weight=10.0
        )
        self.solver = ik.IKSolver(
            model=self.model,
            n_problems=n,
            objectives=[self.pos_obj, self.rot_obj, limits],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )

    def solve_batch(self, tip_pos, yaw, q_init=None, iters=48):
        """solve の n 問題版。tip_pos は [n, 3]、yaw は [n]、q_init は [n, 6]。戻り値は [n, 6]。"""
        tip_pos = np.asarray(tip_pos, dtype=np.float32).reshape(self.n, 3)
        h = np.asarray(yaw, dtype=np.float64).reshape(self.n) / 2
        # yaw 回転 (z 軸まわり) と基準姿勢の積 (xyzw)
        d = np.array(self.down, dtype=np.float64)
        qy = np.stack([np.zeros(self.n), np.zeros(self.n), np.sin(h), np.cos(h)], axis=1)
        x1, y1, z1, w1 = qy.T
        x2, y2, z2, w2 = d
        rot = np.stack(
            [w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
             w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2], axis=1
        ).astype(np.float32)
        self.pos_obj.target_positions.assign(tip_pos)
        self.rot_obj.target_rotations.assign(rot)
        q0 = np.tile(self.q_seed, (self.n, 1))
        if q_init is not None:
            q0[:, :6] = q_init
        q = wp.array(q0, dtype=float)
        self.solver.step(q, q, iterations=iters)
        return q.numpy()[:, :6].copy()

    def solve(self, tip_pos, yaw, q_init=None, iters=48):
        """爪先中点を tip_pos、手先の yaw を基準姿勢から yaw [rad] 回した姿勢にする関節角 (アーム 6 軸) を返す。"""
        self.pos_obj.set_target_position(0, wp.vec3(*tip_pos))
        self.rot_obj.set_target_rotation(0, wp.vec4(*(yaw_quat(yaw) * self.down)))
        q0 = self.q_seed.copy()
        if q_init is not None:
            q0[:6] = q_init
        q = wp.array(q0.reshape(1, -1), dtype=float)
        self.solver.step(q, q, iterations=iters)
        return q.numpy()[0, :6].copy()


class GraspEnv(sim.Sim):
    ARM_Q_DEG = ARM_Q_DEG
    CAMERA_TARGET = (OBJ_XY[0], OBJ_XY[1], OBJ_SIZE)

    def add_scene(self, builder):
        obj = builder.add_body(xform=wp.transform(wp.vec3(*OBJ_XY, OBJ_SIZE / 2), wp.quat_identity()), label="box")
        h = OBJ_SIZE / 2
        builder.add_shape_box(obj, hx=h, hy=h, hz=h, cfg=newton.ModelBuilder.ShapeConfig(density=OBJ_MASS / OBJ_SIZE**3, mu=1.0))

    def __init__(self, viewer, args, seed=0):
        super().__init__(viewer, args)
        self.rng = np.random.default_rng(seed)
        self.arm_q = np.array(self.arm_q_index)  # アームは q と dof の並びが同じ
        self.ik = ArmIK(args.urdf, np.radians(ARM_Q_DEG))
        labels = list(self.model.body_label)
        self.tips = [next(i for i, l in enumerate(labels) if l.endswith(f"{f}_thf_tip_link")) for f in ("left", "right")]
        self.init_objects()
        self.reset()

    def init_objects(self):
        j = next(i for i, l in enumerate(self.model.joint_label) if self.model.joint_child.numpy()[i] == self.box_body)
        self.obj_q = int(self.model.joint_q_start.numpy()[j])  # ワークの自由関節 (位置 3 + 四元数 4)
        self.obj_dof = int(self.model.joint_qd_start.numpy()[j])
        self.obj_q_slots = [self.obj_q]

    # ---- RL インターフェース ----
    @property
    def obs_dim(self):
        return len(self.observe())

    act_dim = len(ACT_JOINTS) + 1
    yaw_period = np.pi / 2  # ワークの yaw 対称性の周期 (立方体は 90 度)
    yaw_valid = 1.0  # 0 なら yaw に意味がない (球など)

    def spawn_objects(self):
        """ワークを置き、状態に反映する。self.est_pos / est_yaw / grasp_yaw / obj_half_h / obj_z0 / box_body を決める。"""
        rng = self.rng
        self.obj_xy = np.array(OBJ_XY) + rng.uniform(-OBJ_XY_RANGE, OBJ_XY_RANGE, 2)
        self.obj_yaw = rng.uniform(-np.pi, np.pi)
        self.est_pos = np.array([*self.obj_xy, OBJ_SIZE / 2]) + rng.normal(0.0, EST_POS_STD, 3)
        self.est_yaw = self.obj_yaw + rng.normal(0.0, EST_YAW_STD)
        # 爪の並び (手先 y) をワークの辺に直交させる。90 度対称なので基準姿勢に近い側を選ぶ
        self.grasp_yaw = wrap_quarter(self.est_yaw)
        self.obj_half_h = OBJ_SIZE / 2  # 推定位置 (物体の中心) から上面までの高さ
        self.obj_z0 = OBJ_SIZE / 2
        q = self.q0.copy()
        q[self.obj_q : self.obj_q + 3] = [*self.obj_xy, OBJ_SIZE / 2]
        q[self.obj_q + 3 : self.obj_q + 7] = list(yaw_quat(self.obj_yaw))
        self.assign_q(q)

    def assign_q(self, q):
        for s in (self.state_0, self.state_1):
            s.joint_q.assign(q)
            s.joint_qd.zero_()
            newton.eval_fk(self.model, s.joint_q, s.joint_qd, s)

    def reset(self):
        self.spawn_objects()
        self.q_pre = self.ik.solve(self.est_pos + [0.0, 0.0, self.obj_half_h + PRE_HEIGHT], self.grasp_yaw)
        # ロボットは前のエピソードの指・ネイルの状態を引きずらないよう q0 から作り直す。物体の姿勢だけ引き継ぐ
        cur = self.state_0.joint_q.numpy()
        q = self.q0.copy()
        for k in self.obj_q_slots:
            q[k : k + 7] = cur[k : k + 7]
        q[self.arm_q] = self.q_pre
        self.assign_q(q)
        tgt = self.control.joint_target_q.numpy()
        tgt[self.arm_q] = self.q_pre
        self.control.joint_target_q.assign(tgt)
        self.q_cmd = self.q_pre.copy()
        self.grip_cmd = 1.0
        self.set_gripper(GRIP_OPEN_MAX_MM)
        self.act_prev = np.zeros(self.act_dim)
        self.act_filt = np.array([0.0] * len(ACT_JOINTS) + [1.0])
        self.nail_hist = [np.zeros(len(NAIL_KEYS))] * NAIL_HIST
        self.t = 0
        self.hold = 0
        self.lift_max = 0.0
        self.tip_z_min = self.tip_mid()[2]
        for _ in range(10):  # 開いた状態で落ち着かせる
            super().step()
        self.nail_hist = [self.nail_vec()] * NAIL_HIST
        obj = self.state_0.body_q.numpy()[self.box_body][:3]
        self.phi_prev = potential(np.linalg.norm(self.tip_mid() - obj), 0, obj[2] - self.obj_z0)
        return self.observe()

    def step_env(self, action):
        a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        self.act_filt = ACT_EMA * a + (1.0 - ACT_EMA) * self.act_filt
        self.q_cmd[list(ACT_JOINTS)] += self.act_filt[: len(ACT_JOINTS)] * ACT_SCALE
        tgt = self.control.joint_target_q.numpy()
        tgt[self.arm_q] = self.q_cmd
        self.control.joint_target_q.assign(tgt)
        self.set_gripper((self.act_filt[-1] + 1.0) / 2.0 * GRIP_OPEN_MAX_MM)
        for _ in range(CTRL_FRAMES):
            super().step()
        self.t += 1
        self.nail_hist = self.nail_hist[1:] + [self.nail_vec()]

        obj = self.state_0.body_q.numpy()[self.box_body][:3]
        lift = obj[2] - self.obj_z0
        self.lift_max = max(self.lift_max, lift)
        d = np.linalg.norm(self.tip_mid() - obj)
        nails = self.nail_hist[-1]
        touching = [
            abs(nails[3 * k + 2]) > NAIL_ACTIVE_MM or max(abs(nails[3 * k]), abs(nails[3 * k + 1])) > NAIL_ACTIVE_DEG
            for k in range(2)
        ]
        self.hold = self.hold + 1 if lift > LIFT_SUCCESS else 0
        success = self.hold >= HOLD_STEPS
        dropped = self.lift_max > 0.01 and lift < 0.003
        # 持ち上げ動作をしたのに物体がついてこない (空振り・掴み損ね)。爪先が最低点から MISS_RISE 上がった時点で打ち切る
        self.tip_z_min = min(self.tip_z_min, self.tip_mid()[2])
        missed = (self.tip_mid()[2] - self.tip_z_min > MISS_RISE) and lift < 0.005 and not dropped

        phi = potential(d, sum(touching), lift)
        reward = (
            phi - self.phi_prev
            + (20.0 if success else 0.0)
            - (5.0 if dropped or missed else 0.0)
            - 0.01 * float(np.sum((a - self.act_prev) ** 2))
        )
        self.act_prev = a
        self.phi_prev = phi
        done = success or dropped or missed or self.t >= EP_STEPS or not np.isfinite(obj).all()
        info = {"success": success, "dropped": dropped, "missed": missed, "lift": lift, "dist": d, "touching": touching}
        return self.observe(), float(reward), done, info

    def observe(self):
        q = self.state_0.joint_q.numpy()
        qd = self.state_0.joint_qd.numpy()
        tip = self.tip_mid()
        # ワークの推定位置を、手先 yaw で回した爪先基準の座標で表す
        tool_yaw = self.grasp_yaw + (q[self.arm_q[5]] - self.q_pre[5])
        c, s = np.cos(-tool_yaw), np.sin(-tool_yaw)
        rel = self.est_pos - tip
        rel = np.array([c * rel[0] - s * rel[1], s * rel[0] + c * rel[1], rel[2]])
        dyaw = wrap_period(self.est_yaw - tool_yaw, self.yaw_period)
        return np.concatenate(
            [
                q[self.arm_q],
                qd[self.arm_q],
                q[self.arm_q] - self.q_pre,
                [self.gripper_opening_mm() / GRIP_OPEN_MAX_MM],
                np.concatenate(self.nail_hist) / np.tile([7.0, 7.0, sim.NAIL_PUSH_MAX_MM] * 2, NAIL_HIST),
                rel / 0.05,
                [self.yaw_valid * np.sin(2 * np.pi / self.yaw_period * dyaw), self.yaw_valid * np.cos(2 * np.pi / self.yaw_period * dyaw)],
                self.act_filt,
            ]
        ).astype(np.float32)

    # ---- 補助 ----
    def tip_mid(self):
        bq = self.state_0.body_q.numpy()
        return (bq[self.tips[0]][:3] + bq[self.tips[1]][:3]) / 2

    def nail_vec(self):
        n = self.nail_state()
        return np.array([n[k] for k in NAIL_KEYS])


class ScriptedPolicy:
    """推定姿勢だけを使う手順方策 (下降 → 閉じる → 持ち上げ)。報酬と成功判定の確認用。
    IK で求めた中継点への関節差分を行動に変換するので、RL と同じ行動空間を通る。"""

    GRASP_DEPTH = 0.012  # 爪先をワーク上面からこれだけ下げる [m]

    def __init__(self, env):
        self.env = env
        e = env
        top = e.est_pos[2] + e.obj_half_h
        xy = e.est_pos[:2]
        depth = min(self.GRASP_DEPTH, 1.2 * e.obj_half_h)  # 小さい物は中心の高さまで
        self.q_grasp = e.ik.solve([*xy, top - depth], e.grasp_yaw, q_init=e.q_pre)
        self.q_lift = e.ik.solve([*xy, top + 0.06], e.grasp_yaw, q_init=e.q_pre)
        self.phase, self.n = "descend", 0

    def __call__(self, obs):
        e = self.env
        self.n += 1
        grip = 1.0
        if self.phase == "descend":
            q_goal = self.q_grasp
            if np.abs(e.q_cmd - q_goal).max() < 1e-3 and self.n > 5:
                self.phase, self.n = "close", 0
        elif self.phase == "close":
            q_goal, grip = self.q_grasp, -1.0
            if self.n > 15:
                self.phase, self.n = "lift", 0
        else:
            q_goal, grip = self.q_lift, -1.0
        dq = (q_goal - e.q_cmd)[list(ACT_JOINTS)] / ACT_SCALE
        dq /= max(1.0, np.abs(dq).max())  # 全関節を同じ比率で縮めて、関節空間で直線的に動かす
        # EMA を打ち消して目標に素直に向かう
        a = (dq - (1.0 - ACT_EMA) * e.act_filt[: len(ACT_JOINTS)]) / ACT_EMA
        g = (grip - (1.0 - ACT_EMA) * e.act_filt[-1]) / ACT_EMA
        return np.clip(np.append(a, g), -1.0, 1.0)


class Demo:
    """newton.examples.run 用のラッパ。スクリプト方策でエピソードを回して結果を表示する。"""

    def __init__(self, viewer, args, env_cls=GraspEnv):
        self.env = env_cls(viewer, args, seed=args.seed)
        self.args = args
        self.episode = 0
        self.results = []
        self.start_episode()

    def start_episode(self):
        if self.episode > 0:
            self.obs = self.env.reset()
        else:
            self.obs = self.env.observe()
        self.policy = ScriptedPolicy(self.env)
        self.ret = 0.0
        e = self.env
        print(
            f"--- ep {self.episode}: ワーク=({e.obj_xy[0]:.3f},{e.obj_xy[1]:.3f}) yaw={np.degrees(e.obj_yaw):+.0f}deg "
            f"推定誤差={np.linalg.norm(e.est_pos[:2] - e.obj_xy) * 1e3:.1f}mm obs_dim={len(self.obs)}"
        )

    def step(self):
        if self.episode >= self.args.episodes:
            return
        self.obs, r, done, info = self.env.step_env(self.policy(self.obs))
        self.ret += r
        if self.env.t % 10 == 0 or done:
            n = self.env.nail_hist[-1]
            print(
                f"t={self.env.t:3d} {self.policy.phase:7s} r={r:+6.2f} lift={info['lift'] * 1e3:+6.1f}mm "
                f"dist={info['dist'] * 1e3:5.1f}mm grip={self.env.gripper_opening_mm():4.1f}mm "
                f"push=({n[2]:+.2f},{n[5]:+.2f})mm touch={info['touching']}"
            )
        if done:
            print(f"=> success={info['success']} dropped={info['dropped']} missed={info['missed']} return={self.ret:.1f}")
            self.results.append(info["success"])
            self.episode += 1
            if self.episode < self.args.episodes:
                self.start_episode()

    def render(self):
        self.env.render()

    def test_final(self):
        print(f"成功 {sum(self.results)}/{len(self.results)}")


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--urdf", default="tma.urdf")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.set_defaults(num_frames=10**9)
    viewer, args = newton.examples.init(parser)
    demo = Demo(viewer, args)
    # num_frames ではなくエピソード数で止める
    while viewer.is_running() and demo.episode < args.episodes:
        demo.step()
        demo.render()
    demo.test_final()
    viewer.close()
