"""ばら積み把持環境の並列版 (num_envs 個のワールドを 1 つのモデルに入れて GPU で同時に回す)。

物理 (衝突・ソルバ) は GPU で一括、観測・報酬・リセットは numpy で全環境まとめて計算する。
単一環境版 (grasp.GraspEnv / grasp_pile.PileEnv) と観測・行動・報酬は同じ。

- 山は起動時に全ワールドで同時に落として静止させ、ライブラリとして溜める。リセットはそこから引いて、
  アームを q_pre (バッチ IK) に置くだけなので、リセット時に物理を回さない
- done になった環境は step() の中で自動リセットされ、戻り値の obs は次のエピソードの最初の観測

    uv run tma/grasp_pile_vec.py --viewer null --num-envs 64 --episodes 200   # スクリプト方策で動作確認
    uv run tma/grasp_pile_vec.py --num-envs 4 --print-every 5                 # 爪の状態をコンソールと GL のパネル・グラフに表示
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # リポジトリ直下 (tma パッケージ)

import numpy as np
import warp as wp

import newton
import newton.examples
from tma import grasp as ge
from tma import grasp_pile as gp
from tma import sim


class VecGraspEnv(sim.Sim):
    NJMAX, NCONMAX = gp.PileEnv.NJMAX, gp.PileEnv.NCONMAX
    RIGID_CONTACTS_PER_ENV = gp.PileEnv.RIGID_CONTACTS_PER_ENV
    ARM_Q_DEG = ge.ARM_Q_DEG
    CAMERA_TARGET = (*gp.BIN_CENTER, 0.02)
    act_dim = ge.GraspEnv.act_dim
    CLOSE_STEPS = 1  # スクリプト方策が開から閉へゆっくり閉じるステップ数。1 だと一気に閉じて物体を弾く
    PHASE_TOL = 1e-3  # スクリプト方策が「到達」とみなす関節目標の誤差 [rad]
    LIBRARY_ROUNDS = 2  # 山のライブラリ: 全環境で同時に作る回数 (num_envs * 回数 個の山)
    # 学習スクリプトが集計に使う結果の種類 (info のキー, 表示名)
    OUTCOMES = (("success", "成功"), ("dropped", "落下"), ("missed", "空振り"), ("timeout", "時間切れ"))

    def add_scene(self, builder):
        gp.PileEnv.add_scene(self, builder)

    def __init__(self, viewer, args, seed=0, scripted=False):
        super().__init__(viewer, args)
        self.rng = np.random.default_rng(seed)
        self.scripted = scripted
        N = self.num_envs
        m = self.model
        self.C, self.D, self.B = m.joint_coord_count // N, m.joint_dof_count // N, m.body_count // N
        self.arm_idx = np.array(self.arm_q_index)  # アーム 6 関節の q / dof インデックス (ワールド内)
        labels = list(m.joint_label)
        q_start = m.joint_q_start.numpy()
        gr = next(i for i, l in enumerate(labels) if l.endswith("robotiq_hande_right_finger_joint"))
        self.gripper_rq = int(q_start[gr])
        gp.PileEnv.init_objects(self)  # obj_bodies / obj_q_slots (ワールド 0)
        self.slot_idx = np.array(self.obj_q_slots)[:, None] + np.arange(7)  # [N_OBJ, 7]
        self.q_init = m.joint_q.numpy().reshape(N, self.C)[0].copy()
        # joint_target_q は dof 数ではなく q の数 (自由関節は 7) で並ぶ。ロボットの関節は q と dof が同じ番号
        self.T = self.control.joint_target_q.shape[0] // N
        self.tgt_init = self.control.joint_target_q.numpy().reshape(N, self.T)[0].copy()
        # URDF の初期値はグリッパ全閉で、バラ積み爪だと左右の爪が 12mm めり込み接触があふれる。開いた状態を初期値にする
        opening = ge.GRIP_OPEN_MAX_MM * 1e-3
        self.q_init[[self.gripper_q, self.gripper_rq]] = opening
        self.tgt_init[[self.gripper_dof, self.gripper_r_dof]] = opening
        self.control.joint_target_q.assign(np.tile(self.tgt_init, N))
        self.write(np.tile(self.q_init, (N, 1)), np.zeros((N, self.D)))
        self.tips = np.array(self.tips_of_world0())
        self.ik = ge.ArmIK(args.urdf, np.radians(ge.ARM_Q_DEG), n=N)

        # 環境ごとの状態
        self.t = np.zeros(N, dtype=int)
        self.hold = np.zeros(N, dtype=int)
        self.lift_max = np.zeros(N)
        self.tip_z_min = np.zeros(N)
        self.phi_prev = np.zeros(N)
        self.obj_z0 = np.zeros(N)
        self.target_k = np.zeros(N, dtype=int)
        self.target_type = [""] * N
        self.est_pos = np.zeros((N, 3))
        self.est_yaw = np.zeros(N)
        self.grasp_yaw = np.zeros(N)
        self.yaw_valid = np.zeros(N)
        self.half_h = np.zeros(N)
        self.q_pre = np.zeros((N, 6))
        self.q_cmd = np.zeros((N, 6))
        self.act_filt = np.zeros((N, self.act_dim))
        self.act_prev = np.zeros((N, self.act_dim))
        self.nail_hist = np.zeros((N, ge.NAIL_HIST, 6))
        self.q_grasp = np.zeros((N, 6))
        self.q_lift = np.zeros((N, 6))
        self.phase = np.zeros(N, dtype=int)
        self.phase_n = np.zeros(N, dtype=int)
        self.disp = None  # 表示用のスナップショット (step_env のリセット前の値)
        self.watch = 0  # 爪の状態を表示する環境

        t0 = time.time()
        self.build_library()
        print(f"山のライブラリ: {len(self.piles)} 個 ({time.time() - t0:.1f} s)")
        self.reset_envs(np.arange(N))
        self.obs = self.observe()
        if hasattr(viewer, "register_ui_callback"):
            viewer.register_ui_callback(self.nail_ui, "side")

    @property
    def obs_dim(self):
        return self.obs.shape[1]

    def tips_of_world0(self):
        labels = list(self.model.body_label)
        return [next(i for i, l in enumerate(labels) if l.endswith(f"{f}_thf_tip_link")) for f in ("left", "right")]

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

    # ---- 山のライブラリ ----
    def build_library(self):
        N, rng = self.num_envs, self.rng
        piles = []
        for _ in range(self.LIBRARY_ROUNDS):
            q = np.tile(self.q_init, (N, 1))
            for w in range(N):
                for slot, k in enumerate(rng.permutation(gp.N_OBJ)):
                    layer, rem = divmod(slot, gp.DROP_GRID**2)
                    ix, iy = divmod(rem, gp.DROP_GRID)
                    s = self.obj_q_slots[k]
                    q[w, s : s + 3] = [
                        gp.BIN_CENTER[0] + (ix - (gp.DROP_GRID - 1) / 2) * gp.DROP_PITCH + rng.uniform(-0.004, 0.004),
                        gp.BIN_CENTER[1] + (iy - (gp.DROP_GRID - 1) / 2) * gp.DROP_PITCH + rng.uniform(-0.004, 0.004),
                        gp.DROP_Z0 + layer * gp.DROP_DZ,
                    ]
                    q[w, s + 3 : s + 7] = gp.random_quat(rng)
            self.write(q, np.zeros((N, self.D)))
            for _ in range(gp.SETTLE_FRAMES):
                sim.Sim.step(self)
            qs, _, _ = self.read()
            for w in range(N):
                pile = qs[w][self.slot_idx]  # [N_OBJ, 7]
                cheb = np.maximum(np.abs(pile[:, 0] - gp.BIN_CENTER[0]), np.abs(pile[:, 1] - gp.BIN_CENTER[1]))
                # ビンの外に出た・飛んだ物体があれば捨てる
                if np.isfinite(pile).all() and (cheb < gp.BIN_HALF + 0.003).all() and (pile[:, 2] < gp.WALL_H).all():
                    piles.append(pile.copy())
        if not piles:
            raise RuntimeError("使える山が 1 つも作れなかった")
        self.piles = np.stack(piles)

    # ---- リセット ----
    def reset_envs(self, idx):
        idx = np.asarray(idx)
        n, N, rng = len(idx), self.num_envs, self.rng
        if n == 0:
            return
        pick = rng.integers(len(self.piles), size=n)
        piles = self.piles[pick]
        for i, w in enumerate(idx):
            t = gp.select_target(piles[i][:, :3], piles[i][:, 3:7], self.obj_names, rng)
            self.target_k[w], self.target_type[w] = t["k"], t["type"]
            self.est_pos[w], self.est_yaw[w], self.grasp_yaw[w] = t["est_pos"], t["est_yaw"], t["grasp_yaw"]
            self.obj_z0[w], self.half_h[w], self.yaw_valid[w] = t["z0"], t["half_h"], t["yaw_valid"]

        def solve(height, q_init=None):
            """全環境ぶんの問題に詰めて解く (idx 以外は idx[0] の問題で埋める)"""
            pos = np.tile(self.est_pos[idx[0]] + [0.0, 0.0, height(idx[0])], (N, 1))
            yaw = np.full(N, self.grasp_yaw[idx[0]])
            qi = None if q_init is None else np.tile(q_init[0], (N, 1))
            pos[:n] = [self.est_pos[w] + [0.0, 0.0, height(w)] for w in idx]
            yaw[:n] = self.grasp_yaw[idx]
            if q_init is not None:
                qi[:n] = q_init
            return self.ik.solve_batch(pos, yaw, qi)[:n]

        self.q_pre[idx] = solve(lambda w: self.half_h[w] + ge.PRE_HEIGHT)
        if self.scripted:
            depth = ge.grasp_depth(self.half_h)
            self.q_grasp[idx] = solve(lambda w: self.half_h[w] - depth[w], self.q_pre[idx])
            self.q_lift[idx] = solve(lambda w: self.half_h[w] + 0.06, self.q_pre[idx])
            self.phase[idx], self.phase_n[idx] = 0, 0

        q, qd, _ = self.read()
        tgt = self.control.joint_target_q.numpy().reshape(N, self.T)
        opening = ge.GRIP_OPEN_MAX_MM * 1e-3
        for i, w in enumerate(idx):
            row = self.q_init.copy()
            row[self.slot_idx] = piles[i]
            row[self.arm_idx] = self.q_pre[w]
            row[self.gripper_q] = row[self.gripper_rq] = opening
            q[w], qd[w] = row, 0.0
            tgt[w] = self.tgt_init
            tgt[w, self.arm_idx] = self.q_pre[w]
            tgt[w, self.gripper_dof] = tgt[w, self.gripper_r_dof] = opening
        self.write(q, qd)
        self.control.joint_target_q.assign(tgt.reshape(-1))

        self.q_cmd[idx] = self.q_pre[idx]
        self.act_filt[idx] = [0.0] * len(ge.ACT_JOINTS) + [1.0]
        self.act_prev[idx] = 0.0
        self.nail_hist[idx] = 0.0
        self.t[idx], self.hold[idx], self.lift_max[idx] = 0, 0, 0.0
        bq = self.read()[2]
        tip = self.tip_mid(bq)
        self.tip_z_min[idx] = tip[idx, 2]
        obj = bq[idx, np.array(self.obj_bodies)[self.target_k[idx]], :3]
        self.phi_prev[idx] = ge.potential(np.linalg.norm(tip[idx] - obj, axis=1), 0, obj[:, 2] - self.obj_z0[idx])

    # ---- 計測 ----
    def tip_mid(self, bq):
        return (bq[:, self.tips[0], :3] + bq[:, self.tips[1], :3]) / 2

    def nail_vec(self, q):
        out = np.empty((self.num_envs, 6))
        for j, key in enumerate(ge.NAIL_KEYS):
            v = q[:, self.nail_q_index[key]]
            out[:, j] = v * 1e3 if key.endswith("z") else np.degrees(v)
        return out

    def observe(self):
        q, qd, bq = self.read()
        tip = self.tip_mid(bq)
        arm_q = q[:, self.arm_idx]
        tool_yaw = self.grasp_yaw + (arm_q[:, 5] - self.q_pre[:, 5])
        c, s = np.cos(-tool_yaw), np.sin(-tool_yaw)
        rel = self.est_pos - tip
        rel = np.stack([c * rel[:, 0] - s * rel[:, 1], s * rel[:, 0] + c * rel[:, 1], rel[:, 2]], axis=1)
        dyaw = ge.wrap_period(self.est_yaw - tool_yaw, np.pi)
        k = 2 * np.pi / np.pi
        scale = np.tile([7.0, 7.0, sim.NAIL_PUSH_MAX_MM] * 2, ge.NAIL_HIST)
        return np.concatenate(
            [
                arm_q,
                qd[:, self.arm_idx],
                arm_q - self.q_pre,
                (q[:, self.gripper_q] * 1e3 / ge.GRIP_OPEN_MAX_MM)[:, None],
                self.nail_hist.reshape(self.num_envs, -1) / scale,
                rel / 0.05,
                (self.yaw_valid * np.sin(k * dyaw))[:, None],
                (self.yaw_valid * np.cos(k * dyaw))[:, None],
                self.act_filt,
            ],
            axis=1,
        ).astype(np.float32)

    # ---- ステップ ----
    def step_env(self, action):
        N = self.num_envs
        a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        self.act_filt = ge.ACT_EMA * a + (1.0 - ge.ACT_EMA) * self.act_filt
        self.q_cmd[:, list(ge.ACT_JOINTS)] += self.act_filt[:, : len(ge.ACT_JOINTS)] * ge.ACT_SCALE
        tgt = self.control.joint_target_q.numpy().reshape(N, self.T)
        tgt[:, self.arm_idx] = self.q_cmd
        grip = np.clip(ge.grip_mm(self.act_filt[:, -1]) * 1e-3, 0.0, sim.HANDE_STROKE)
        tgt[:, self.gripper_dof] = tgt[:, self.gripper_r_dof] = grip
        self.control.joint_target_q.assign(tgt.reshape(-1))
        for _ in range(ge.CTRL_FRAMES):
            sim.Sim.step(self)
        self.t += 1

        q, qd, bq = self.read()
        self.nail_hist = np.concatenate([self.nail_hist[:, 1:], self.nail_vec(q)[:, None]], axis=1)
        obj = bq[np.arange(N), np.array(self.obj_bodies)[self.target_k], :3]
        tip = self.tip_mid(bq)
        lift = obj[:, 2] - self.obj_z0
        self.lift_max = np.maximum(self.lift_max, lift)
        d = np.linalg.norm(tip - obj, axis=1)
        nails = self.nail_hist[:, -1].reshape(N, 2, 3)  # [env, 指, (x, y, z)]
        touching = (np.abs(nails[:, :, 2]) > ge.NAIL_ACTIVE_MM) | (np.abs(nails[:, :, :2]).max(axis=2) > ge.NAIL_ACTIVE_DEG)
        self.disp = dict(
            t=self.t.copy(), phase=self.phase.copy(), nail=self.nail_hist[:, -1].copy(), touching=touching,
            grip=q[:, self.gripper_q] * 1e3, lift=lift, type=list(self.target_type),
        )
        self.hold = np.where(lift > ge.LIFT_SUCCESS, self.hold + 1, 0)
        success = self.hold >= ge.HOLD_STEPS
        dropped = (self.lift_max > 0.01) & (lift < 0.003)
        self.tip_z_min = np.minimum(self.tip_z_min, tip[:, 2])
        missed = (tip[:, 2] - self.tip_z_min > ge.MISS_RISE) & (lift < 0.005) & ~dropped

        phi = ge.potential(d, touching.sum(axis=1), lift)
        reward = (
            phi - self.phi_prev
            + 20.0 * success
            - 5.0 * (dropped | missed)
            - 0.01 * ((a - self.act_prev) ** 2).sum(axis=1)
        )
        # 物理が発散した環境 (まれ。256 環境で PPO の 1 イテレーション目に起きた)。報酬に NaN が 1 つでも混ざると
        # 学習側の報酬スケールの統計が NaN になり以降の更新が全部壊れるので、報酬 0 で打ち切り、時間切れにも数えない
        blown = ~(np.isfinite(obj).all(axis=1) & np.isfinite(q).all(axis=1) & np.isfinite(qd).all(axis=1))
        reward[blown] = 0.0
        phi[blown] = 0.0
        self.act_prev = a
        self.phi_prev = phi
        done = success | dropped | missed | (self.t >= ge.EP_STEPS) | blown
        info = dict(
            success=success & done, dropped=dropped & done, missed=missed & done,
            timeout=done & ~(success | dropped | missed | blown), blown=blown, lift=lift, dist=d,
            type=[self.target_type[w] for w in range(N)],  # リセット前の対象の形状
        )
        done_idx = np.nonzero(done)[0]
        if info["timeout"].any():
            # 時間切れ環境の value ブートストラップ用 (リセット前の観測)。発散した環境の行は使わないが NaN * 0 にならないよう 0 に
            info["final_obs"] = np.nan_to_num(self.observe(), nan=0.0, posinf=0.0, neginf=0.0)
        self.reset_envs(done_idx)
        if blown.any():
            # MuJoCo は前ステップの加速度 (qacc_warmstart) を次の解の初期値に使うので、関節状態を書き直しても
            # NaN が残って毎ステップ発散し続ける。そのワールドの内部バッファだけ消す (flags=0 で関節状態は触らない)
            mask = np.zeros(N + 1, dtype=bool)
            mask[:N] = blown
            self.solver.reset(self.state_0, wp.array(mask, dtype=wp.bool), flags=0)
        self.obs = self.observe()
        return self.obs, reward.astype(np.float32), done, info

    # ---- 確認用のスクリプト方策 (下降 → 閉じる → 持ち上げ) ----
    def scripted_action(self):
        goal = np.where((self.phase == 2)[:, None], self.q_lift, self.q_grasp)
        ramp = 1.0 - 2.0 * np.clip(self.phase_n / self.CLOSE_STEPS, 0.0, 1.0)  # phase 1 に入ってから 1 → -1
        grip = np.where(self.phase == 0, 1.0, np.where(self.phase == 1, ramp, -1.0))
        dq = (goal - self.q_cmd)[:, list(ge.ACT_JOINTS)] / ge.ACT_SCALE
        dq /= np.maximum(1.0, np.abs(dq).max(axis=1, keepdims=True))
        e = ge.ACT_EMA
        a = (dq - (1.0 - e) * self.act_filt[:, : len(ge.ACT_JOINTS)]) / e
        g = (grip - (1.0 - e) * self.act_filt[:, -1]) / e
        self.phase_n += 1
        to_close = (self.phase == 0) & (np.abs(self.q_cmd - self.q_grasp).max(axis=1) < self.PHASE_TOL) & (self.phase_n > 5)
        to_lift = (self.phase == 1) & (self.phase_n > self.CLOSE_STEPS + 14)
        self.phase = np.where(to_close, 1, np.where(to_lift, 2, self.phase))
        self.phase_n = np.where(to_close | to_lift, 0, self.phase_n)
        return np.clip(np.column_stack([a, g]), -1.0, 1.0)

    # ---- 爪の状態の表示 ----
    PHASE_NAMES = ("descend", "close", "lift")

    def nail_text(self, w):
        d = self.disp
        n = d["nail"][w]
        phase = self.PHASE_NAMES[d["phase"][w]] if self.scripted else ""
        tc = "".join("*" if t else "-" for t in d["touching"][w])
        return (
            f"env{w} t={d['t'][w]:3d} {phase:7s} grip={d['grip'][w]:4.1f}mm lift={d['lift'][w] * 1e3:+5.1f}mm | "
            f"left push={n[2]:4.2f}mm x={n[0]:+5.2f} y={n[1]:+5.2f}deg | "
            f"right push={n[5]:4.2f}mm x={n[3]:+5.2f} y={n[4]:+5.2f}deg | touch={tc}"
        )

    def nail_ui(self, imgui):
        """GL ビューアの左パネル: 注目環境の爪の値をバーで出す"""
        _, self.watch = imgui.slider_int("watch env", self.watch, 0, self.num_envs - 1)
        d = self.disp
        if d is None:
            return
        w = self.watch
        phase = self.PHASE_NAMES[d["phase"][w]] if self.scripted else ""
        imgui.text(f"t={d['t'][w]}  {phase}  target={d['type'][w]}")
        imgui.text(f"grip {d['grip'][w]:.1f} mm   lift {d['lift'][w] * 1e3:+.1f} mm")
        for f in range(2):
            x, y, z = d["nail"][w, 3 * f : 3 * f + 3]
            imgui.separator()
            imgui.text(f"{('left', 'right')[f]} nail   {'TOUCH' if d['touching'][w, f] else '-'}")
            imgui.progress_bar(float(np.clip(z / sim.NAIL_PUSH_MAX_MM, 0, 1)), imgui.ImVec2(-1, 0), f"push {z:.2f} mm")
            for name, v in (("x", x), ("y", y)):  # 傾きは中央が 0、±7deg が端
                imgui.progress_bar(float(np.clip(0.5 + v / 14.0, 0, 1)), imgui.ImVec2(-1, 0), f"tilt {name} {v:+.2f} deg")

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        if self.disp is not None:
            w = self.watch
            for j, key in enumerate(ge.NAIL_KEYS):
                unit = "mm" if key.endswith("z") else "deg"
                self.viewer.log_scalar(f"nail/{key}_{unit}", float(self.disp["nail"][w, j]))
            # 全環境の押し込み量 [env, 指] をヒートマップで
            self.viewer.log_array("nail/push_mm_all_envs", self.disp["nail"][:, [2, 5]].astype(np.float32))
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--urdf", default="tma.urdf")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--episodes", type=int, default=200, help="終わったエピソードの合計がこの数に達したら止める")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pos-noise-mm", type=float, default=ge.EST_POS_STD * 1e3)
    parser.add_argument("--yaw-noise-deg", type=float, default=float(np.degrees(ge.EST_YAW_STD)))
    parser.add_argument("--close-steps", type=int, default=VecGraspEnv.CLOSE_STEPS, help="スクリプト方策が閉じるのにかけるステップ数")
    parser.add_argument("--impratio", type=float, default=None, help="MuJoCo の impratio (摩擦拘束の硬さ)。既定は sim.Sim.SOLVER_KW")
    parser.add_argument("--contacts-per-env", type=int, default=None, help="Newton の接触バッファの 1 環境あたりの点数 (0 で既定の見積もり)")
    parser.add_argument("--watch", type=int, default=0, help="爪の状態を表示する環境の番号 (GL ではパネルで変えられる)")
    parser.add_argument("--print-every", type=int, default=0, help="この step ごとに注目環境の爪の状態を表示 (0 で表示しない)")
    parser.set_defaults(num_frames=10**9)
    viewer, args = newton.examples.init(parser)
    ge.EST_POS_STD = args.pos_noise_mm * 1e-3
    ge.EST_YAW_STD = np.radians(args.yaw_noise_deg)
    VecGraspEnv.CLOSE_STEPS = args.close_steps
    if args.contacts_per_env is not None:
        VecGraspEnv.RIGID_CONTACTS_PER_ENV = args.contacts_per_env or None
    if args.impratio is not None:
        VecGraspEnv.SOLVER_KW = {**sim.Sim.SOLVER_KW, "impratio": args.impratio}
    env = VecGraspEnv(viewer, args, seed=args.seed, scripted=True)
    env.watch = args.watch
    if hasattr(viewer, "hide_loading_splash"):  # init() が出したスプラッシュ。run() を使わないので自分で消す
        viewer.hide_loading_splash()
    print(f"num_envs={env.num_envs} obs_dim={env.obs_dim} act_dim={env.act_dim}")

    stats = {}
    finished, steps = 0, 0
    t0 = time.time()
    while finished < args.episodes and viewer.is_running():
        obs, rew, done, info = env.step_env(env.scripted_action())
        env.render()  # null ビューアでは何もしない。GL / viser ではここでイベント処理・描画が回る
        steps += env.num_envs
        w = env.watch
        if args.print_every and (done[w] or env.disp["t"][w] % args.print_every == 0):
            print(env.nail_text(w))
            if done[w]:
                res = next(k for k in ("success", "dropped", "missed", "timeout") if info[k][w])
                print(f"=> env{w} {info['type'][w]}: {res}")
        for w in np.nonzero(done)[0]:
            s = stats.setdefault(info["type"][w], [0, 0, 0, 0, 0])  # 試行, 成功, 落下, 空振り, 時間切れ
            s[0] += 1
            s[1] += bool(info["success"][w])
            s[2] += bool(info["dropped"][w])
            s[3] += bool(info["missed"][w])
            s[4] += bool(info["timeout"][w])
            finished += 1
    dt = time.time() - t0
    print(f"{steps} env-steps, {dt:.1f} s -> {steps / dt:.0f} env-steps/s, {finished} episodes")
    tot = np.sum(list(stats.values()), axis=0)
    print(f"  {'形状':8s} 試行 成功 落下 空振り 時間切れ")
    for name in sorted(stats):
        print(f"  {name:8s} " + " ".join(f"{v:4d}" for v in stats[name]))
    print(f"  {'合計':8s} " + " ".join(f"{v:4d}" for v in tot) + f"   成功率 {tot[1] / tot[0]:.2f}")
    viewer.close()
