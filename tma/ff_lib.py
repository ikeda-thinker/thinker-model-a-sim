"""既存の手探り制御ライブラリ (共有ドライブ nakano/ur_server_202608 の robot/driver.py) をばら積み把持環境に移植したもの。

実機版は 8ms 周期で手先座標系の差分 (slave_move_tool: [mm, mm, mm, deg, deg, deg]、z は下向き) を送る。ここでは
環境の制御周期 (20Hz) ごとに同じだけ手先の目標姿勢を進め、IK で関節目標にして VecGraspEnv.step_env(q_goal=...) で動かす。

- 高速アプローチ (ff_z_direction_approach に test_module1 の smootherstep 立ち上げを足したもの):
  手先 z に進み、どちらかの爪の押し込み d が d_th を超えるか z_lim (スクリプト方策と同じ把持深さ) に達したら止まる
- 位置決め (test_module2_exploratory_positioning_control): 無負荷で記録した爪の (d, θ1, θ2) を
  ライブラリと同じ前処理 (アップサンプル → 移動平均 → 差分 → スプライン平滑化) にかけ、手先の (z, rx, ry) として再生する。
  右爪の t2 か d が閾値を超えたら止まる。開始時にセンサの初期値を取り直す (ff_set_initial_sensor 相当)
- そのあと閉じて持ち上げる (ライブラリの範囲外。z_lim で止まった (接触しなかった) ときは位置決めを飛ばす)

センサの対応 (実機 → シミュ): [d1, t1_1, t2_1, d2, t1_2, t2_2] → [left_z, left_x, left_y, right_z, right_x, right_y] [mm, deg]

    uv run tma/ff_lib.py --viewer null --num-envs 16 --episodes 200           # ライブラリだけで把持
    uv run scripts/train_ppo.py --viewer null --eval ckpt/pile40_bc/ppo_latest.pt --policy hybrid   # アプローチだけライブラリ
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # リポジトリ直下 (tma パッケージ)

import numpy as np
from scipy.interpolate import UnivariateSpline

from tma import grasp as ge

TRAJ_CSV = Path(__file__).resolve().parents[1] / "data" / "ff_lib" / "no_load_motion.csv"
LIB_DT = 0.008  # 実機ライブラリの制御周期 [s] (params.control_period)
GAIN_SAVED = (1000.0, 60.0, 60.0)  # 保存軌道の拡大率 (assembly_task_trajectory_processor.gain_for_saved_trajectory)

APPROACH, POSITION, CLOSE, LIFT = range(4)
PHASE_NAMES = ("approach", "position", "close", "lift")


def smootherstep(s):
    s = np.clip(s, 0.0, 1.0)
    return 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5


def load_offsets(upsample, ma_window=100, spline_s=(5e-3, 2e-2, 5e-2)):
    """位置決めで再生する手先の変位 [k, 3] (z [mm], rx [deg], ry [deg]、ゲイン 1、LIB_DT 刻み)。
    ライブラリは差分 delta_pos をさらに前回との差 (delta_pos - delta_pos_old) にして送るので、
    手先の変位は軌道の差分 (爪の速度) にゲインを掛けたものになる"""
    a = np.loadtxt(TRAJ_CSV, delimiter=",", skiprows=1)[:, 1:4]  # t, d, θ1, θ2, ... の d, θ1, θ2
    s = np.linspace(0.0, 1.0, upsample, endpoint=False)
    up = ((1 - s)[None, :, None] * a[:-1, None] + s[None, :, None] * a[1:, None]).reshape(-1, 3)  # 線形補間
    up = np.vstack([up, a[-1:]])
    k = np.ones(ma_window) / ma_window
    pad = (ma_window // 2, ma_window - 1 - ma_window // 2)
    sm = np.stack([np.convolve(np.pad(up[:, c], pad, mode="edge"), k, mode="valid") for c in range(3)], axis=1)
    d = np.diff(sm, axis=0, prepend=sm[:1])
    t = np.arange(len(d), dtype=float)
    d = np.stack([UnivariateSpline(t, d[:, c], s=sv)(t) for c, sv in zip(range(3), spline_s)], axis=1)
    return d * GAIN_SAVED


def rot_xyz(r):
    """手先座標系の微小回転 (度, 各軸) [n, 3] → 回転行列 [n, 3, 3] (scipy の 'xyz' = 固定軸 x → y → z)"""
    a = np.radians(r)
    c, s = np.cos(a), np.sin(a)
    o, z = np.ones(len(a)), np.zeros(len(a))
    Rx = np.stack([o, z, z, z, c[:, 0], -s[:, 0], z, s[:, 0], c[:, 0]], 1).reshape(-1, 3, 3)
    Ry = np.stack([c[:, 1], z, s[:, 1], z, o, z, -s[:, 1], z, c[:, 1]], 1).reshape(-1, 3, 3)
    Rz = np.stack([c[:, 2], -s[:, 2], z, s[:, 2], c[:, 2], z, z, z, o], 1).reshape(-1, 3, 3)
    return Rz @ Ry @ Rx


def quat_to_mat(q):
    x, y, z, w = np.moveaxis(np.asarray(q, dtype=float), -1, 0)
    return np.stack(
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
         2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
         2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], axis=-1
    ).reshape(q.shape[:-1] + (3, 3))


def mat_to_quat(R):
    """回転行列 [n, 3, 3] → 四元数 [n, 4] (xyzw)"""
    w = np.sqrt(np.maximum(0.0, 1.0 + R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2])) / 2
    x = np.sqrt(np.maximum(0.0, 1.0 + R[:, 0, 0] - R[:, 1, 1] - R[:, 2, 2])) / 2
    y = np.sqrt(np.maximum(0.0, 1.0 - R[:, 0, 0] + R[:, 1, 1] - R[:, 2, 2])) / 2
    z = np.sqrt(np.maximum(0.0, 1.0 - R[:, 0, 0] - R[:, 1, 1] + R[:, 2, 2])) / 2
    x = np.copysign(x, R[:, 2, 1] - R[:, 1, 2])
    y = np.copysign(y, R[:, 0, 2] - R[:, 2, 0])
    z = np.copysign(z, R[:, 1, 0] - R[:, 0, 1])
    return np.stack([x, y, z, w], axis=1)


class FFController:
    """VecGraspEnv の全環境をライブラリの手順で動かす。act() で (行動, 関節目標) を返し、
    step_env のあと done になった環境を reset() に渡す"""

    # 高速アプローチ
    APPROACH_V = 50.0  # 手先 z の速さ [mm/s]
    RAMP_T = 0.3  # 0 → APPROACH_V の立ち上げ時間 [s] (test_module1 の ramp_time)
    # 押し込みの閾値 [mm]。実機は |d| > 0.12 だが、シミュの爪は動き出しの慣性で ±0.2mm ほど揺れて (引く向きが大きい)
    # すぐ誤検出するので、押し込む向きだけを ge.NAIL_ACTIVE_MM (シミュの接触判定) で見る
    D_TH = ge.NAIL_ACTIVE_MM
    APPROACH_TIMEOUT = 5.0  # [s]
    # 位置決め
    POSITION = True  # False で位置決めを飛ばす (アプローチ → すぐ閉じる)
    Z_GAIN, RX_GAIN, RY_GAIN = 0.2, 2.0, 2.0  # ゲイン 1 だと z ±10mm / rx ±1° / ry ±2.5° (upsample 2)
    UPSAMPLE = 2  # 記録軌道のアップサンプル倍率。2 で記録と同じ速さ (記録は 16ms 刻み、実行は 8ms 刻み)
    POS_T2_TH = 1.0  # 右爪の t2 の閾値 [deg]
    POS_D_TH = 0.5  # 右爪の d の閾値 [mm]
    POS_MAX_T = 3.0  # 位置決めの打ち切り [s] (記録軌道を全部再生すると UPSAMPLE=2 でも 9 s かかりエピソードに収まらない)
    # 把持
    CLOSE_T = 0.7  # 閉じ始めてから持ち上げるまで [s]
    LIFT_V = 60.0  # [mm/s]
    LIFT_H = 60.0  # [mm]

    def __init__(self, env):
        self.env = env
        N = env.num_envs
        self.dt = ge.CTRL_FRAMES / env.fps
        R_down = quat_to_mat(np.array(env.ik.down, dtype=float))
        # 手先座標系 (ライブラリの TCP 座標系: z = 下向き) をツールリンク座標で表した基底。x は基準姿勢でワールド +x
        z = R_down.T @ [0.0, 0.0, -1.0]
        x = R_down.T @ [1.0, 0.0, 0.0]
        self.C = np.stack([x, np.cross(z, x), z], axis=1)
        self.R_down = R_down
        self.offsets = load_offsets(self.UPSAMPLE)
        self.phase = np.zeros(N, dtype=int)
        self.t = np.zeros(N)
        self.pos = np.zeros((N, 3))  # 爪先中点の目標 [m]
        self.R = np.tile(np.eye(3), (N, 1, 1))  # ツールリンクの目標の向き
        self.z_lim = np.zeros(N)
        self.nail0 = np.zeros((N, 6))  # センサの初期値 (位置決め開始時に取り直す)
        self.k_prev = np.zeros(N, dtype=int)
        self.lifted = np.zeros(N)
        self.stop = np.array([""] * N, dtype=object)  # アプローチが止まった理由 (表示用)

    def reset(self, idx):
        e = self.env
        idx = np.asarray(idx)
        if len(idx) == 0:
            return
        self.phase[idx], self.t[idx], self.lifted[idx], self.nail0[idx] = APPROACH, 0.0, 0.0, 0.0
        self.pos[idx] = e.est_pos[idx] + np.stack([np.zeros(len(idx)), np.zeros(len(idx)), e.half_h[idx] + ge.PRE_HEIGHT], 1)
        c, s = np.cos(e.grasp_yaw[idx]), np.sin(e.grasp_yaw[idx])
        o, z = np.ones(len(idx)), np.zeros(len(idx))
        Rz = np.stack([c, -s, z, s, c, z, z, z, o], 1).reshape(-1, 3, 3)
        self.R[idx] = Rz @ self.R_down
        self.z_lim[idx] = e.est_pos[idx, 2] + e.half_h[idx] - ge.grasp_depth(e.half_h[idx])

    def move_tool(self, rows, dp_mm, drot_deg):
        """手先座標系の差分 (slave_move_tool と同じ単位) を目標姿勢に積む。rows は bool [N]"""
        if not rows.any():
            return
        Rt = self.R[rows] @ self.C  # 手先座標系 → ワールド
        self.pos[rows] += np.einsum("nij,nj->ni", Rt, dp_mm[rows] * 1e-3)
        self.R[rows] = Rt @ rot_xyz(drot_deg[rows]) @ self.C.T

    def act(self):
        e, N, dt = self.env, self.env.num_envs, self.dt
        raw = e.nail_hist[:, -1]  # [left_x, left_y, left_z, right_x, right_y, right_z]
        nail = raw - self.nail0
        d1, d2, t2 = nail[:, 2], nail[:, 5], nail[:, 4]
        dp, dr = np.zeros((N, 3)), np.zeros((N, 3))

        # ---- 終了判定 (ライブラリと同じく、動かす前にセンサを見る) ----
        ap = self.phase == APPROACH
        contact = ap & ((d1 > self.D_TH) | (d2 > self.D_TH))
        reach = ap & ~contact & (self.pos[:, 2] <= self.z_lim + 1e-6)
        tout = ap & ~contact & ~reach & (self.t >= self.APPROACH_TIMEOUT)
        self.stop[contact], self.stop[reach], self.stop[tout] = "contact", "z_lim", "timeout"
        to_pos = contact if self.POSITION else np.zeros(N, dtype=bool)
        to_close = (contact | reach | tout) & ~to_pos
        self.nail0[to_pos] = raw[to_pos]
        self.k_prev[to_pos] = 0

        po = self.phase == POSITION
        k = np.minimum((self.t / LIB_DT).astype(int), len(self.offsets) - 1)
        pos_done = po & ((np.abs(t2) > self.POS_T2_TH) | (d2 > self.POS_D_TH)
                         | (k >= len(self.offsets) - 1) | (self.t >= self.POS_MAX_T))
        to_close |= pos_done
        to_lift = (self.phase == CLOSE) & (self.t >= self.CLOSE_T)

        self.phase[to_pos], self.phase[to_close], self.phase[to_lift] = POSITION, CLOSE, LIFT
        self.t[to_pos | to_close | to_lift] = 0.0

        # ---- 動作 ----
        ap = self.phase == APPROACH
        v = self.APPROACH_V * smootherstep(self.t / self.RAMP_T)
        dp[ap, 2] = np.minimum(v[ap] * dt, np.maximum(self.pos[ap, 2] - self.z_lim[ap], 0.0) * 1e3)

        po = self.phase == POSITION
        k = np.minimum(((self.t + dt) / LIB_DT).astype(int), len(self.offsets) - 1)
        g = np.array([self.Z_GAIN, self.RX_GAIN, self.RY_GAIN])
        step = (self.offsets[k] - self.offsets[self.k_prev]) * g  # この制御周期ぶんの差分 (実機の 8ms ぶんを足し合わせたもの)
        dp[po, 2] = step[po, 0]
        dp[po, 0] = step[po, 0] / 10.0  # ライブラリの「急遽 X 軸方向の動きも追加」
        dr[po, 0], dr[po, 1] = step[po, 1], step[po, 2]
        self.k_prev[po] = k[po]
        self.move_tool(ap | po, dp, dr)

        li = self.phase == LIFT
        up = np.minimum(self.LIFT_V * dt, np.maximum(self.LIFT_H - self.lifted, 0.0))
        self.pos[li, 2] += up[li] * 1e-3
        self.lifted[li] += up[li]

        self.t += dt
        grip = np.where(self.phase >= CLOSE, -1.0, 1.0)
        q_goal = e.ik.solve_pose_batch(self.pos, mat_to_quat(self.R), e.q_cmd)
        act = np.zeros((N, e.act_dim))
        act[:, -1] = grip
        return act, q_goal


def add_args(parser):
    C = FFController
    parser.add_argument("--ff-approach-v", type=float, default=C.APPROACH_V, help="高速アプローチの速さ [mm/s]")
    parser.add_argument("--ff-d-th", type=float, default=C.D_TH, help="アプローチを止める押し込み [mm]")
    parser.add_argument("--ff-no-position", action="store_true", help="位置決めを飛ばす (アプローチ → すぐ閉じる)")
    parser.add_argument("--ff-gains", type=float, nargs=3, default=(C.Z_GAIN, C.RX_GAIN, C.RY_GAIN), help="位置決めの z_gain rx_gain ry_gain")
    parser.add_argument("--ff-upsample", type=int, default=C.UPSAMPLE, help="位置決めの記録軌道のアップサンプル倍率 (大きいほどゆっくり)")
    parser.add_argument("--ff-pos-th", type=float, nargs=2, default=(C.POS_T2_TH, C.POS_D_TH), help="位置決めを止める右爪の t2 [deg] と d [mm]")
    parser.add_argument("--ff-pos-time", type=float, default=C.POS_MAX_T, help="位置決めの打ち切り [s]")


def apply_args(args):
    C = FFController
    C.APPROACH_V, C.D_TH, C.POSITION = args.ff_approach_v, args.ff_d_th, not args.ff_no_position
    C.Z_GAIN, C.RX_GAIN, C.RY_GAIN = args.ff_gains
    C.UPSAMPLE, C.POS_MAX_T = args.ff_upsample, args.ff_pos_time
    C.POS_T2_TH, C.POS_D_TH = args.ff_pos_th


if __name__ == "__main__":
    import newton.examples

    from tma import grasp_pile_vec as gv

    parser = newton.examples.create_parser()
    parser.add_argument("--urdf", default="tma.urdf")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--print-every", type=int, default=0, help="この step ごとに環境 0 の状態を表示 (0 で表示しない)")
    add_args(parser)
    parser.set_defaults(num_frames=10**9)
    viewer, args = newton.examples.init(parser)
    apply_args(args)
    env = gv.VecGraspEnv(viewer, args, seed=args.seed)
    if hasattr(viewer, "hide_loading_splash"):
        viewer.hide_loading_splash()
    ctrl = FFController(env)
    ctrl.reset(np.arange(env.num_envs))
    stats, finished = {}, 0
    stops = {}
    while finished < args.episodes and viewer.is_running():
        act, q_goal = ctrl.act()
        _, _, done, info = env.step_env(act, q_goal)
        env.render()
        if args.print_every and (done[0] or env.disp["t"][0] % args.print_every == 0):
            tip_err = (env.tip_mid(env.read()[2])[0] - ctrl.pos[0]) * 1e3 if not done[0] else np.zeros(3)
            print(f"{PHASE_NAMES[ctrl.phase[0]]:8s} stop={ctrl.stop[0] or '-':7s} tip_err={np.round(tip_err, 1)}mm "
                  + env.nail_text(0).replace("env0 ", ""))
            if done[0]:
                print("=> " + next((k for k, _ in env.OUTCOMES if info[k][0]), "blown"))
        for w in np.nonzero(done)[0]:
            s = stats.setdefault(info["type"][w], [0] * (1 + len(env.OUTCOMES)))
            s[0] += 1
            for j, (k, _) in enumerate(env.OUTCOMES):
                s[1 + j] += bool(info[k][w])
            res = next((k for k, _ in env.OUTCOMES if info[k][w]), "blown")
            stops.setdefault(ctrl.stop[w], {}).setdefault(res, 0)
            stops[ctrl.stop[w]][res] += 1
            finished += 1
        ctrl.stop[done] = ""
        ctrl.reset(np.nonzero(done)[0])
    tot = np.sum(list(stats.values()), axis=0)
    print(f"  {'種類':8s} 試行 " + " ".join(name for _, name in env.OUTCOMES))
    for name in sorted(stats):
        print(f"  {name:8s} " + " ".join(f"{v:4d}" for v in stats[name]))
    print(f"  {'合計':8s} " + " ".join(f"{v:4d}" for v in tot) + f"   成功率 {tot[1] / tot[0]:.2f}")
    print("  アプローチの止まり方ごとの結果:", {k or "-": v for k, v in stops.items()})
    viewer.close()
