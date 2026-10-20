"""VecGraspEnv を PPO で学習する (自前の最小実装、torch)。

    uv run tools/train_ppo.py --viewer null --num-envs 64 --iters 500
    uv run tools/train_ppo.py --viewer null --eval ckpt/ppo_latest.pt --episodes 200   # 決定的方策で評価

- 観測は実行時平均・分散で正規化、報酬は割引リターンの標準偏差でスケーリング
- 時間切れ (timeout) は終端扱いにせず、リセット前の最終観測の V でブートストラップする
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn as nn

import newton.examples
import grasp_env as ge
import grasp_vec as gv

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RunningMeanStd:
    def __init__(self, shape):
        self.mean, self.var, self.count = np.zeros(shape), np.ones(shape), 1e-4

    def update(self, x):
        n = x.shape[0]
        bm, bv = x.mean(axis=0), x.var(axis=0)
        d, tot = bm - self.mean, self.count + n
        self.mean = self.mean + d * n / tot
        self.var = (self.var * self.count + bv * n + d**2 * self.count * n / tot) / tot
        self.count = tot


def mlp(i, o, hidden=256):
    return nn.Sequential(nn.Linear(i, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, o))


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.pi, self.v = mlp(obs_dim, act_dim), mlp(obs_dim, 1)
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

    def dist(self, obs):
        return torch.distributions.Normal(self.pi(obs), self.log_std.exp())

    def value(self, obs):
        return self.v(obs).squeeze(-1)


def normalize(rms, obs):
    return torch.as_tensor(np.clip((obs - rms.mean) / np.sqrt(rms.var + 1e-8), -10, 10), dtype=torch.float32, device=DEV)


def save(path, net, obs_rms, ret_rms):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(net=net.state_dict(), obs=(obs_rms.mean, obs_rms.var, obs_rms.count), ret=(ret_rms.var, ret_rms.count)), path)


def load(path, net, obs_rms, ret_rms):
    ck = torch.load(path, map_location=DEV, weights_only=False)
    net.load_state_dict(ck["net"])
    obs_rms.mean, obs_rms.var, obs_rms.count = ck["obs"]
    ret_rms.var, ret_rms.count = ck["ret"]


def evaluate(env, net, obs_rms, episodes, noise=0.0):
    stats, finished = {}, 0
    obs = env.obs
    if hasattr(env.viewer, "hide_loading_splash"):
        env.viewer.hide_loading_splash()
    while finished < episodes and env.viewer.is_running():
        with torch.no_grad():
            act = net.dist(normalize(obs_rms, obs)).mean.cpu().numpy()
        act = act + noise * np.random.randn(*act.shape)
        obs, _, done, info = env.step_env(act)
        env.render()  # null ビューアでは何もしない
        for w in np.nonzero(done)[0]:
            s = stats.setdefault(info["type"][w], [0, 0, 0, 0, 0])
            s[0] += 1
            s[1:] = [s[1] + bool(info["success"][w]), s[2] + bool(info["dropped"][w]), s[3] + bool(info["missed"][w]), s[4] + bool(info["timeout"][w])]
            finished += 1
    if not stats:  # 1 エピソードも終わらないうちにビューアを閉じた
        return
    tot = np.sum(list(stats.values()), axis=0)
    print(f"  {'形状':8s} 試行 成功 落下 空振り 時間切れ")
    for name in sorted(stats):
        print(f"  {name:8s} " + " ".join(f"{v:4d}" for v in stats[name]))
    print(f"  {'合計':8s} " + " ".join(f"{v:4d}" for v in tot) + f"   成功率 {tot[1] / tot[0]:.2f}")


def collect_labels(env, net, obs_rms, steps, noise=0.0):
    """steps ステップ進め、各状態でのスクリプト方策の行動をラベルとして集める。
    net が None ならスクリプト方策で動かし、あれば net の平均行動で動かす (DAgger)。
    実行する行動には環境ごとに 0〜noise の std のノイズを足し、ずれた状態から戻すラベルも集める (DART)"""
    X, Y = [], []
    succ = fin = 0
    std = np.random.uniform(0.0, noise, (env.num_envs, 1))
    for _ in range(steps):
        label = env.scripted_action()
        X.append(env.obs.copy())
        Y.append(label)
        if net is None:
            act = label
        else:
            with torch.no_grad():
                act = net.dist(normalize(obs_rms, env.obs)).mean.cpu().numpy()
        _, _, done, info = env.step_env(act + std * np.random.randn(*act.shape))
        fin += done.sum()
        succ += info["success"].sum()
    print(f"  {'スクリプト' if net is None else '方策'}で {steps} ステップ (ノイズ ≤{noise}): 終了 {fin} 成功 {succ}", flush=True)
    return np.concatenate(X).astype(np.float32), np.concatenate(Y).astype(np.float32)


def behavior_clone(net, obs_rms, X, Y, epochs, lr=1e-3, batch=256):
    Xt, Yt = normalize(obs_rms, X), torch.as_tensor(Y, device=DEV)
    opt = torch.optim.Adam(net.pi.parameters(), lr=lr)
    for ep in range(epochs):
        perm, tot = torch.randperm(len(Xt), device=DEV), 0.0
        for mb in perm.split(batch):
            loss = (net.pi(Xt[mb]) - Yt[mb]).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(mb)
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"  BC epoch {ep:3d} loss {tot / len(Xt):.4f}", flush=True)


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--urdf", default="thinker-TMA-F1-REFX4-U3.urdf")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--horizon", type=int, default=32, help="1 イテレーションで各環境を進めるステップ数")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--minibatches", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--ent", type=float, default=0.0)
    parser.add_argument("--bc-steps", type=int, default=300, help="スクリプト方策で集めるステップ数 (模倣学習の初期データ)。0 で無効")
    parser.add_argument("--dagger-rounds", type=int, default=10, help="方策で動かしてスクリプト方策でラベルを付け直す回数")
    parser.add_argument("--dart-noise", type=float, default=0.15, help="デモ・DAgger 収集時に実行行動へ足すノイズ std の上限")
    parser.add_argument("--phase-tol", type=float, default=0.01, help="スクリプト方策の到達判定の許容誤差 [rad] (方策が模倣しやすいよう緩める)")
    parser.add_argument("--dagger-steps", type=int, default=160)
    parser.add_argument("--bc-epochs", type=int, default=30)
    parser.add_argument("--critic-warmup", type=int, default=10, help="BC 後、このイテレーション数は actor を更新せず critic だけ学習")
    parser.add_argument("--init-log-std", type=float, default=-2.5, help="BC 後の探索 std (log)。大きいと模倣した把持の精度が崩れる")
    parser.add_argument("--save-dir", default="ckpt")
    parser.add_argument("--resume", default=None, help="続きから学習するチェックポイント")
    parser.add_argument("--eval", default=None, help="このチェックポイントを決定的方策で評価して終わる")
    parser.add_argument("--eval-noise", type=float, default=0.0, help="--eval で行動に足すガウスノイズの std (探索ノイズへの感度を見る)")
    parser.add_argument("--episodes", type=int, default=200, help="--eval のエピソード数")
    parser.add_argument("--pos-noise-mm", type=float, default=ge.EST_POS_STD * 1e3)
    parser.add_argument("--yaw-noise-deg", type=float, default=float(np.degrees(ge.EST_YAW_STD)))
    parser.set_defaults(num_frames=10**9)
    viewer, args = newton.examples.init(parser)
    ge.EST_POS_STD = args.pos_noise_mm * 1e-3
    ge.EST_YAW_STD = np.radians(args.yaw_noise_deg)
    torch.manual_seed(args.seed)

    use_bc = args.bc_steps > 0 and not (args.resume or args.eval)
    gv.VecGraspEnv.PHASE_TOL = args.phase_tol
    env = gv.VecGraspEnv(viewer, args, seed=args.seed, scripted=use_bc)
    N, T = env.num_envs, args.horizon
    print(f"num_envs={N} obs_dim={env.obs_dim} act_dim={env.act_dim} device={DEV}")
    net = ActorCritic(env.obs_dim, env.act_dim).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)
    obs_rms, ret_rms = RunningMeanStd(env.obs_dim), RunningMeanStd(())
    if args.resume or args.eval:
        load(args.resume or args.eval, net, obs_rms, ret_rms)

    if args.eval:
        evaluate(env, net, obs_rms, args.episodes, args.eval_noise)
        viewer.close()
        sys.exit()

    warmup = 0
    if use_bc:
        X, Y = collect_labels(env, None, obs_rms, args.bc_steps, args.dart_noise)
        obs_rms.update(X)
        obs_rms.var = np.maximum(obs_rms.var, 1e-3)  # ほぼ動かない次元の正規化で増幅しすぎない
        behavior_clone(net, obs_rms, X, Y, args.bc_epochs)
        for r in range(args.dagger_rounds):
            x, y = collect_labels(env, net, obs_rms, args.dagger_steps, args.dart_noise)
            X, Y = np.concatenate([X, x]), np.concatenate([Y, y])
            behavior_clone(net, obs_rms, X, Y, args.bc_epochs)
        net.log_std.data.fill_(args.init_log_std)
        env.scripted = False  # 以降は学習中の方策で動かす (リセット時の IK 2 回分を省く)
        env.reset_envs(np.arange(N))
        env.obs = env.observe()
        save(Path(args.save_dir) / "bc.pt", net, obs_rms, ret_rms)
        warmup = args.critic_warmup
    elif args.resume:
        warmup = args.critic_warmup  # BC の重みから再開するときも、critic が育つまで actor を動かさない

    ret_run = np.zeros(N)  # 報酬スケーリング用の割引リターン
    recent = []  # 直近エピソードの (成功, 落下, 空振り, 時間切れ, 報酬和)
    ep_ret = np.zeros(N)
    t_start, total_steps = time.time(), 0
    for it in range(1, args.iters + 1):
        obs_buf = np.zeros((T, N, env.obs_dim), dtype=np.float32)
        act_buf = torch.zeros((T, N, env.act_dim), device=DEV)
        logp_buf, val_buf = torch.zeros((T, N), device=DEV), torch.zeros((T, N), device=DEV)
        rew_buf, term_buf = np.zeros((T, N)), np.zeros((T, N))
        obs = env.obs
        for t in range(T):
            obs_buf[t] = obs
            o = normalize(obs_rms, obs)
            with torch.no_grad():
                d = net.dist(o)
                a = d.sample()
                act_buf[t], logp_buf[t], val_buf[t] = a, d.log_prob(a).sum(-1), net.value(o)
            obs, rew, done, info = env.step_env(a.cpu().numpy())
            ep_ret += rew
            if info["timeout"].any():  # 時間切れは終端にせず V でブートストラップ
                with torch.no_grad():
                    vf = net.value(normalize(obs_rms, info["final_obs"])).cpu().numpy()
                rew = rew + args.gamma * vf * info["timeout"]
            ret_run = ret_run * args.gamma + rew
            ret_rms.update(ret_run[None].T)
            ret_run[done] = 0.0
            rew_buf[t], term_buf[t] = rew, done
            for w in np.nonzero(done)[0]:
                recent.append((info["success"][w], info["dropped"][w], info["missed"][w], info["timeout"][w], ep_ret[w]))
            ep_ret[done] = 0.0
        total_steps += T * N
        if not (use_bc or args.resume):  # 模倣・再開から始めたときは正規化を固定する (入力スケールがずれると模倣した方策が崩れる)
            obs_rms.update(obs_buf.reshape(-1, env.obs_dim))

        # GAE
        rew_t = torch.as_tensor(rew_buf / np.sqrt(ret_rms.var + 1e-8), dtype=torch.float32, device=DEV)
        nonterm = torch.as_tensor(1.0 - term_buf, dtype=torch.float32, device=DEV)
        with torch.no_grad():
            last_v = net.value(normalize(obs_rms, obs))
        adv, last = torch.zeros((T, N), device=DEV), torch.zeros(N, device=DEV)
        for t in reversed(range(T)):
            nv = last_v if t == T - 1 else val_buf[t + 1]
            delta = rew_t[t] + args.gamma * nv * nonterm[t] - val_buf[t]
            last = delta + args.gamma * args.lam * nonterm[t] * last
            adv[t] = last
        ret = adv + val_buf

        # 更新 (obs は更新後の統計で正規化し直す)
        b_obs = normalize(obs_rms, obs_buf.reshape(-1, env.obs_dim))
        b_act, b_logp, b_adv, b_ret = act_buf.reshape(-1, env.act_dim), logp_buf.reshape(-1), adv.reshape(-1), ret.reshape(-1)
        B = T * N
        for _ in range(args.epochs):
            perm = torch.randperm(B, device=DEV)
            for mb in perm.chunk(args.minibatches):
                d = net.dist(b_obs[mb])
                logp = d.log_prob(b_act[mb]).sum(-1)
                ratio = (logp - b_logp[mb]).exp()
                a_n = (b_adv[mb] - b_adv[mb].mean()) / (b_adv[mb].std() + 1e-8)
                pg = -torch.min(ratio * a_n, ratio.clamp(1 - args.clip, 1 + args.clip) * a_n).mean()
                vl = 0.5 * (net.value(b_obs[mb]) - b_ret[mb]).pow(2).mean()
                pg = pg * (it > warmup)  # ウォームアップ中は critic だけ更新
                loss = pg + vl - args.ent * d.entropy().sum(-1).mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()

        if it % 5 == 0 or it == 1:
            r = np.array(recent[-200:], dtype=float) if recent else np.zeros((1, 5))
            sps = total_steps / (time.time() - t_start)
            print(
                f"it {it:4d} steps {total_steps:8d} | 成功 {r[:, 0].mean():.2f} 落下 {r[:, 1].mean():.2f} 空振り {r[:, 2].mean():.2f} "
                f"時間切れ {r[:, 3].mean():.2f} 報酬和 {r[:, 4].mean():6.1f} | std {net.log_std.exp().mean().item():.2f} | {sps:.0f} step/s",
                flush=True,
            )
        if it % 25 == 0 or it == args.iters:
            save(Path(args.save_dir) / "ppo_latest.pt", net, obs_rms, ret_rms)
    viewer.close()
