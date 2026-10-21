"""VecGraspEnv の環境数ごとの速度と GPU メモリを測る (ランダム行動で数十ステップ。リセットの頻度が高い最悪側の条件)。

    uv run scripts/bench_envs.py --viewer null --num-envs 256
"""
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # リポジトリ直下 (tma パッケージ)

import numpy as np
import warp as wp

import newton.examples
from tma.envs import grasp_pile_vec as gv

if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--urdf", default="tma.urdf")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=60)
    parser.set_defaults(num_frames=10**9)
    viewer, args = newton.examples.init(parser)
    t0 = time.time()
    env = gv.VecGraspEnv(viewer, args, seed=0, scripted=False)
    t_build = time.time() - t0
    rng = np.random.default_rng(0)
    for _ in range(5):
        env.step_env(rng.normal(0, 0.3, (env.num_envs, env.act_dim)))
    t0 = time.time()
    for _ in range(args.steps):
        env.step_env(rng.normal(0, 0.3, (env.num_envs, env.act_dim)))
    dt = time.time() - t0
    wp.synchronize()
    used = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
    print("BENCH " + json.dumps(dict(num_envs=env.num_envs, steps_per_s=round(env.num_envs * args.steps / dt, 1), build_s=round(t_build, 1), gpu_mem_mib=int(used.splitlines()[0]))), flush=True)
    viewer.close()
