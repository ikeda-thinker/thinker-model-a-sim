# TMA sim

Thinker Model A (THF + 爪, Robotiq Hand-E, RealSense D405, UR3e) の URDF と、Newton Physics での把持・ピン挿入のシミュレーション・学習。
URDF は ROS2 の think_hand_f_description の `robots/thinker-model-a` を xacro で展開したものを元に、手で直している。

## 構成

```
tma.urdf          ロボットの URDF
tma/              シミュレータ本体と環境 (import して使う)
  sim.py          URDF を Newton に読み込んでアーム・Hand-E・THF を動かす土台 (単体でも実行できる)
  envs/
    grasp.py           把持の RL 環境 (1 環境・ワーク 1 個)、アームの IK
    grasp_pile.py      ばら積み版 (1 環境)
    grasp_pile_vec.py  ばら積み版の並列版 (学習に使う)
    peg.py             ピン挿入の RL 環境 (並列版)
scripts/          実行するもの
  view.py         URDF をブラウザで見る
  train_ppo.py    PPO で学習・評価
  bench_envs.py   環境数ごとの速度・GPU メモリ
  check_grasp.py  箱を置いて Hand-E を閉じる確認
  check_press.py  台を爪先に押し当てる確認
tools/            メッシュの取得・作成
packages/         メッシュ (package:// の解決先)
ckpt/             学習のチェックポイント (gitignore)
```

各ファイルの先頭の docstring に実行例がある (`uv run tma/envs/peg.py ...`、`uv run scripts/train_ppo.py ...` など)。

## セットアップ

```sh
sh tools/fetch_packages.sh   # UR3e / Hand-E の公式メッシュを packages/ に取得
uv run scripts/view.py       # http://localhost:8080 を開く
```

別の URDF を見る場合は `uv run scripts/view.py <file.urdf>`。

## メッシュ

`package://<name>/...` は `packages/<name>/...` として解決される。

| パッケージ | 状態 |
|---|---|
| `ur_description` | 公式。`fetch_packages.sh` で取得 (gitignore) |
| `robotiq_hande_description` | 非公式 (macmacal)。`fetch_packages.sh` で取得 (gitignore) |
| `realsense2_description` | realsense-ros 由来の D405 のみコミット済み |
| `think_hand_f_description` | 爪 (`pile-nail`)・THF (`thf`)・Hand-E とのアダプタ (`hande-adapter`) は CAD から変換。残りは**仮の箱型メッシュ (STL)** で形状は推定。本物の `.dae` を同じパスに置けば優先される |

CAD のメッシュは `packages/*.stl` (Fusion 360 から書き出した mm 単位) から `tools/convert_meshes.py` で、仮メッシュは `tools/make_placeholders.py` で再生成できる。
