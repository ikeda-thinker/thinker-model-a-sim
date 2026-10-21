# thinker-TMA-F1-REFX4-U3

Thinker Model A (F1 finger, Robotiq Hand-E, RealSense, UR3e) の URDF と、ブラウザで確認するための簡易ビューア。

## セットアップ

```sh
sh tools/fetch_packages.sh   # UR3e / Hand-E の公式メッシュを packages/ に取得
uv run tools/view.py         # http://localhost:8080 を開く
```

別の URDF を見る場合は `uv run tools/view.py <file.urdf>`。

## メッシュ

`package://<name>/...` は `packages/<name>/...` として解決される。

| パッケージ | 状態 |
|---|---|
| `ur_description` | 公式。`fetch_packages.sh` で取得 (gitignore) |
| `robotiq_hande_description` | 非公式 (macmacal)。`fetch_packages.sh` で取得 (gitignore) |
| `realsense2_description` | realsense-ros 由来の D405 のみコミット済み |
| `thinker_ros2_description` | 爪 (`pile-nail`)・THF (`thf`)・Hand-E とのアダプタ (`hande-adapter`) は CAD から変換。残りは**仮の箱型メッシュ (STL)** で形状は推定。本物の `.dae` を同じパスに置けば優先される |

CAD のメッシュは `packages/*.stl` (Fusion 360 から書き出した mm 単位) から `tools/convert_meshes.py` で、仮メッシュは `tools/make_placeholders.py` で再生成できる。
