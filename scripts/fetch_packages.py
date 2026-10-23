"""コミットしていない公開メッシュパッケージを取ってくる (.gitignore 参照)。meshes/ だけ残す。fetch_packages.sh の Python 版。

    uv run scripts/fetch_packages.py
"""
import shutil
import subprocess
from pathlib import Path

PACKAGES = Path(__file__).resolve().parents[1] / "packages"
REPOS = {
    "ur_description": "https://github.com/UniversalRobots/Universal_Robots_ROS2_Description.git",
    "robotiq_hande_description": "https://github.com/macmacal/robotiq_hande_description.git",
}


def rmtree(path):
    def force(func, p, _):  # git の pack ファイルは読み取り専用で、Windows ではそのままだと消せない
        Path(p).chmod(0o700)
        func(p)

    shutil.rmtree(path, onexc=force)


for name, url in REPOS.items():
    dst = PACKAGES / name
    if dst.is_dir():
        print(f"{name}: あるので飛ばす")
        continue
    tmp = PACKAGES / f"{name}.tmp"
    if tmp.exists():
        rmtree(tmp)
    subprocess.run(["git", "clone", "--depth", "1", url, str(tmp)], check=True)
    dst.mkdir(parents=True)
    shutil.move(tmp / "meshes", dst / "meshes")
    rmtree(tmp)
    print(f"{name}: 取得")
