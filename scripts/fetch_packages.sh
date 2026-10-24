#!/bin/sh
# Fetch the public mesh packages that are not committed (see .gitignore). Only meshes/ is kept.
set -e
cd "$(dirname "$0")/../packages"
fetch() {
  [ -d "$1" ] && return
  git clone --depth 1 "$2" "$1.tmp"
  mkdir "$1" && mv "$1.tmp/meshes" "$1/" && rm -rf "$1.tmp"
}
fetch ur_description https://github.com/UniversalRobots/Universal_Robots_ROS2_Description.git
fetch robotiq_hande_description https://github.com/macmacal/robotiq_hande_description.git
# COBOTTA (cobotta.urdf)。メッシュはリポジトリの denso_cobotta_descriptions/cobotta_description にある .dae
if [ ! -d denso_cobotta_descriptions ]; then
  git clone --depth 1 https://github.com/DENSORobot/denso_cobotta_ros.git denso_cobotta.tmp
  mkdir -p denso_cobotta_descriptions/cobotta_description
  mv denso_cobotta.tmp/denso_cobotta_descriptions/cobotta_description/*.dae denso_cobotta_descriptions/cobotta_description/
  rm -rf denso_cobotta.tmp
fi
