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
