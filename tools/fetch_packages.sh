#!/bin/sh
# Fetch the public mesh packages that are not committed (see .gitignore).
set -e
cd "$(dirname "$0")/../packages"
[ -d ur_description ] || git clone --depth 1 https://github.com/UniversalRobots/Universal_Robots_ROS2_Description.git ur_description
[ -d robotiq_hande_description ] || git clone --depth 1 https://github.com/macmacal/robotiq_hande_description.git robotiq_hande_description
rm -rf ur_description/.git robotiq_hande_description/.git
