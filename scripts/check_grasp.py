"""手順1: 箱を置いて Hande を閉じ、爪の変位と箱の位置を表示する。

    uv run scripts/check_grasp.py --viewer null      # 数値だけ
    uv run scripts/check_grasp.py --viewer viser     # ブラウザで見る (http://localhost:8080)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # リポジトリ直下 (tma パッケージ)

import newton
import newton.examples
from tma import sim

OPEN_MM = 25.0
CLOSE_MM = 0.0
OPEN_STEPS = 60  # 開いて落ち着かせる
CLOSE_STEPS = 120


class GraspTest(sim.Sim):
    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        self.frame = 0
        self.set_gripper(OPEN_MM)

    def step(self):
        if self.frame == OPEN_STEPS:
            self.set_gripper(CLOSE_MM)
        super().step()
        self.frame += 1
        if self.frame % 10 == 0:
            n = self.nail_state()
            box = self.box_pos()
            print(
                f"{self.frame:4d} grip={self.gripper_opening_mm():5.1f}mm "
                f"box=({box[0]:.3f},{box[1]:.3f},{box[2]:.3f}) "
                f"pushL={n['left_z']:+.2f} pushR={n['right_z']:+.2f}mm (目標 {sim.NAIL_PUSH_REF_MM}) "
                f"xL={n['left_x']:+.2f} yL={n['left_y']:+.2f} xR={n['right_x']:+.2f} yR={n['right_y']:+.2f}deg"
            )


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--urdf", default="tma.urdf")
    parser.add_argument("--box", action="store_true", default=True)
    parser.set_defaults(num_frames=OPEN_STEPS + CLOSE_STEPS)
    viewer, args = newton.examples.init(parser)
    newton.examples.run(GraspTest(viewer, args), args)
