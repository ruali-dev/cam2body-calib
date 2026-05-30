"""交互式手动打码工具。

鼠标拖拽框选区域，高斯模糊。可叠加多个区域。

用法:
    uv run python scripts/interactive_blur.py <image_path> [--output <path>]

操作:
    鼠标左键拖拽    框选打码区域
    鼠标右键         撤销上一个区域
    S                保存并退出
    Q / ESC          退出（不保存）
    C                清除所有区域
    R                重置视图
    滚轮             缩放
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


class InteractiveBlur:
    def __init__(self, image: np.ndarray, output_path: str = "outputs/blurred.png"):
        self._original = image.copy()
        self._img = image.copy()
        self._h, self._w = image.shape[:2]
        self._output = output_path
        self._regions = []  # [(x, y, w, h), ...]
        self._drawing = False
        self._start = (0, 0)
        self._current = (0, 0)
        self._scale = 1.0
        self._ox, self._oy = 0, 0

    def run(self):
        cv2.namedWindow("blur", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("blur", self._mouse)
        cv2.resizeWindow("blur", self._w, self._h)

        # 去抖
        last_key_time = 0
        import time

        while True:
            canvas = self._render()
            cv2.imshow("blur", canvas)
            key = cv2.waitKey(30) & 0xFF

            if key == 255:
                continue
            now = time.time()
            if now - last_key_time < 0.3:
                continue
            last_key_time = now

            if key in (ord("q"), 27):
                break
            elif key == ord("s"):
                self._apply_and_save()
                break
            elif key == ord("c"):
                self._regions.clear()
                self._img = self._original.copy()
                print("[c] 已清除")
            elif key == ord("r"):
                self._scale = 1.0
                self._ox = self._oy = 0
            elif key in (ord("+"), ord("=")):
                self._scale = min(10, self._scale * 1.2)
            elif key == ord("-"):
                self._scale = max(0.1, self._scale / 1.2)
            elif key == ord("z"):
                if self._regions:
                    self._regions.pop()
                    self._rebuild()
                    print(f"[z] 撤销 (剩余 {len(self._regions)} 区域)")

        cv2.destroyAllWindows()

    def _mouse(self, event, x, y, flags, param):
        ix = int((x - self._ox) / self._scale)
        iy = int((y - self._oy) / self._scale)

        if event == cv2.EVENT_LBUTTONDOWN:
            self._drawing = True
            self._start = (max(0, min(ix, self._w - 1)), max(0, min(iy, self._h - 1)))

        elif event == cv2.EVENT_MOUSEMOVE:
            self._current = (ix, iy)

        elif event == cv2.EVENT_LBUTTONUP and self._drawing:
            self._drawing = False
            ex = max(0, min(ix, self._w - 1))
            ey = max(0, min(iy, self._h - 1))
            sx, sy = self._start
            x1, x2 = min(sx, ex), max(sx, ex)
            y1, y2 = min(sy, ey), max(sy, ey)
            rw, rh = x2 - x1, y2 - y1
            if rw > 5 and rh > 5:
                self._regions.append((x1, y1, rw, rh))
                self._rebuild()
                print(f"[{len(self._regions)}] 区域: ({x1},{y1}) {rw}x{rh}")

        elif event == cv2.EVENT_RBUTTONDOWN:
            if self._regions:
                r = self._regions.pop()
                self._rebuild()
                print(f"[撤销] ({r[0]},{r[1]}) {r[2]}x{r[3]}")

        elif event == cv2.EVENT_MOUSEWHEEL:
            old = self._scale
            self._scale = min(10, max(0.1, self._scale * (1.15 if flags > 0 else 0.87)))
            ratio = self._scale / old
            self._ox = int(x - ratio * (x - self._ox))
            self._oy = int(y - ratio * (y - self._oy))

    def _rebuild(self):
        """重建打码图像。"""
        self._img = self._original.copy()
        for (rx, ry, rw, rh) in self._regions:
            roi = self._img[ry:ry + rh, rx:rx + rw]
            k = max(3, min(rw, rh) // 6)
            if k % 2 == 0:
                k += 1
            self._img[ry:ry + rh, rx:rx + rw] = cv2.GaussianBlur(roi, (k, k), 0)

    def _render(self):
        sw, sh = int(self._w * self._scale), int(self._h * self._scale)
        img = cv2.resize(self._img, (sw, sh)) if self._scale != 1 else self._img
        canvas = np.zeros((sh + 40, sw, 3), dtype=np.uint8)
        canvas[:sh, :sw] = img

        # 已保存的区域边框
        for (rx, ry, rw, rh) in self._regions:
            sx, sy = int(rx * self._scale), int(ry * self._scale)
            ex, ey = int((rx + rw) * self._scale), int((ry + rh) * self._scale)
            cv2.rectangle(canvas, (sx, sy), (ex, ey), (0, 255, 255), 2)

        # 正在拖拽的临时框
        if self._drawing:
            sx, sy = self._start
            ex, ey = self._current
            sx, sy = int(sx * self._scale), int(sy * self._scale)
            ex, ey = int(ex * self._scale), int(ey * self._scale)
            cv2.rectangle(canvas, (sx, sy), (ex, ey), (255, 255, 0), 1)

        # 状态栏
        bar = sh
        cv2.rectangle(canvas, (0, bar), (sw, bar + 40), (40, 40, 40), -1)
        info = f"Regions: {len(self._regions)} | Zoom: {self._scale:.1f}x | S:save Q:quit C:clear"
        cv2.putText(canvas, info, (10, bar + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        M = np.float32([[1, 0, self._ox], [0, 1, self._oy]])
        return cv2.warpAffine(canvas, M, (sw, sh + 40),
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(40, 40, 40))

    def _apply_and_save(self):
        path = Path(self._output)
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), self._img)
        print(f"[已保存] {len(self._regions)} 个打码区域 -> {path}")


def main():
    parser = argparse.ArgumentParser(description="交互式手动图像打码")
    parser.add_argument("image", help="输入图像路径")
    parser.add_argument("--output", "-o", default="outputs/blurred.png", help="输出路径")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        print(f"无法读取图像: {args.image}")
        sys.exit(1)

    print(f"图像: {img.shape[1]}x{img.shape[0]}")
    print("操作: 鼠标左键拖拽框选 | 右键撤销 | S 保存退出 | Q 退出")
    InteractiveBlur(img, args.output).run()


if __name__ == "__main__":
    main()
