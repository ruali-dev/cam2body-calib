"""交互式图像点选标注工具。

操作：
    鼠标左键    添加点
    鼠标右键    撤销
    滚轮        缩放
    s           保存到 YAML
    q / ESC     退出
"""

from pathlib import Path

import cv2
import numpy as np


class PointAnnotator:
    """交互式角点标注器。"""

    def __init__(self, image, output_path=None):
        self._img = image.copy()
        self._h, self._w = image.shape[:2]
        self._output = output_path
        self._points = []  # [(u, v), ...]
        self._scale = 1.0
        self._ox, self._oy = 0, 0
        self._mx, self._my = 0, 0
        self._saved = False  # 防止重复保存

    def run(self):
        cv2.namedWindow("annotator", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("annotator", self._mouse)
        cv2.resizeWindow("annotator", self._w, self._h)

        # 使用 waitKeyEx + 时间间隔双重防护
        last_action = {"key": -1, "time": 0}
        import time as _time

        while True:
            canvas = self._render()
            cv2.imshow("annotator", canvas)
            raw = cv2.waitKeyEx(50)
            key = raw & 0xFF

            # 按键释放（无按键）时重置
            if raw == -1:
                last_action["key"] = -1
                continue

            # 同键 300ms 内不重复触发
            now = _time.time()
            if key == last_action["key"] and (now - last_action["time"]) < 0.3:
                continue
            last_action["key"] = key
            last_action["time"] = now

            if key in (ord("q"), 27):
                break
            elif key == ord("s"):
                if not self._saved:
                    self._save()
                    break  # 保存后自动退出
            elif key == ord("c"):
                self._points.clear()
                self._saved = False
                print("[c] 已清除")
            elif key == ord("z"):
                if self._points:
                    p = self._points.pop()
                    self._saved = False
                    print(f"[z] 撤销 ({p[0]:.0f},{p[1]:.0f})")
            elif key == ord("r"):
                self._scale = 1.0
                self._ox = self._oy = 0
            elif key in (ord("+"), ord("=")):
                self._scale = min(10, self._scale * 1.2)
            elif key == ord("-"):
                self._scale = max(0.1, self._scale / 1.2)

        cv2.destroyAllWindows()
        return self._points.copy()

    def _mouse(self, event, x, y, flags, param):
        self._mx, self._my = x, y

        if event == cv2.EVENT_LBUTTONDOWN:
            u = (x - self._ox) / self._scale
            v = (y - self._oy) / self._scale
            if 0 <= u < self._w and 0 <= v < self._h:
                self._points.append((u, v))
                self._saved = False  # 新加点后允许重新保存
                print(f"[{len(self._points)}] u={u:.1f} v={v:.1f}")

        elif event == cv2.EVENT_RBUTTONDOWN:
            if self._points:
                p = self._points.pop()
                self._saved = False
                print(f"[撤销] ({p[0]:.0f},{p[1]:.0f})")

        elif event == cv2.EVENT_MOUSEWHEEL:
            old = self._scale
            self._scale = min(10, max(0.1, self._scale * (1.15 if flags > 0 else 0.87)))
            r = self._scale / old
            self._ox = int(x - r * (x - self._ox))
            self._oy = int(y - r * (y - self._oy))

    def _render(self):
        sw, sh = int(self._w * self._scale), int(self._h * self._scale)
        img = cv2.resize(self._img, (sw, sh)) if self._scale != 1 else self._img
        canvas = np.zeros((sh + 40, sw, 3), dtype=np.uint8)
        canvas[:sh, :sw] = img

        for i, (u, v) in enumerate(self._points):
            sx, sy = int(u * self._scale), int(v * self._scale)
            cv2.circle(canvas, (sx, sy), 5, (0, 0, 255), -1)
            cv2.putText(canvas, str(i + 1), (sx + 8, sy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        bar = sh
        cv2.rectangle(canvas, (0, bar), (sw, bar + 40), (40, 40, 40), -1)
        saved_text = "[SAVED]" if self._saved else ""
        info = (f"Points: {len(self._points)} {saved_text} | "
                f"Zoom: {self._scale:.1f}x | S:save+exit C:clear Q:quit")
        cv2.putText(canvas, info, (10, bar + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        M = np.float32([[1, 0, self._ox], [0, 1, self._oy]])
        return cv2.warpAffine(canvas, M, (sw, sh + 40),
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(40, 40, 40))

    def _save(self):
        if not self._points:
            print("[无点可保存]")
            return
        path = Path(self._output or "outputs/annotations.yaml")
        path.parent.mkdir(parents=True, exist_ok=True)
        import yaml
        yaml.dump({
            "image_points": [{"u": float(u), "v": float(v)} for u, v in self._points],
            "num_points": len(self._points),
        }, open(path, "w"), default_flow_style=False, allow_unicode=True, sort_keys=False)
        self._saved = True
        print(f"\n[OK] 已保存 {len(self._points)} 点 -> {path}")
        print(f"{'='*55}")
        print(f"{'点':<5} {'u':<10} {'v':<10} {'x_body':<10} {'y_body':<10} {'z_body':<10}")
        print(f"{'-'*55}")
        for i, (u, v) in enumerate(self._points):
            print(f"{i+1:<5} {u:<10.1f} {v:<10.1f} {'?':<10} {'?':<10} {'?':<10}")
        print(f"{'='*55}")


def run_annotator(image_path, camera_config=None, output_path=None, fisheye_balance=0.0):
    from ..io.image_io import read_image
    img = read_image(image_path)
    if camera_config:
        from ..config.load import load_camera
        cam = load_camera(camera_config)
        if np.any(cam.D != 0):
            print(f"去畸变: {cam.model}, balance={fisheye_balance}")
            img, cam_r = cam.undistort_image(img, balance=fisheye_balance)
            kw, kh = cam_r.K[0, 0], cam_r.K[1, 1]
            print(f"  完成: {img.shape[1]}x{img.shape[0]}, K=({kw:.1f},{kh:.1f})")
    return PointAnnotator(img, output_path).run()
