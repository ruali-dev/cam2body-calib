"""批量图像脱敏打码脚本。

对指定目录中的图像进行人脸检测并打码。
支持 OpenCV Haar Cascade 和手动 ROI 两种模式。

用法:
    uv run python scripts/blur_faces.py <input_dir> <output_dir>
    uv run python scripts/blur_faces.py <input_dir> <output_dir> --roi configs/roi.yaml
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def detect_faces_haar(gray: np.ndarray) -> list[tuple[int, int, int, int]]:
    """使用 OpenCV Haar Cascade 检测人脸。返回 [(x, y, w, h), ...]."""
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if not Path(cascade_path).exists():
        print(f"  [WARN] Haar cascade not found: {cascade_path}")
        return []
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        print("  [WARN] Failed to load Haar cascade")
        return []
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]


def load_roi_config(roi_path: str) -> dict[str, list[tuple[int, int, int, int]]]:
    """从 YAML 加载手动 ROI 配置。

    格式:
        basename_0000.png:
          - [x, y, w, h]
    """
    import yaml
    with open(roi_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def blur_regions(img: np.ndarray, regions: list[tuple[int, int, int, int]]) -> np.ndarray:
    """对图像中指定区域进行高斯模糊。"""
    result = img.copy()
    h, w = img.shape[:2]
    for (rx, ry, rw, rh) in regions:
        rx, ry = max(0, rx), max(0, ry)
        rw, rh = min(rw, w - rx), min(rh, h - ry)
        if rw <= 0 or rh <= 0:
            continue
        roi = result[ry:ry + rh, rx:rx + rw]
        kernel = max(3, min(rw, rh) // 8)
        if kernel % 2 == 0:
            kernel += 1
        result[ry:ry + rh, rx:rx + rw] = cv2.GaussianBlur(roi, (kernel, kernel), 0)
    return result


def process_directory(
    input_dir: str,
    output_dir: str,
    roi_config: dict[str, list] | None = None,
    dry_run: bool = False,
):
    """批量处理目录中所有图像。"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    images = sorted([
        f for f in input_path.iterdir()
        if f.suffix.lower() in image_extensions
    ])

    if not images:
        print(f"目录 {input_dir} 中没有图像文件。")
        return

    total_faces = 0
    total_manual = 0
    processed = 0

    for img_file in images:
        img = cv2.imread(str(img_file))
        if img is None:
            print(f"  [SKIP] 无法读取: {img_file.name}")
            continue

        regions = []

        # 自动检测
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detect_faces_haar(gray)
        if faces:
            regions.extend(faces)
            total_faces += len(faces)

        # 手动 ROI
        if roi_config:
            key = img_file.name
            if key in roi_config:
                manual_rois = roi_config[key]
                regions.extend([tuple(r) for r in manual_rois])
                total_manual += len(manual_rois)

        if regions:
            img = blur_regions(img, regions)

        if not dry_run:
            out_file = output_path / img_file.name
            cv2.imwrite(str(out_file), img)

        status = f"{len(regions)} regions" if regions else "clean"
        print(f"  [{status}] {img_file.name}")
        processed += 1

    print(f"\n处理完成: {processed} 张图像")
    print(f"  自动检测人脸: {total_faces}")
    print(f"  手动 ROI:     {total_manual}")
    if dry_run:
        print("  (dry-run 模式，未实际写入)")


def main():
    parser = argparse.ArgumentParser(description="批量图像人脸打码")
    parser.add_argument("input_dir", help="输入图像目录")
    parser.add_argument("output_dir", help="输出图像目录")
    parser.add_argument("--roi", help="手动 ROI YAML 配置文件", default=None)
    parser.add_argument("--dry-run", action="store_true", help="仅检测，不写入")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"错误: 输入目录不存在: {input_dir}")
        sys.exit(1)

    roi_config = None
    if args.roi:
        roi_config = load_roi_config(args.roi)
        print(f"已加载手动 ROI 配置: {len(roi_config)} 个文件")

    print(f"输入: {input_dir}  ->  输出: {args.output_dir}")
    process_directory(args.input_dir, args.output_dir, roi_config, args.dry_run)


if __name__ == "__main__":
    main()
