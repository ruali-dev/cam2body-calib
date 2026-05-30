# cam2body-calib

单目相机到车体坐标系的外参标定工具。

通过图像中已知 3D 位置的标志物（ArUco 标记），用 PnP 方法估计相机在车体坐标系（body/base_link）下的 6-DoF 位姿。

## 原理

已知相机内参 K/D、标志物在车体系下的 3D 坐标、以及图像中对应角点的 2D 像素坐标，通过 `solvePnPRansac` + LM 精化求解相机外参：

```
P_camera = R · P_body + t
```

**关键**：`solvePnP` 输出的 tvec 不是相机位置，而是 body 原点在相机系下的坐标。相机在 body 系下的位姿需要取逆：`T_body_cam = inv(T_cam_body)`。

详见配套博文：[从一张图反推相机在哪：PnP 外参估计的原理、实现与坐标系之坑](https://blog.csdn.net/)

## 坐标系

### Body 坐标系（base_link）

| 轴 | 方向 |
|----|------|
| X | 前（车头） |
| Y | 左 |
| Z | 上 |

默认右手系。也支持在 marker_layout.yaml 中声明左手系（`handedness: left`，x=前, y=右, z=上），工具会自动转换。

### OpenCV 相机坐标系

| 轴 | 方向 |
|----|------|
| X | 右 |
| Y | 下 |
| Z | 前 |

工具同时输出三个矩阵：
- `T_cam_body`（optical frame，solvePnP 直接输出）
- `T_body_camera_optical`（取逆后）
- `T_body_camera_link`（camera_link frame，和 body 系同轴，**推荐使用**）

### RPY 约定

```
R = Rz(yaw) · Ry(pitch) · Rx(roll)   （固定轴 XYZ 外旋）

roll:  绕 body x（前）
pitch: 绕 body y（左）
yaw:   绕 body z（上），左转为正
```

> RPY 值包含 camera 系与 body 系之间约 90° 的基础旋转。判断相机朝向请以 "Camera Axes in Body Frame" 方向向量为准，不要单独依赖 RPY 数值。

## 安装

需要 Python 3.10+ 和 [uv](https://docs.astral.sh/uv/)：

```bash
git clone https://github.com/your-username/cam2body-calib.git
cd cam2body-calib
uv sync
```

## 快速开始

### 1. 准备文件

最少需要两个配置文件和一个图像：

**camera.yaml** — 相机内参：

```yaml
camera_name: "my_camera"
model: pinhole              # pinhole 或 fisheye
image_width: 1920
image_height: 1080
K:
  - [800.0, 0.0, 960.0]
  - [0.0, 800.0, 540.0]
  - [0.0, 0.0, 1.0]
D: [-0.3, 0.1, 0.0, 0.0, 0.0]  # pinhole: 5系数; fisheye: 4系数
```

**marker_layout.yaml** — 标志物 3D 坐标（body 系，单位米）：

```yaml
body_frame:
  name: vehicle
  convention: x_forward_y_left_z_up
  handedness: right

dictionary: DICT_4X4_50

markers:
  0:
    corners_body:
      - [1.0,  0.25, 0.2]   # top-left
      - [1.0, -0.25, 0.2]   # top-right
      - [1.0, -0.25, 0.1]   # bottom-right
      - [1.0,  0.25, 0.1]   # bottom-left
```

> 角点顺序必须和 OpenCV ArUco 检测顺序一致（顺时针，从 marker 的左上角开始）。顺序不一致会导致 PnP 算出错误姿态。

### 2. 运行 ArUco 自动标定

```bash
uv run cam2body-calib estimate \
  -i data/image.jpg \
  -c configs/camera.yaml \
  -l configs/marker_layout.yaml \
  -o outputs/result.png
```

### 3. 查看结果

终端输出包括：
- T_cam_body、T_body_camera_optical、T_body_camera_link 三个 4×4 矩阵
- 右手系下的 xyz 位置和 link RPY
- 重投影误差（mean / max / inlier count）
- 质量评级（Excellent / Good / Fair / Poor）

可视化图像中：
- 绿色边框 = 检测到的 marker
- 红色实心圆 = 检测角点
- 蓝色十字 = 重投影角点
- 黄色线 = 误差向量

## 手动标注模式

适用于 marker 自动检测不可用或需要更高精度的场景：

```bash
uv run cam2body-calib annotate \
  -i data/image.jpg \
  -c configs/camera.yaml \
  -o outputs/annotations.yaml
```

操作：
| 操作 | 按键 |
|------|------|
| 点击角点 | 鼠标左键 |
| 撤销 | 鼠标右键 |
| 缩放 | 滚轮 |
| 平移 | 中键拖拽 |
| 保存退出 | S |
| 清除所有点 | C |
| 重置视图 | R |
| 退出 | Q / ESC |

保存后编辑 YAML 文件填入 body 系 3D 坐标，然后用 Python 调用 `PnPSolver`。

## 坐标系切换（Export Profile）

不同的下游系统对 pose6 输出格式的要求不同。工具支持通过 export profile 切换，在 `marker_layout.yaml` 中声明：

```yaml
export_profiles:
  - name: vehicle_lh_pose6
    type: pose6
```

可选导出格式：

| Profile | 坐标系 | yaw |
|---------|--------|-----|
| 不配置（默认） | x=前, y=左, z=上 (右手系) | 左转为正 |
| `vehicle_lh_pose6` | x=前, y=右, z=上 (左手系) | 右转为正 |

核心 PnP 计算始终在右手系完成，左手系结果通过 `S@R@S`（det=+1）导出。

## 判断结果质量

| 重投影误差均值 | 评价 |
|---------------|------|
| < 0.5 px | 优秀 |
| < 1.5 px | 良好 |
| < 3.0 px | 一般，建议多张图验证 |
| > 3.0 px | 不可信 |

其他检查：
- 相机位置是否在物理合理范围内
- 相机朝向是否与安装方向一致
- 多个视角结果是否一致

## 运行测试

```bash
uv sync --dev
uv run pytest tests/ -v
```

## 工程结构

```
cam2body-calib/
├── pyproject.toml
├── configs/                      # 相机内参 + marker 布局示例
│   ├── camera.example.yaml       # pinhole 相机示例
│   └── marker_layout.example.yaml
├── data/                         # 放置待标定图像
├── outputs/                      # 标定结果和可视化
├── scripts/
│   ├── generate_sample_data.py   # 合成样本数据
│   ├── blur_faces.py             # 批量人脸模糊
│   └── interactive_blur.py       # 交互式区域模糊
├── src/cam2body_calib/
│   ├── cli.py                    # CLI 入口（estimate / annotate）
│   ├── camera/model.py           # CameraModel（pinhole/fisheye + 去畸变）
│   ├── config/                   # YAML 加载 + Pydantic 校验
│   ├── estimation/               # PnP 求解、重投影误差
│   ├── exporters/                # 坐标约定导出（右手系 → 左手系）
│   ├── fiducials/                # 标志物检测（ArUco）
│   ├── geometry/                 # 4×4 变换、RPY、坐标系
│   ├── interactive/              # 手动标注工具
│   ├── io/                       # 图像 / YAML 读写
│   ├── layouts/                  # 标志物 3D 布局
│   └── visualization/            # 可视化绘制
└── tests/
```

## 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| 重投影误差大 | 内参不准 / 3D 坐标有误 / 标注偏差 | 逐点检查误差分布 |
| 位置/朝向不合理 | 坐标系映射错了 | 检查 3D 坐标的轴定义 |
| PnP 失败（0 inlier） | 3D-2D 对应关系错误 | 检查坐标轴方向和点序 |
| 鱼眼图像检测失败 | 未去畸变 | annotate/estimate 命令会自动去畸变 |
| 标志物太小/太远 | 角点定位误差大 | 保持标志物 > 30px |

## License

MIT
