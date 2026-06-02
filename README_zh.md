# cam2body-calib

单目相机到车体坐标系的外参标定工具：一张图 + 若干个已知 3D 标志点 → 反求相机在 body/base_link 系下的 6-DoF 位姿。

核心是解一个 PnP 问题：你知道相机内参，也量出了几个标志物角点在车上的真实 3D 位置，图上找到了这些点的像素位置，反推相机装在哪、朝哪个方向。

## 怎么用

一行命令启动 Web 界面：

```bash
uv run cam2body-calib ui
```

浏览器自动打开，然后：

1. 上传相机内参 YAML（可选，如果不传也能看图，但 Solve 之前必须加载）
2. 拖入标定图片
3. 在图上点击标志物的四个角点（右键撤销）
4. 右侧表格填入每个角点的 3D 坐标（单位米，body 系）
5. 点 Solve PnP
6. 选择导出坐标系（右手系 / 左手系），Export YAML

![](assets/效果图.png)

`demo/` 目录下有一份完整的复现样例，包含内参、图片和 3D 坐标。

如果不想用 Web 界面，也可以走命令行手动标注：

```bash
uv run cam2body-calib --image data/image.jpg --camera configs/camera.yaml
```

弹出 OpenCV 窗口，鼠标点角点，S 保存。

## 安装

```bash
git clone https://github.com/ruali-dev/cam2body-calib.git
cd cam2body-calib
uv sync
```

需要 Python 3.10+。

## 坐标系

Body 系（base_link）：

| 轴 | 方向 |
|----|------|
| X | 前 |
| Y | 左 |
| Z | 上 |

右手系。如果你用的坐标系 Y 指向右（左手系），导出时选 `left_handed` profile 就行，工具会自动做 S@R@S 转换。

PnP 内部计算靠 `solvePnPRansac`。它返回的 `tvec` 是 **body 原点在相机系下的坐标**，不是相机位置。工具帮你取逆过了，拿 `T_body_camera_link` 用就行。

## 结果怎么看

看两个东西就够了：重投影误差和相机位置。

| 重投影误差均值 | 怎么样 |
|---------------|--------|
| < 0.5 px | 很准 |
| < 1.5 px | 还行 |
| < 3.0 px | 勉强，建议换张图再跑一次 |
| > 3.0 px | 有问题——查内参、3D 坐标、标注位置 |

相机位置 (x, y, z) 得在物理上说得通。朝向的话看 camera forward axis 方向向量比盯着 RPY 数值直观。

## 导出坐标系

导出时可选两个 profile。默认是 `left_handed`（左手系，y=右，yaw 右转为正）。ROS 用户选 `right_handed`（右手系，y=左，yaw 左转为正，REP-103 兼容）。

Web 界面有个 `?` 图标，鼠标悬停会显示每个 profile 的具体约定。

## 工程结构

```
cam2body-calib/
├── demo/                             # 复现样例
│   ├── README.md                     # 步骤 + 3D 坐标
│   ├── intrinsics.yaml               # NE 鱼眼内参
│   └── image.png                     # 标定图片
├── src/cam2body_calib/
│   ├── cli.py                        # 命令行入口
│   ├── web/                          # Web UI (FastAPI + 前端)
│   ├── estimation/                   # PnP 求解、重投影误差
│   ├── exporters/                    # 坐标系导出 (左右手系)
│   ├── geometry/                     # 变换矩阵、RPY、坐标系转换
│   ├── camera/                       # 相机模型 (pinhole/fisheye)
│   ├── interactive/                  # OpenCV 手动标注
│   └── config/                       # YAML 加载与校验
├── tests/                            # 34 个测试
└── pyproject.toml
```

## 常见问题

| 症状 | 大概率是 | 怎么查 |
|------|---------|--------|
| 重投影误差 > 3px | 内参不对、3D 坐标量错了、标注点偏了 | 逐点看误差分布，偏大的那点重点检查 |
| 相机位置离谱 | 3D 坐标的 x/y/z 轴定义和你以为的不一样 | 确认你量的坐标系和 body 系定义一致 |
| PnP 直接失败 | 2D 标注点和 3D 坐标对应关系错了 | 检查点序和方向 |
| 鱼眼图畸变严重 | 没加载内参文件 | 上传 camera YAML 让工具去畸变 |

## License

Apache-2.0
