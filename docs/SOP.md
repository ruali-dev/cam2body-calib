# 基于视觉标志物的单目相机外参标定 SOP

> 版本：v0.1.0 | 最后更新：【待补充】

---

## 1. 目的

本 SOP 用于通过**已知 3D 位置的视觉标志物**（手动标注的特征点），估计**单目相机相对于车体坐标系（body frame）的外参**（位置 x/y/z 和姿态 roll/pitch/yaw）。

最终交付下游系统的是一组 6 自由度位姿参数（pose6）或 4×4 齐次变换矩阵。

---

## 2. 适用范围

- 普通单目相机（pinhole 模型）：已标定内参 K 和畸变 D
- 鱼眼相机（fisheye 模型）：已标定内参，通过工具自动去畸变后标定
- 标志物类型：手动标注角点
- 适用场景：标志物相对于车体坐标系的位置已通过物理测量获得

---

## 3. 核心原理简述

```
已知:  标志物角点的 3D 车体坐标 (x, y, z)      → 来自物理测量
      图像中对应角点的 2D 像素坐标 (u, v)       → 来自检测或手动标注
      相机内参矩阵 K、畸变系数 D                → 来自离线标定

步骤:
  1. solvePnP 求解 → T_cam_body (body → camera)
     X_cam = R @ X_body + t

  2. 取逆 → T_body_cam (camera → body)
     这就是相机在车体坐标系下的位姿

  3. 从 T_body_cam 提取位置和姿态角
```

**关键区分**：

| 矩阵 | 方向 | 含义 |
|------|------|------|
| `T_cam_body` | body → camera | solvePnP 直接输出，body 原点在 camera 系下的位姿 |
| `T_body_cam` | camera → body | **我们需要的结果**，相机在 body 系下的位姿 |

> 记住：`T_body_cam = inv(T_cam_body)`。solvePnP 返回的 tvec 不是相机位置。

---

## 4. 坐标系确认（最重要）

**在开始标定前，必须确认以下事项。坐标系搞错，后面一切白做。**

### 4.1 你的 3D 点用的是什么坐标系？

填写 marker_layout.yaml 时，你给的 3D 坐标用的是哪个坐标系？请确认：

| 问题 | 你的答案（请填写） |
|------|-------------------|
| X 轴指向哪个方向？ | 【待补充】例：车头前方 |
| Y 轴指向哪个方向？ | 【待补充】例：左侧 / 右侧 |
| Z 轴指向哪个方向？ | 【待补充】例：上方 |
| 是右手系还是左手系？ | 【待补充】例：伸右手，拇指 X 指前、食指 Y 指左 → 中指 Z 指上 = 右手系 |

在 `marker_layout.yaml` 中声明你的坐标系：

```yaml
body_frame:
  name: vehicle
  convention: x_forward_y_right_z_up   # 或 x_forward_y_left_z_up
  handedness: left                    # "right" 或 "left"
```

### 4.2 工具内部的 canonical frame 是什么？

工具内部所有 PnP 计算均在**右手系 body_rh** 中进行：

| 轴 | 方向 |
|----|------|
| X | 前 (forward) |
| Y | 左 (left) |
| Z | 上 (up) |

如果你在 `body_frame` 中声明了 `handedness: left`，工具会自动将你的输入点转换到 body_rh，计算完成后再转回你的坐标系。

### 4.3 下游系统要求的输出坐标系是什么？

不同下游系统对 pose6 的定义不同。工具支持通过 **export profile** 切换输出格式。

#### Profile 1: `standard_right_hand_pose6`（右手系，ROS REP-103 兼容）

| 参数 | 约定 |
|------|------|
| 坐标系 | x=前, y=左, z=上 (右手系) |
| yaw 正方向 | **左转**（yaw > 0 = 朝左方） |
| pitch 正方向 | **朝上**（pitch > 0 = 朝上） |
| Euler 顺序 | extrinsic XYZ（R = Rz·Ry·Rx） |

> 这是工具 `body_rh camera_link` 的直接输出。适用于 ROS、标准机器人坐标系。

#### Profile 2: `vehicle_lh_pose6`（左手系）

| 参数 | 约定 |
|------|------|
| 坐标系 | x=前, y=右, z=上 **(左手系)** |
| yaw 正方向 | **右转**（yaw > 0 = 朝右方） |
| pitch 正方向 | **朝上**（pitch > 0 = 朝上） |
| pitch 来源 | 相机前轴的 elevation angle（非 scipy Euler） |
| Euler 顺序 | extrinsic XYZ |

> 通过 `S @ R @ S`（det=+1）转换，safe for RPY extraction。

在 `marker_layout.yaml` 中选择：

```yaml
export_profiles:
  - name: vehicle_lh_pose6
    type: pose6
```

### 4.4 坐标系不一致的后果

即使**重投影误差很小**（< 1 px），以下症状说明坐标系搞错了：

- 相机位置 x/y/z 符合直觉，但 yaw 符号相反
- 相机看起来朝左但 yaw 显示朝右
- pitch 符号和相机实际俯仰方向相反
- 多个相机标定结果相互矛盾

**排查步骤**：检查 `body_frame` 的 `handedness` 和 `export_profiles` 配置，不要上来就怀疑 PnP 算错了。

---

## 5. 准备工作

### 5.1 硬件和软件

| 项目 | 要求 |
|------|------|
| 相机 | 已标定内参（K, D），pinhole 或 fisheye 均可 |
| 标志物 | 物理尺寸和角点坐标已知 |
| 测量工具 | 卷尺或全站仪，用于测量 marker 角点的 3D 车体坐标 |
| Python 环境 | Python 3.10+，uv 包管理器 |
| 图像 | 包含标志物的清晰图像，建议 960×768 或更高 |

### 5.2 安装

```bash
cd cam2body-calib
uv sync
```

### 5.3 需要准备的文件

| 文件 | 说明 |
|------|------|
| `camera.yaml` | 相机内参，见第 6 节 |
| `marker_layout.yaml` | marker 角点 3D 坐标 + 坐标系声明 + export profile |
| 待标定图像 | 单张或多张，放入 `data/` 目录 |

---

## 6. 配置文件说明

### 6.1 camera.yaml

```yaml
camera_name: "ne"           # 相机名称（任意）
model: fisheye              # "pinhole" 或 "fisheye"
image_width: 960
image_height: 768
K:
  - [255.2, 0.0, 478.3]    # fx, 0, cx
  - [0.0, 364.5, 386.4]    # 0, fy, cy
  - [0.0, 0.0, 1.0]
D: [0.139, -0.033, -0.006, 0.002]  # fisheye: 4系数; pinhole: 5系数
```

- `K`：内参矩阵（3×3），fx/fy 为焦距（像素），cx/cy 为主点
- `D`：畸变系数，**务必填写**。鱼眼 4 个，普通相机 5 个。填写错误会导致重投影误差大且不可排查
- `model`：必填，**填错会导致去畸变失败**

### 6.2 marker_layout.yaml

```yaml
# ── 1. 声明你的坐标系 ──
body_frame:
  name: vehicle
  convention: x_forward_y_right_z_up
  handedness: left             # 左手系 → 工具自动转到右手系计算

# ── 2. 选择输出格式 ──
export_profiles:
  - name: vehicle_lh_pose6
    type: pose6

# ── 3. ArUco 字典 ──
# dictionary: DICT_4X4_50   # 当前使用手动标注，无需字典

# ── 4. Marker 定义 ──
# 坐标单位：米
# 角点顺序：top-left → top-right → bottom-right → bottom-left（OpenCV 顺序）
markers:
  0:                          # marker ID = 0
    corners_body:
      - [1.5, 0.8, -1.4]     # corner 0: top-left
      - [2.5, 0.8, -1.4]     # corner 1: top-right
      - [2.5, 1.8, -1.4]     # corner 2: bottom-right
      - [1.5, 1.8, -1.4]     # corner 3: bottom-left
```

**角点顺序必须和 OpenCV 检测顺序一致**。如果不一致，PnP 会得到错误姿态（即使重投影误差可能看起来正常）。

### 6.3 不填 `body_frame` 和 `export_profiles` 的默认行为

- 默认 body 系为 `x=前, y=左, z=上` (右手系)
- 默认不输出任何 export profile，仅输出 canonical right-handed 结果

---

## 7. 操作步骤

### 步骤 1：固定标志物并测量

1. 将 标志物 固定在车体周围的已知位置
2. 用卷尺/全站仪测量每个 marker **四个角点**在车体坐标系下的 3D 坐标
3. 记录单位：**米**
4. 记录每个 marker 的 ID

> 测量精度直接影响标定结果。1 cm 的测量误差 ≈ 1 cm 的相机位置误差。

### 步骤 2：填写 marker_layout.yaml

按第 6.2 节的格式填写。**务必确认坐标系定义和角点顺序**。

### 步骤 3：拍摄图像

- 相机固定安装，保持和正常使用时相同的姿态
- 图像中包含至少 1 个完整可见的 marker（建议 2-4 个）
- 避免过曝、过暗、运动模糊

### 步骤 4：运行标定

#### 方式 A：手动标注角点（推荐用于验证和少量图像）

```bash
uv run cam2body-calib annotatev \
  -i data/image.png \
  -c configs/camera.yaml \
  -o outputs/annotations.yaml
```

- 在弹出窗口中鼠标左键点击角点（建议 6+ 点）
- 右键撤销，滚轮缩放
- 按 **S** 保存并自动退出
- 终端会打印坐标表，填入 3D 坐标后运行 PnP（使用 Python 脚本，参考第 8 节）

#### 方式 B：ArUco 自动检测（预留，当前未启用）

> 工程的 `estimate` 命令支持 ArUco 自动检测 + marker_layout.yaml 的批量模式。
> 当前主要使用手动标注工作流（方式 A）。如需启用自动检测，参考 `README.md`。

### 步骤 5：查看结果

#### 5a. 查看终端输出

关注以下内容：
- `Reprojection Errors`：mean、max、inlier 数量
- `Camera Pose — body_rh`：右手系下的相机位置和 link RPY
- `Export Profile — vehicle_lh_pose6`（如果配置了）：左手系格式的 x/y/z/roll/pitch/yaw

#### 5b. 查看 result.png

打开保存的可视化图像：
- 绿色边框 = 检测到的 marker
- 红色实心圆 = 真实检测角点
- 蓝色十字 = 重投影角点（PnP 预测位置）
- 黄色线 = 每个点的误差

如果红色圆和蓝色十字基本重合 → 标定质量好。如果某个点偏离很远 → 该点的 2D 或 3D 坐标可能有误。

#### 5c. 物理合理性检查

| 检查项 | 方法 |
|--------|------|
| x/y/z 是否符合相机安装位置？ | 和实际安装位置比对，误差应在 cm 级 |
| yaw 方向是否符合相机朝向？ | NE 相机应该看到右前方，yaw 应 > 0（vehicle_lh profile 约定） |
| pitch 是否合理？ | 鱼眼通常朝下一些（覆盖地面），pitch < 0 正常 |
| camera forward axis 方向 | 看终端输出的前轴方向 (fx, fy, fz) |

### 步骤 6：保存结果

记录以下数据交给下游：
- x, y, z（米）
- roll, pitch, yaw（度）
- T_body_camera_link（4×4 矩阵）
- 重投影误差统计

---

## 8. 运行命令示例

### 示例 1：鱼眼相机 + 手动标注 + 左手系输出

```bash
# 步骤 1：标注角点（弹出窗口，鼠标点击）
uv run cam2body-calib annotate \
  -i ne/0000.png \
  -c configs/camera_ne.yaml \
  -o outputs/annotations_ne.yaml
```

标注完成后终端打印坐标表。填入 3D 坐标后运行 PnP 得到 pose6 结果。

### 示例 2：切换 export profile

在 `marker_layout.yaml` 中修改 `export_profiles` 段即可切换输出格式，不修改命令行。

---

## 9. 结果检查

### 9.1 定量检查

| 指标 | 优秀 | 良好 | 需核查 |
|------|------|------|--------|
| mean reprojection error | < 0.5 px | < 1.5 px | > 3.0 px |
| max reprojection error | < 1.5 px | < 3.0 px | > 5.0 px |
| inlier ratio | > 90% | > 70% | < 50% |

### 9.2 定性检查

- result.png 上红点和蓝色十字是否基本重合
- 各个点的误差是否均匀，某个点是否明显偏大
- 偏大的点：优先检查该点的 3D 坐标测量和 2D 点击精度

### 9.3 物理检查

- x/y/z 是否在物理合理范围（如相机装在车顶，z 应为正值且约等于相机高度）
- 多个相机的结果是否自洽（如 NE 和 NW 相机的 y 值应符号相反、大小接近）
- 用不同图像重复标定，结果是否一致（变化应在 2-3 cm、1-2° 以内）

---

## 10. 常见问题与排查

### 10.1 标志物在图像中不清晰

- 确认标志物完整可见，不被遮挡
- 鱼眼图像是否已去畸变（`--camera` 参数指定了相机 YAML，工具会自动去畸变）
- 尝试换一张光照更好的图像

### 10.2 标注点与 3D 坐标数量不一致

确保标注的 2D 点数和你提供的 3D 坐标数相同，且一一对应。

### 10.3 2D-3D 对应关系错误

**这是最隐蔽的问题**。2D 标注点和 3D 坐标的对应关系如果搞错了（如点 1 的 2D 对应了点 2 的 3D），PnP 会算出错误姿态，但重投影误差可能仍然很小。

**排查方法**：
1. 在 result.png 上逐个检查每个点的重投影
2. 确认标注顺序和 3D 坐标 列表的顺序一致

### 10.4 内参或畸变参数错误

- 确认 `camera.yaml` 中的 K、D 来自该相机的离线标定
- 不能跨相机混用内参
- fisheye 相机的 D 是 4 个系数，pinhole 是 5 个，不要混用
- `model: fisheye` 和 `model: pinhole` 必须和实际相符

### 10.5 Marker 坐标测量错误

- 重新测量有问题的 marker 角点
- 注意测量单位（米）和坐标系原点
- z 轴方向容易搞错（向上 vs 向下）

### 10.6 左右手系混用

症状：位置看起来合理，但 yaw 符号反了。

- 检查 `body_frame.handedness`
- 如果不确定，分别用 `left` 和 `right` 跑一次，看哪个 yaw 符合物理直觉
- 右手系 body_rh (x=前, y=左) 中**左转为正**，左手系 vehicle_lh (x=前, y=右) 中**右转为正**

### 10.7 OpenCV optical frame 和 camera_link 混淆

- OpenCV optical frame：x=右, y=下, z=前
- camera_link frame：x=前, y=左, z=上（和 body_rh 一致）

工具输出同时给出三个矩阵：`T_cam_body`（optical）、`T_body_camera_optical`、`T_body_camera_link`。如果不确定用哪个，**用 `T_body_camera_link`**，它的 RPY 在 camera 对齐 body 时为 (0,0,0)。

### 10.8 RPY 符号和下游系统不一致

- 确认下游系统的坐标系约定（x/y/z 方向、yaw/pitch 正方向）
- 选择对应的 export profile
- 如果下游系统的约定不在现有 profile 中，可以新增一个 exporter

### 10.9 重投影误差大

按概率从高到低排查：

1. 角点 3D 坐标测量有误（某个点标错或测量不准）
2. 相机内参不准（K 或 D 有误）
3. 角点 2D 点击位置偏差大
4. 相机外参本身不稳定（marker 太小、太远、共面）

### 10.10 位置看起来对但姿态不对

- 检查是否误用了 optical RPY 而非 link RPY
- 检查 export profile 的 pitch 约定
- 看 camera forward axis 的方向向量，这比 RPY 数值更直观

---

## 11. 输出交付物

每次标定完成后，应保存以下文件：

| 文件 | 内容 |
|------|------|
| `result.png` | 可视化：检测角点 + 重投影点 + 误差线 |
| `result.yaml` 或 `.json` | x, y, z, roll, pitch, yaw + 重投影误差 + 使用的 profile 名称 |
| `T_body_camera_link` | 4×4 齐次变换矩阵 |
| 终端输出截图 | 包含重投影误差表和 pose 表 |

给下游系统的 6 个数示例（vehicle_lh_pose6 profile）：

```
x     =  0.05  m
y     =  1.20  m
z     =  1.40  m
roll  =  0.12  deg
pitch = -5.32  deg
yaw   =  2.15  deg
```

---

## 12. 附录

### A. OpenCV optical frame vs camera_link frame

```
OpenCV optical frame:          camera_link frame:
     z (fwd)                       x (fwd)
     ↑                             ↑
     |                            /
     |                           /
     |                          /
     +-----→ x (right)         +-----→ y (left)
     |                         |
     |                         |
     y (down)                  z (up)
```

两者相差一个 90° 旋转。`camera_link` 的轴方向和 body 系一致，更容易理解和验证。

### B. T_cam_body 和 T_body_cam

| 名称 | 公式 | 含义 | 来源 |
|------|------|------|------|
| T_cam_body | [R, t; 0, 1] | X_optical = R·X_body + t | solvePnP 输出 |
| T_body_camera_optical | inv(T_cam_body) | X_body = R'·X_optical + t' | 取逆 |
| T_body_camera_link | convert(T_body_camera_optical) | X_body = R''·X_link + t'' | 轴变换后 |

相机在 body 系的位置 = `T_body_camera_link[:3, 3]`。

### C. 为什么使用 export profile 而不是在核心流程里写死特定坐标系

1. **PnP 计算必须在右手系**：OpenCV solvePnP 和 scipy RPY 提取都假设右手系。左手系旋转矩阵 det=-1，scipy 拒绝提取 RPY。
2. **不同下游系统要求不同**：ROS 用 x=前 y=左 z=上（右手系），某些系统用 x=前 y=右 z=上（左手系）。写死任何一个都不灵活。
3. **可测试性**：canonical right-handed result 的 identity 验证很容易（RPY=0,0,0）。左手系的验证需要专门的转换逻辑。
4. **扩展性**：新增一个坐标系约定只需新增一个 exporter，不影响核心代码和已有测试。

### D. 相关文件

| 文件 | 说明 |
|------|------|
| `README.md` | 工程总览和快速开始 |
| `configs/camera.example.yaml` | 相机内参示例 |
| `configs/marker_layout.example.yaml` | Marker 布局示例（含坐标系声明） |
| `outputs/result.png` | 标定结果可视化 |

---

> **如有疑问或发现文档与实际行为不符，请联系【待补充】。**
