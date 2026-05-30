"""CLI entry point for cam2body-calib.

Usage:
    cam2body-calib estimate --image <path> --camera <path> --layout <path> --output <path>
"""

from typing import Annotated

import numpy as np
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config.load import load_camera
from .estimation.pnp_solver import PnPSolver
from .estimation.result import PoseResult
from .exporters.vehicle_lh_pose6 import (
    VehicleLHPose6Exporter,
    build_exporter_from_config,
)
from .fiducials.aruco_detector import ArucoDetector
from .geometry.frames import FrameConvention, points_lh_to_rh
from .geometry.rotations import rpy_to_degrees
from .interactive.annotator import run_annotator
from .io.image_io import read_image, write_image
from .io.yaml_io import read_yaml
from .layouts.custom_marker_layout import CustomMarkerLayout
from .visualization.draw import create_visualization

app = typer.Typer(
    name="cam2body-calib",
    help="Camera-to-body extrinsics calibration using visual fiducial markers.",
)
console = Console()


@app.command()
def estimate(
    image: Annotated[
        str,
        typer.Option("--image", "-i", help="Path to input image file."),
    ],
    camera: Annotated[
        str,
        typer.Option("--camera", "-c", help="Path to camera intrinsics YAML config."),
    ],
    layout: Annotated[
        str,
        typer.Option("--layout", "-l", help="Path to marker layout YAML config."),
    ],
    output: Annotated[
        str,
        typer.Option(
            "--output", "-o",
            help="Path to save visualization image.",
        ),
    ] = "outputs/result.png",
    no_refine: Annotated[
        bool,
        typer.Option("--no-refine", help="Skip LM refinement step."),
    ] = False,
    ransac_threshold: Annotated[
        float,
        typer.Option(
            "--ransac-threshold",
            help="RANSAC reprojection error threshold in pixels.",
        ),
    ] = 3.0,
    show_rejected: Annotated[
        bool,
        typer.Option("--show-rejected", help="Print info about skipped marker IDs."),
    ] = False,
    fisheye_balance: Annotated[
        float,
        typer.Option(
            "--fisheye-balance",
            min=0.0,
            max=1.0,
            help="Fisheye undistortion balance: 0=max crop, 1=keep all pixels.",
        ),
    ] = 0.0,
):
    """Estimate camera extrinsics (T_body_cam) from a single image.

    Detects ArUco markers, matches them to known 3D positions from the layout,
    runs solvePnPRansac + LM refinement, and outputs the camera pose in the
    body/base_link coordinate frame.
    """
    # ── Load inputs ──────────────────────────────────────────────
    console.print("[bold]Loading inputs...[/bold]")

    try:
        cam = load_camera(camera)
        model_label = f"[cyan]{cam.model}[/cyan]"
        console.print(f"  Camera: [green]{cam.camera_name}[/green] "
                       f"({cam.image_width}x{cam.image_height}, model={model_label})")
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[red]Error loading camera config:[/red] {e}")
        raise typer.Exit(code=1)

    try:
        img = read_image(image)
        console.print(f"  Image: [green]{image}[/green] "
                       f"({img.shape[1]}x{img.shape[0]})")
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[red]Error reading image:[/red] {e}")
        raise typer.Exit(code=1)

    # ── Undistort if needed ──────────────────────────────────────
    if np.any(cam.D != 0):
        console.print(f"\n[bold]Undistorting ({cam.model} model)...[/bold]")
        img, cam = cam.undistort_image(img, balance=fisheye_balance)
        console.print(f"  Rectified K: fx={cam.K[0, 0]:.1f}, fy={cam.K[1, 1]:.1f}, "
                       f"cx={cam.K[0, 2]:.1f}, cy={cam.K[1, 2]:.1f}")
        console.print(f"  Output size: {cam.image_width}x{cam.image_height}")

    try:
        layout_data = read_yaml(layout)
        marker_layout = CustomMarkerLayout(layout_data)
        body_frame = FrameConvention.from_config(layout_data.get("body_frame"))
        console.print(f"  Layout: [green]{layout}[/green] "
                       f"({len(marker_layout.known_ids())} markers, "
                       f"dict={marker_layout.dictionary_name})")
        if body_frame.is_left_handed:
            console.print(f"  Body frame: [cyan]{body_frame.name}[/cyan] "
                           f"({body_frame.handedness}-handed, "
                           f"x={body_frame.x_axis} y={body_frame.y_axis} z={body_frame.z_axis})")
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[red]Error loading marker layout:[/red] {e}")
        raise typer.Exit(code=1)

    # ── Detect markers ───────────────────────────────────────────
    console.print("\n[bold]Detecting markers...[/bold]")

    dictionary_name = marker_layout.dictionary_name
    try:
        detector = ArucoDetector(dictionary_name)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)

    detection = detector.detect(img)
    console.print(f"  Detected: [bold]{detection.num_detected}[/bold] markers")

    if detection.num_detected == 0:
        console.print(
            "[red]Error:[/red] No ArUco markers detected in the image. "
            "Check image quality, lighting, marker visibility, and dictionary choice."
        )
        raise typer.Exit(code=1)

    # ── Build 3D-2D correspondences ──────────────────────────────
    console.print("\n[bold]Building 3D-2D correspondences...[/bold]")

    obj_points_list = []
    img_points_list = []
    skipped_ids = []
    matched_ids = []

    for m_id, corners_2d in zip(detection.marker_ids, detection.corners_2d):
        corners_body = marker_layout.get_corners_body(int(m_id))
        if corners_body is None:
            skipped_ids.append(int(m_id))
            continue

        matched_ids.append(int(m_id))
        # Corner order: 0=top-left, 1=top-right, 2=bottom-right, 3=bottom-left
        # Each row in corners_body[i] corresponds to corners_2d[i]
        for i in range(4):
            obj_points_list.append(corners_body[i])
            img_points_list.append(corners_2d[i])

    if skipped_ids:
        console.print(
            f"  [yellow]Skipped {len(skipped_ids)} marker(s) not in layout: "
            f"{skipped_ids}[/yellow]"
        )
        if show_rejected:
            for sid in skipped_ids:
                console.print(f"    - Marker ID {sid} detected but not in layout.")

    console.print(f"  Matched: [green]{len(matched_ids)}[/green] markers "
                   f"-> {len(obj_points_list)} point correspondences")

    if len(obj_points_list) < 4:
        console.print(
            f"[red]Error:[/red] Only {len(obj_points_list)} point correspondences "
            f"(from {len(matched_ids)} markers). Need at least 4 points for PnP. "
            "Add more markers to the layout or ensure more are detected."
        )
        raise typer.Exit(code=1)

    object_points = np.array(obj_points_list, dtype=np.float64)
    image_points = np.array(img_points_list, dtype=np.float64)

    # Convert from vehicle_lh to body_rh if left-handed.
    # All PnP math uses body_rh (right-handed).
    if body_frame.is_left_handed:
        object_points = points_lh_to_rh(object_points)

    # ── Solve PnP ────────────────────────────────────────────────
    console.print("\n[bold]Solving PnP...[/bold]")

    solver = PnPSolver(
        camera=cam,
        ransac_threshold=ransac_threshold,
        refine=not no_refine,
    )
    result = solver.solve(object_points, image_points)

    if not result.success:
        console.print(f"[red]Error:[/red] {result.message}")
        raise typer.Exit(code=1)

    # ── Build exporter from layout config ────────────────────────
    exporter = None
    export_profiles = layout_data.get("export_profiles", [])
    for profile_cfg in export_profiles if isinstance(export_profiles, list) else []:
        exporter = build_exporter_from_config(profile_cfg)
        if exporter is not None:
            break

    # ── Print results ────────────────────────────────────────────
    _print_results(result, body_frame, exporter)

    # ── Visualize and save ───────────────────────────────────────
    console.print("\n[bold]Generating visualization...[/bold]")

    vis = create_visualization(img, detection, object_points, image_points, result, cam)
    try:
        write_image(output, vis)
        console.print(f"  Saved: [green]{output}[/green]")
    except OSError as e:
        console.print(f"[red]Error saving visualization:[/red] {e}")
        raise typer.Exit(code=1)

    console.print("\n[bold green]Done.[/bold green]")


@app.command()
def annotate(
    image: Annotated[
        str,
        typer.Option("--image", "-i", help="Path to image file for annotation."),
    ],
    camera: Annotated[
        str | None,
        typer.Option("--camera", "-c", help="Optional: camera YAML for undistortion."),
    ] = None,
    output: Annotated[
        str,
        typer.Option("--output", "-o", help="Output YAML path for saved points."),
    ] = "outputs/annotations.yaml",
    fisheye_balance: Annotated[
        float,
        typer.Option(
            "--fisheye-balance",
            min=0.0,
            max=1.0,
            help="Fisheye undistortion balance parameter.",
        ),
    ] = 0.0,
):
    """交互式图像点选标注工具。

    打开图像窗口，鼠标左键点选角点，右键撤销。
    按 S 保存到 YAML，按 Q 退出并打印坐标表。

    操作说明：
        Left-click  添加标注点
        Right-click 撤销上一个点
        Scroll      缩放
        Middle-drag 平移
        S           保存到 YAML
        C           清除所有点
        R           重置视图
        Q / ESC     退出
    """
    console.print("[bold]启动交互式标注工具[/bold]")
    console.print(f"  图像: [green]{image}[/green]")
    if camera:
        console.print(f"  相机: [green]{camera}[/green]")

    points = run_annotator(
        image_path=image,
        camera_config=camera,
        output_path=output,
        fisheye_balance=fisheye_balance,
    )

    console.print(f"\n[bold]标注完成[/bold]: 共 [green]{len(points)}[/green] 个点")
    if points:
        console.print(f"  坐标已保存到: [green]{output}[/green]")
        console.print("\n  [bold]下一步[/bold]: 编辑 {output}，为每个点填入 body 坐标系 3D 坐标")
        console.print("  然后运行: cam2body-calib solve-from-annotations ...")


def _print_results(
    result: PoseResult,
    body_frame: FrameConvention | None = None,
    exporter: VehicleLHPose6Exporter | None = None,
) -> None:
    """Print pose estimation results. Optionally includes export profile output."""
    if body_frame is None:
        body_frame = FrameConvention()

    pos_rh = result.position_body
    rpy_link_rad = result.rpy_link_body_cam
    rpy_link_deg = rpy_to_degrees(rpy_link_rad)
    stats = result.reprojection_stats

    # ── body_rh transforms ──
    console.print()
    console.print(Panel.fit(
        "[bold]T_cam_body_rh (body_rh -> camera_optical, solvePnP output)[/bold]\n"
        "[dim]body_rh: x=fwd, y=left, z=up (right-handed). "
        "camera_optical: x=right, y=down, z=fwd.[/dim]",
        border_style="dim",
    ))
    _print_matrix(result.T_cam_body)

    console.print()
    console.print(Panel.fit(
        "[bold]T_body_rh_camera_optical[/bold]\n"
        "[dim]= inv(T_cam_body_rh). x=fwd, y=left, z=up.[/dim]",
        border_style="dim",
    ))
    _print_matrix(result.T_body_camera_optical)

    console.print()
    console.print(Panel.fit(
        "[bold]T_body_rh_camera_link[/bold]\n"
        "[dim]camera_link: x=fwd, y=left, z=up (same as body_rh). "
        "When aligned: R=I, RPY=0,0,0.[/dim]",
        border_style="dim",
    ))
    _print_matrix(result.T_body_camera_link)

    # ── body_rh position + RPY ──
    pos_table = Table(title="Camera Pose — body_rh (x=fwd, y=left, z=up, right-handed)")
    pos_table.add_column("Parameter", style="cyan")
    pos_table.add_column("Value", style="green")
    pos_table.add_row("x (forward)", f"{pos_rh[0]:.4f} m")
    pos_table.add_row("y (left)", f"{pos_rh[1]:.4f} m")
    pos_table.add_row("z (up)", f"{pos_rh[2]:.4f} m")
    pos_table.add_row("link roll", f"{rpy_link_deg[0]:.3f} deg")
    pos_table.add_row("link pitch", f"{rpy_link_deg[1]:.3f} deg")
    pos_table.add_row("link yaw", f"{rpy_link_deg[2]:.3f} deg")
    console.print(pos_table)

    # ── Export profile output ──
    if exporter is not None:
        _print_export_profile(result, exporter)

    # ── Reprojection errors ──
    if stats is not None:
        err_table = Table(title="Reprojection Errors")
        err_table.add_column("Metric", style="cyan")
        err_table.add_column("Value", style="green")
        err_table.add_row("Mean error", f"{stats.mean_error:.4f} px")
        err_table.add_row("Max error", f"{stats.max_error:.4f} px")
        err_table.add_row("Inliers", f"{stats.inlier_count}/{stats.total_points}")
        console.print(err_table)

        if stats.mean_error < 0.5:
            quality = "[green]Excellent[/green]"
        elif stats.mean_error < 1.5:
            quality = "[green]Good[/green]"
        elif stats.mean_error < 3.0:
            quality = "[yellow]Fair[/yellow]"
        else:
            quality = "[red]Poor[/red]"
        console.print(f"  Quality: {quality}")


def _print_export_profile(
    result: PoseResult,
    exporter: VehicleLHPose6Exporter,
) -> None:
    """Print exported pose in a custom coordinate convention."""
    pose6 = exporter.export(result.T_body_camera_link)

    console.print()
    console.print(Panel.fit(
        f"[bold]Export Profile: {pose6.profile_name}[/bold]\n"
        f"[dim]Parent: {pose6.parent_frame}\n"
        f"Child:  {pose6.child_frame}\n"
        f"Euler:  {pose6.euler_order} (extrinsic), degrees\n"
        f"Yaw:    {pose6.yaw_convention}\n"
        f"Pitch:  {pose6.pitch_convention}[/dim]",
        border_style="cyan",
    ))
    _print_matrix(pose6.T_parent_child)

    exp_table = Table(title=f"Pose6 — {pose6.profile_name}")
    exp_table.add_column("Parameter", style="cyan")
    exp_table.add_column("Value", style="green")
    exp_table.add_column("Note", style="dim")
    exp_table.add_row("x", f"{pose6.x:.4f} m", "forward")
    exp_table.add_row("y", f"{pose6.y:.4f} m", "right (+), left (-)")
    exp_table.add_row("z", f"{pose6.z:.4f} m", "up")
    exp_table.add_row("roll", f"{pose6.roll:.4f} deg", "")
    exp_table.add_row("pitch", f"{pose6.pitch:.4f} deg", pose6.pitch_convention)
    exp_table.add_row("yaw", f"{pose6.yaw:.4f} deg", pose6.yaw_convention)
    console.print(exp_table)

    console.print(
        "[dim]注: 以上 RPY 为 vehicle_lh 左手系导出格式。"
        "核心 PnP 计算在右手系完成，此处仅为坐标约定转换。[/dim]"
    )


def _print_matrix(M: np.ndarray) -> None:
    """Print a 4x4 matrix with formatting."""
    for i in range(4):
        row_str = "  ".join(f"{M[i, j]:12.6f}" for j in range(4))
        console.print(f"  {row_str}")


def main():
    """Entry point for console_scripts."""
    app()


if __name__ == "__main__":
    main()
