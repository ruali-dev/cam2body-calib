"""CLI entry point for cam2body-calib — manual annotation workflow."""

from typing import Annotated

import typer
from rich.console import Console

from .interactive.annotator import run_annotator

app = typer.Typer(
    name="cam2body-calib",
    help="Camera-to-body extrinsics calibration — manual annotation workflow.",
)
console = Console()


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
    按 S 保存到 YAML 并自动退出。

    操作说明：
        Left-click  添加标注点
        Right-click 撤销上一个点
        Scroll      缩放
        Middle-drag 平移
        S           保存并退出
        C           清除所有点
        R           重置视图
        Q / ESC     退出（不保存）
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
        console.print("\n  [bold]下一步[/bold]: 将输出的坐标表填入 3D body 坐标，然后运行 PnP。")


@app.command()
def ui(
    host: Annotated[
        str,
        typer.Option("--host", help="Server host address."),
    ] = "127.0.0.1",
    port: Annotated[
        int,
        typer.Option("--port", "-p", help="Server port."),
    ] = 8765,
):
    """Launch the web-based calibration UI.

    Starts a local web server and opens the browser.
    Full workflow: upload image → annotate points → fill 3D coords → solve PnP.
    """
    from .web.app import main as run_web

    console.print(f"[bold]启动 Web 标定界面[/bold]")
    console.print(f"  地址: [green]http://{host}:{port}[/green]")
    console.print(f"  按 Ctrl+C 停止服务器")
    console.print()
    run_web(host=host, port=port)


def main():
    """Entry point for console_scripts."""
    app()


if __name__ == "__main__":
    main()
