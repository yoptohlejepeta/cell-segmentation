"""Main file."""

from pathlib import Path

import mahotas as mh
import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, Task, TextColumn, TimeElapsedColumn
from skimage import measure, morphology, segmentation

import cellseg.src.convert_worker as cw
import cellseg.src.image_worker as iw


def img_processing_with_steps(
    image_path: Path,
    output_path: Path,
    save_steps: bool = False,
    task: Task | None = None,
    progress: Progress | None = None,
) -> None:
    """Process an image.

    Steps:
        1. Unsharp
        2. Otsu binarization
        3. Morphological operation
        4. Distnace transformation
        5. Watershed segmentation

    Args:
        image_path (Path): Path to images.
        output_path (Path): Path for results.
        save_steps (bool, optional): Save each step as image. Defaults to False.
        task (Optional[Progress.TaskID], optional): Task to update. Defaults to None.
        progress (Optional[Progress], optional): Progress bar. Defaults to None.

    Returns:
        _type_: _description_

    """
    img = mh.imread(image_path)

    output_path.mkdir(parents=True, exist_ok=True)

    plt.imsave(output_path / "00_original.png", img)

    progress.update(task, description="[bold blue]Processing - Step 1/6: Unsharp Mask[/bold blue]")
    img_unsharp = iw.unsharp_mask_img(img)
    if save_steps:
        plt.imsave(output_path / "01_unsharp.png", img_unsharp)
    progress.advance(task)

    progress.update(
        task, description="[bold blue]Processing - Step 2/6: Separate Layers[/bold blue]"
    )
    r, g, b = cw.separate_layers(img_unsharp)

    progress.update(
        task, description="[bold blue]Processing - Step 3/6: Otsu Binarization[/bold blue]"
    )
    bin_otsu = cw.convert_grayscale_to_bin_otsu(0.2 * r + 0.6 * g + 0.2 * b)
    if save_steps:
        plt.imsave(output_path / "02_bin_otsu.png", bin_otsu, cmap="gray")
    progress.advance(task)

    progress.update(
        task, description="[bold blue]Processing - Step 4/6: Morphological Operations[/bold blue]"
    )
    g_bin_otsu_morp = iw.close_holes_remove_noise(bin_otsu, iterations=6)
    if save_steps:
        plt.imsave(output_path / "03_morph.png", g_bin_otsu_morp, cmap="gray")
    progress.advance(task)

    progress.update(
        task, description="[bold blue]Processing - Step 5/6: Distance Transform[/bold blue]"
    )
    distance = mh.distance(g_bin_otsu_morp)
    if save_steps:
        plt.imsave(output_path / "04_distance.png", distance, cmap="gray")
    progress.advance(task)

    progress.update(
        task, description="[bold blue]Processing - Step 6/6: Watershed Segmentation[/bold blue]"
    )
    local_maxi = morphology.local_maxima(distance)
    markers = measure.label(local_maxi)
    labels = segmentation.watershed(-distance, markers, mask=g_bin_otsu_morp, connectivity=5)

    if save_steps:
        plt.imsave(output_path / "05_watershed.png", labels, cmap="jet")
    progress.advance(task)

    boundaries = segmentation.find_boundaries(labels)

    boundaries_mask = np.zeros_like(img[:, :, 0], dtype=bool)
    boundaries_mask[boundaries] = True
    boundaries_overlay = img.copy()
    boundaries_overlay[boundaries_mask] = [255, 0, 0]

    plt.imsave(output_path / "06_boundaries.png", boundaries_overlay)


def main() -> None:
    """Run main script."""
    import sys
    import time

    from cellseg.parser import parser

    args = parser.parse_args()

    console = Console()

    console.print(
        Panel(
            f"[bold][yellow]Analysis started![/yellow][/bold]\n\n"
            f"[bold]Image Directory:[/bold] {args.image_dir}\n"
            f"[bold]Output Directory:[/bold] {args.output_dir}",
            title="Info",
            border_style="blue",
            expand=False,
        )
    )

    start_time = time.monotonic()

    try:
        image_list = [f for f in args.image_dir.iterdir() if f.is_file()]

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("[cyan]Processing images...", total=len(image_list))

            for image_path in image_list:
                mod_output = args.output_dir / f"{image_path.stem}/"

                mod_output.mkdir(parents=True, exist_ok=True)

                img_processing_with_steps(image_path, mod_output, args.save_steps, task, progress)

                console.print(
                    f"[bold cyan]Completed processing image: {image_path.name}.[/bold cyan]"
                )

                progress.update(task, description="Processed image")
                progress.advance(task)

        end_time = time.monotonic()
        elapsed_time = end_time - start_time

        console.print(
            Panel(
                f"[bold][yellow]All images processed![/yellow][/bold]\n\n"
                f"[bold]Processed {len(image_list)} images in {elapsed_time:.2f} seconds.[/bold]",
                title="Completed",
                border_style="green",
                expand=False,
            )
        )
    except Exception as e:  # noqa: BLE001
        console.print(
            Panel(
                f"[bold][red]Error:[/red][/bold] {e}",
                title="Error",
                border_style="red",
                expand=False,
            )
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
