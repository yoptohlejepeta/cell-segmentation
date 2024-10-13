"""Main script.

DATA_PATH = "../Images/image/"
OUTPUT_PATH = "../Results/"
"""

import time

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
import mahotas as mh


from cellseg.parser import parser
from cellseg.src.script import check_dirs, img_processing_3

console = Console()


def main():  # noqa: ANN201, D103
    args = parser.parse_args()

    console.print(
        Panel(
            f"[bold][yellow]Analysis started![/yellow][/bold]\n\n"
            f"[bold]Image Directory:[/bold] {escape(args.image_dir)}\n"
            f"[bold]Output Directory:[/bold] {escape(args.output_dir)}",
            title="Info",
            border_style="blue",
            expand=False,
        )
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("[cyan]Processing images...", total=None)

        start_time = time.monotonic()
        try:
            list_of_input_data, output_path = check_dirs(args.image_dir, args.output_dir)
        except TypeError as e:
            console.print(
                Panel(
                    f"[bold][red]Error:[/red][/bold] {e}",
                    title="Error",
                    border_style="red",
                    expand=False,
                )
            )
            return
        except FileNotFoundError as e:
            console.print(
                Panel(
                    f"[bold][red]Error:[/red][/bold] {e}",
                    title="Error",
                    border_style="red",
                    expand=False,
                )
            )
            return
        for image in list_of_input_data:
            mod_output = output_path + f"{image}/"

            try:
                input_data = args.image_dir + image
                img = mh.imread(input_data)
            except Exception as e:
                console.print(
                    Panel(
                        f"[bold][red]Error:[/red][/bold] {e}",
                        title="Error",
                        border_style="red",
                        expand=False,
                    )
                )
                continue

            img_processing_3(img, mod_output)

        finish_time = time.monotonic()
        elapsed_time = finish_time - start_time

        progress.update(task, completed=True)

    console.print(
        Panel(
            f"[bold][yellow]Analysis completed in {elapsed_time:.2f} seconds![/yellow][/bold]\n\n"
            f"[bold]Image Directory:[/bold] {escape(args.image_dir)}\n"
            f"[bold]Output Directory:[/bold] {escape(args.output_dir)}",
            title="Completed",
            border_style="green",
            expand=False,
        )
    )


if __name__ == "__main__":
    main()
