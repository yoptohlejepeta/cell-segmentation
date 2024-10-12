"""Main script.

DATA_PATH = "../Images/image/"
OUTPUT_PATH = "../Results/"
"""

import time
from parser import parser

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.script import analysis

console = Console()

if __name__ == "__main__":
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

        analysis(args.image_dir, args.output_dir)

        finish_time = time.monotonic()
        elapsed_time = finish_time - start_time

        progress.update(task, completed=True)

    console.print(
        Panel(
            f"Analysis completed in {elapsed_time:.2f} seconds!\n\n"
            f"[bold]Image Directory:[/bold] {escape(args.image_dir)}\n"
            f"[bold]Output Directory:[/bold] {escape(args.output_dir)}",
            title="Completed",
            border_style="green",
            expand=False,
        )
    )
