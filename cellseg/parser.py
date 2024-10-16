"""Script argument parser."""

from argparse import ArgumentParser
from pathlib import Path

parser = ArgumentParser(
    prog="Image processing",
)
parser.add_argument(
    "image_dir", help="Directory with the input images.", metavar="image-dir", type=Path
)

parser.add_argument(
    "-o",
    help="Directory for the output of the processing. Default is `Results/`.",
    metavar="output-dir",
    default="Results/",
    dest="output_dir",
    type=Path,
)

parser.add_argument(
    "--save-steps",
    help="Save each step of the segmentation process as an image.",
    action="store_true",
    dest="save_steps",
)
