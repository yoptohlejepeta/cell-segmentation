"""Script argument parser."""

from argparse import ArgumentParser

parser = ArgumentParser(
    prog="Image processing",
)
parser.add_argument("image_dir", help="Directory with the input images.")
parser.add_argument(
    "--output_dir",
    default="Results/",
    help="Directory for the output of the processing. Default is `Results/`.",
)
parser.add_argument(
    "--save_steps",
    help="Save each step of the segmentation process as an image.",
    action="store_true",
)
