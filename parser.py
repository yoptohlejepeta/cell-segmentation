"""Script argument parser."""
from argparse import ArgumentParser

parser = ArgumentParser(
    prog="Image processing",
)
parser.add_argument("--image_dir", required=True)
parser.add_argument("--output_dir", required=True)
