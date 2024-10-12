"""Main script.

DATA_PATH = "../Images/image/"
OUTPUT_PATH = "../Results/"
"""

import time
from parser import parser

from loguru import logger

from src.script import analysis

logger.add("app.log", format="{time} - {level} - {message}")

if __name__ == "__main__":
    args = parser.parse_args()

    start_time = time.monotonic()

    analysis(args.image_dir, args.output_dir)

    finish_time = time.monotonic()
    logger.info(str(finish_time - start_time))
