# Cells segmentation

This repository contains a script for segmenting cells from images.

## Install dependencies

```bash
uv sync
```

## Run script

```bash
uv run cellseg Images/test_images/ -o Results/ --save-steps
```

### Output

```text
╭─────────────── Info ───────────────╮
│ Analysis started!                  │
│                                    │
│ Image Directory: Images/BAL_image2 │
│ Output Directory: Resultstest      │
╰────────────────────────────────────╯
Completed processing image: P00002.jpg.
Completed processing image: 11a.jpg.
Completed processing image: P00001.jpg.
Completed processing image: 2c.jpg.
╭───────────── Completed ──────────────╮
│ All images processed!                │
│                                      │
│ Processed 4 images in 19.56 seconds. │
╰──────────────────────────────────────╯
```

## Odkazy na články

<https://www.sciencedirect.com/science/article/pii/S174680942100402X?casa_token=guMvylOtnCcAAAAA:egkrTmgl_iQlS44iW3CB5xMEZEQEk_m9pFcBStpRit9Poyubqg6gz32WpDIFBJek-z0pQIw>
