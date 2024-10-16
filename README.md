# Cells segmentation

This repository contains a script for segmenting cells from images.

## Install dependencies

```bash
poetry install
```

## Run script

```bash
poetry run cellseg Images/test_images/ -o Results/ --save-steps
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

## TODO

- [ ] segmentovat jenom obaly (vypnout segmentaci jadra)
    - fc `close_holes_remove_noise` upraven pocet defaultnich iteraci (3 -> 5)
- [ ] pouzit na snimky Fridrich/LabeledData/Images
