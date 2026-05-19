# Product Recognition on Store Shelves

Computer Vision project — University of Bologna, course taught by Luigi Di Stefano. A SIFT-based pipeline that detects cereal-box products on supermarket shelves, given one reference image per product.

The project specification is in `product-recognition-on-store-shelves.pdf`; the final report is `Report.pdf`.

## Tasks

- **Task A** — single-instance detection on the five easy shelves (`scenes/e1.png` … `e5.png`). SIFT + FLANN + Lowe's ratio test + RANSAC homography + a Euclidean BGR-mean colour check, all wrapped in an iterative masking sweep. Code: `tasks/Task_A.ipynb`.
- **Task B** — multi-instance detection on the five medium shelves (`scenes/m1.png` … `m5.png`). Same front-end as Task A; the RANSAC homography is replaced with a star-model Generalised Hough Transform whose accumulator peaks correspond to individual instances. Code: `tasks/Task_B.ipynb`.

## Layout

```
tasks/         Jupyter notebooks (Task_A.ipynb, Task_B.ipynb)
models/        product reference images (0.jpg through 26.jpg)
scenes/        shelf images (easy: e1–e5; medium: m1–m5; hard: h1–h5)
figures/       figures referenced by the report
results/       per-scene detection output images
Report.typ     report source (Typst)
Report.pdf     compiled report
```

## Running the notebooks

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Open `tasks/Task_A.ipynb` or `tasks/Task_B.ipynb` and run all cells.

## Building the report

The report is written in [Typst](https://typst.app/). With Typst installed:

```bash
typst compile Report.typ          # one-shot build → Report.pdf
typst watch Report.typ            # live rebuild on save
```

## Author

Carl Fredrik Cornelis Nexmark — University of Bologna, 2026.
