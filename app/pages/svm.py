import marimo

__generated_with = "0.9.27"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo

    mo.sidebar(
        [
            mo.md("# Segmentace buněk"),
            mo.md("---"),
            mo.nav_menu(
                {
                    "/": f"{mo.icon('lucide:home')} Přehled",
                }
            ),
            mo.md("---"),
            mo.md("__Watershed__"),
            mo.nav_menu(
                {
                    "/watershed_nucleus": f"{mo.icon('lucide:droplet')} Jádra",
                },
                orientation="vertical",
            ),
            mo.md("---"),
            mo.md("__SVM__"),
            mo.nav_menu(
                {
                    "/svm": f"{mo.icon('lucide:scissors')} SVM segmentace",
                },
                orientation="vertical",
            ),
            mo.md("---"),
            mo.nav_menu(
                {
                    "/comparation": f"{mo.icon('lucide:scale')} Porovnání metod",
                }
            )
        ]
    )
    return (mo,)


@app.cell
def __(mo):
    mo.md(r"""# Support vector machine""")
    return


@app.cell
def __(mo):
    from pathlib import Path
    import numpy as np

    svm_result = np.load(Path("cellseg/Results/predicted/nucleus/2023_12_14_image_006.npy"))
    resized_label = np.load(Path("cellseg/Labels/resized/nucleus/2023_12_14_image_006.npy"))

    mo.hstack(
        [
            mo.image(svm_result, rounded=True),
            mo.image(resized_label, rounded=True),
        ]
    )
    return Path, np, resized_label, svm_result


@app.cell
def __(mo):
    mo.center(mo.md(r"""## Čištění"""))
    return


@app.cell
def __(mo, np, svm_result):
    import cellseg.src.image_worker as iw

    cleaned_svm = iw.close_holes_remove_noise(
        svm_result, mask_size=3, iterations=3
    )
    cleaned_svm_img = np.array(cleaned_svm, dtype=np.uint8)

    mo.image(cleaned_svm_img, rounded=True)
    return cleaned_svm, cleaned_svm_img, iw


@app.cell
def __(cleaned_svm, mo, resized_label):
    from sklearn.metrics import f1_score

    f1_score(resized_label.flatten(), cleaned_svm.flatten())

    mo.stat(
        f1_score(resized_label.flatten(), cleaned_svm.flatten()),
        label="F1 skóre",
        bordered=True,
    )
    return (f1_score,)


if __name__ == "__main__":
    app.run()
