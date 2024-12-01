import marimo

__generated_with = "0.9.27"
app = marimo.App(
    width="medium",
    app_title="Cell segmentation",
    auto_download=["html"],
)


@app.cell
def __():
    import marimo as mo

    mo.sidebar(
        [
            mo.md("# Segmentace buněk"),
            mo.nav_menu(
                {
                    # "#home": f"{mo.icon('lucide:square-chart-gantt')} Porovnání metod",
                    "/morph_nucleus.html": f"{mo.icon('lucide:blend')} Segmentace jader",
                    "/morph_cytoplasm.html": f"{mo.icon('lucide:scissors')} Segmentace cytoplazmy",
                },
                orientation="vertical",
            ),
        ]
    )
    return (mo,)


@app.cell
def __(mo):
    mo.md(r"""# <ins>Segmentace jader pomocí morfologických operátorů</ins>""")
    return


@app.cell
def __(mo):
    mo.mermaid("""
    graph LR
        A(Načtení obrázku) --> B(Unsharp masking) --> C(Modrý kanál) --> D(Binarizace) --> E(Eroze a dilatace) --> F(Odstranění malých objektů)
    """)
    return


@app.cell
def __():
    from optuna import load_study

    study = load_study(
        storage="sqlite:///params.db", study_name="nucleis_segmentation"
    )
    params = study.best_params
    return load_study, params, study


@app.cell
def __(mo):
    import mahotas as mh
    import matplotlib.pyplot as plt
    import numpy as np

    label = np.load("cellseg/Labels/orig/nucleus/2023_12_14_image_000.npy")
    img = mh.imread("cellseg/Images/all_images/2023_12_14_image_000.png")

    mo.carousel(
        [
            mo.vstack(
                [
                    mo.md("Původní obrázek"),
                    mo.image(
                        img,
                        rounded=True,
                        width=500,
                        style={"margin-left": "auto", "margin-right": "auto"},
                    ),
                ]
            ),
            mo.vstack(
                [
                    mo.md("Label"),
                    mo.image(
                        label,
                        rounded=True,
                        width=500,
                        style={"margin-left": "auto", "margin-right": "auto"},
                    ),
                ]
            ),
        ]
    )
    return img, label, mh, np, plt


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Unsharp masking

        **Unsharp Masking** je technika zostření obrazu, která zvýrazňuje hrany a detaily pomocí rozostřeného obrazu. Funguje na principu přičítání vysokofrekvenčních složek k původnímu obrazu.

        ### Princip fungování

        1. **Rozostření obrazu:**
               - Gaussovské rozostření vytvoří hladkou verzi obrazu.
        3. **Výpočet rozdílu:**
               - Odečte se rozostřený obraz od původního, čímž se získají detaily (vysoké frekvence).
        4. **Zesílení detailů:**
               - Tyto detaily se zesílí a přičtou zpět k původnímu obrazu.

        Matematicky lze proces vyjádřit takto:

        \[
        \text{Ostřený obraz} = \text{Původní obraz} + \alpha \cdot (\text{Původní obraz} - \text{Rozostřený obraz})
        \]

        kde \( \alpha \) představuje zesílení detailů (v procentech).

        ### Parametry

        #### 1. **Radius (poloměr)**
        - Ovlivňuje velikost rozostření a tím i velikost detailů, které se zvýrazní.
        - Určuje rozsah oblasti, která se použije pro rozostření.

        #### 2. **Percent (procento zesílení)**
        - Určuje, jak silně budou detaily zvýrazněny.
        - Vyjadřuje se v procentech a ovlivňuje kontrast detailů.

        #### 3. **Threshold**
        - Hodnoty změny jasu menší než threshold budou ignorovány.
        - Hodnoty větší než threshold budou zesíleny podle nastavení parametru Percent.
        """
    )
    return


@app.cell
def __(mo, params):
    mo.hstack(
        [
            mo.stat(params["radius"], label="Radius", bordered=True),
            mo.stat(params["percent"], label="Percent", bordered=True),
            mo.stat(params["threshold"], label="Threshold", bordered=True),
        ],
        justify="center",
    )
    return


@app.cell
def __(img, mo, params):
    import cellseg.src.image_worker as iw

    img_unsharp = iw.unsharp_mask_img(
        img,
        radius=params["radius"],
        percent=params["percent"],
        threshold=params["threshold"],
    )

    mo.image(
        img_unsharp,
        "Obrázek 2: Unsharping",
        rounded=True,
        width=500,
        style={"margin-left": "auto", "margin-right": "auto"},
    )
    return img_unsharp, iw


@app.cell
def __(mo):
    mo.md(r"""## Modrý kanál""")
    return


@app.cell
def __(img_unsharp, mo):
    r, g, b = img_unsharp[:, :, 0], img_unsharp[:, :, 1], img_unsharp[:, :, 2]

    mo.image(
        r,
        "Obrázek 3: Modrý kanál",
        rounded=True,
        width=500,
        style={"margin-left": "auto", "margin-right": "auto"},
    )
    return b, g, r


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Binarizace

        Binarizování obrázku pomocí _Otsu thresholding_
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ### Otsu Thresholding

        Otsu thresholding je metoda automatické segmentace obrazu, která určuje optimální prahovou hodnotu pro převod obrazu do binární. Cílem je minimalizovat rozptyl uvnitř tříd nebo maximalizovat rozptyl mezi třídami.
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        #### Princip fungování

        1. **Histogram obrazu:** Nejprve se vytvoří histogram zobrazující rozložení intenzity pixelů.
        2. **Rozdělení na dvě třídy:**
           - **Třída \(C_1\) (popředí):** Pixely s intenzitou \(I \leq t\)
           - **Třída \(C_2\) (pozadí):** Pixely s intenzitou \(I > t\)
        3. **Výpočet váženého rozptylu:**

               - \[
               \sigma_w^2(t) = \omega_1(t) \sigma_1^2(t) + \omega_2(t) \sigma_2^2(t)
               \]

               - \( \omega_1(t) \), \( \omega_2(t) \) jsou váhy (pravděpodobnosti tříd).
               - \( \sigma_1^2(t) \), \( \sigma_2^2(t) \) jsou rozptyly jednotlivých tříd.

        5. **Minimalizace rozptylu uvnitř tříd:** Otsu hledá hodnotu \(t\), která minimalizuje \( \sigma_w^2(t) \).
        6. **Maximalizace rozptylu mezi třídami:**

               - \[ \sigma_b^2(t) = \omega_1(t)\omega_2(t) \left(\mu_1(t) - \mu_2(t)\right)^2 \]

               - \( \mu_1(t) \), \( \mu_2(t) \) jsou střední hodnoty intenzit ve třídách.

        #### Algoritmus krok za krokem

        1. Vypočítej histogram a pravděpodobnosti intenzit pixelů.
        2. Iteruj přes všechny možné prahové hodnoty \(t\) a spočítej vážený rozptyl.
        3. Vyber hodnotu \(t\), která minimalizuje vážený rozptyl nebo maximalizuje mezitřídní rozptyl.

        #### Výhody
        - **Jednoduchost a efektivita.**
        - **Automatická detekce prahu.**
        - Vhodné pro obrazy s **bimodálním histogramem**.

        #### Nevýhody
        - Funguje optimálně, pouze pokud je histogram jasně bimodální.
        """
    )
    return


@app.cell
def __(b, mo, np):
    import cellseg.src.convert_worker as cw

    b_bin_otsu = cw.convert_grayscale_to_bin_otsu(b)
    b_bin_otsu_image = np.array(b_bin_otsu, dtype=np.float32)

    mo.image(
        b_bin_otsu_image,
        "Obrázek 4: Binarizace pomocí Otsu",
        rounded=True,
        width=500,
        style={"margin-left": "auto", "margin-right": "auto"},
    )
    return b_bin_otsu, b_bin_otsu_image, cw


@app.cell
def __(mo):
    mo.md(r"""## Eroze a dilatace""")
    return


@app.cell
def __(b_bin_otsu, iw, mo, np, params):
    b_bin_otsu_morp = iw.close_holes_remove_noise(
        b_bin_otsu, mask_size=params["mask_size"], iterations=params["iterations"]
    )
    b_bin_otsu_morp_image = np.array(b_bin_otsu_morp, dtype=np.float32)

    mo.image(
        b_bin_otsu_morp_image,
        "Obrázek 5: Binarizace pomocí Otsu a morfologické operátory",
        rounded=True,
        width=500,
        style={"margin-left": "auto", "margin-right": "auto"},
    )
    return b_bin_otsu_morp, b_bin_otsu_morp_image


@app.cell
def __(mo):
    mo.md(r"""## Odstranění malých objektů""")
    return


@app.cell
def __(b_bin_otsu_morp, iw, mo, params):
    img_labeled_nuclei = iw.remove_small_regions(
        b_bin_otsu_morp, min_size=params["min_size"]
    )

    mo.image(
        img_labeled_nuclei,
        "Obrázek 6: Odstranění malých objektů",
        rounded=True,
        width=500,
        style={"margin-left": "auto", "margin-right": "auto"},
    )
    return (img_labeled_nuclei,)


@app.cell
def __(img_labeled_nuclei, label, mo):
    from sklearn.metrics import f1_score

    f1 = f1_score(label.flatten(), img_labeled_nuclei.flatten())

    mo.stat(f1, label="F1 skóre", bordered=True)
    return f1, f1_score


@app.cell
def __(img_labeled_nuclei, label, mo):
    mo.carousel(
        [
            mo.vstack(
                [
                    mo.md("Původní label"),
                    mo.image(
                        label,
                        rounded=True,
                        width=500,
                        style={"margin-left": "auto", "margin-right": "auto"},
                    ),
                ]
            ),
            mo.vstack(
                [
                    mo.md("Segmentovaný label"),
                    mo.image(
                        img_labeled_nuclei,
                        rounded=True,
                        width=500,
                        style={"margin-left": "auto", "margin-right": "auto"},
                    ),
                ]
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
