import marimo

__generated_with = "0.9.27"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    return (mo,)


@app.cell
def __(mo):
    mo.md(r"""# Porovnání morfologických operátorů a SVM""")
    return


if __name__ == "__main__":
    app.run()
