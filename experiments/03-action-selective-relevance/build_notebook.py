"""Build hermes_exp3_colab.ipynb from nb_src.py.

nb_src.py is the source of truth: plain Python in "percent" format, where a line
"# %% [markdown]" starts a markdown cell (each line prefixed with "# ") and "# %%"
starts a code cell. Edit nb_src.py, then run:  python build_notebook.py
For the v2 notebook:        python build_notebook.py nb_src_v2.py
"""
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).parent
SRC = HERE / (sys.argv[1] if len(sys.argv) > 1 else "nb_src.py")
OUT = HERE / SRC.name.replace("nb_src", "hermes_exp3").replace(".py", "_colab.ipynb")


def to_cells(text):
    cells, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        lines = list(buf)
        if kind == "markdown":
            lines = [l[2:] if l.startswith("# ") else (l[1:] if l.startswith("#") else l)
                     for l in lines]
        while lines and not lines[-1].strip():
            lines.pop()
        while lines and not lines[0].strip():
            lines.pop(0)
        if not lines:
            return
        source = [l + "\n" for l in lines[:-1]] + [lines[-1]]
        cell = {"cell_type": kind, "metadata": {}, "source": source}
        if kind == "code":
            cell.update({"execution_count": None, "outputs": []})
        cells.append(cell)

    for line in text.splitlines():
        if line.startswith("# %% [markdown]"):
            flush()
            kind, buf = "markdown", []
        elif line.startswith("# %%"):
            flush()
            kind, buf = "code", []
        else:
            buf.append(line)
    flush()
    return cells


def main():
    cells = to_cells(SRC.read_text(encoding="utf-8"))
    nb = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4", "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "cells": cells,
    }
    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    n_code = sum(c["cell_type"] == "code" for c in cells)
    print(f"wrote {OUT.name}: {len(cells)} cells ({n_code} code, {len(cells) - n_code} markdown)")


if __name__ == "__main__":
    main()
