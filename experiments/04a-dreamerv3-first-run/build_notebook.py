"""Build hermes_dreamer_first_run_colab.ipynb from first_run_src.py.

first_run_src.py is the source of truth, written in "percent" format:

  # %% [markdown]        starts a markdown cell (each line prefixed with "# ")
  # %%                   starts a code cell
  # %% [file] NAME       emits a code cell that writes the file NAME, taken
                         verbatim from this folder, so that the notebook is
                         self-contained but the repo keeps the single source

Edit first_run_src.py (or one of the embedded files), then run:
  python build_notebook.py
"""
import json
import pathlib

HERE = pathlib.Path(__file__).parent
SRC = HERE / "first_run_src.py"
OUT = HERE / "hermes_dreamer_first_run_colab.ipynb"


def file_cell(name):
    text = (HERE / name).read_text(encoding="utf-8").replace("\r\n", "\n")
    assert "'''" not in text, f"{name} contains ''' which the embedding cannot hold"
    return [
        f"# {name} is copied verbatim from experiments/04a-dreamerv3-first-run/{name}",
        "# in the HERMES repo by build_notebook.py. Edit it there, not here.",
        "import pathlib",
        f"pathlib.Path({name!r}).write_text(r'''{text}''', encoding='utf-8')",
        f"print('wrote', pathlib.Path({name!r}).resolve())",
    ]


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
        elif line.startswith("# %% [file] "):
            flush()
            kind, buf = "code", file_cell(line[len("# %% [file] "):].strip())
            flush()
            kind, buf = None, []
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
            "colab": {"provenance": [], "gpuType": "A100", "toc_visible": True},
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
