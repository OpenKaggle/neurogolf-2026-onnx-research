from __future__ import annotations

import argparse
import json
import pathlib
import shutil


def patch_source(source: str, input_root: pathlib.Path, comp_dir: pathlib.Path, working: pathlib.Path, output_zip: pathlib.Path) -> str:
    source = source.replace(
        "INPUT_ROOT = Path('/kaggle/input')",
        f"INPUT_ROOT = Path({str(input_root)!r})",
    )
    source = source.replace(
        "WORKING = Path('/kaggle/working')",
        f"WORKING = Path({str(working)!r})",
    )
    source = source.replace(
        "OUTPUT_ZIP = WORKING / 'submission.zip'",
        f"OUTPUT_ZIP = Path({str(output_zip)!r})",
    )
    source = source.replace(
        "COMP_DIR = Path('/kaggle/input/competitions/neurogolf-2026')\nif not COMP_DIR.exists():\n    COMP_DIR = Path('/kaggle/input/neurogolf-2026')",
        f"COMP_DIR = Path({str(comp_dir)!r})",
    )
    source = source.replace(
        'ROOT = Path("/kaggle/working")',
        f"ROOT = Path({str(working)!r})",
    )
    return source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook", default="public_kernels/octavi_6042/6042-85-per-task-hand-built-onnx-solvers.ipynb")
    parser.add_argument("--input-root", default="data")
    parser.add_argument("--base-dir", default="data/jsrdcht_6029/raw")
    parser.add_argument("--comp-dir", default="data/neurogolf-2026/raw")
    parser.add_argument("--working", default="submissions/octavi_6042_work")
    parser.add_argument("--output-zip", default="submissions/submission_octavi_6042.zip")
    args = parser.parse_args()

    notebook = pathlib.Path(args.notebook).resolve()
    input_root = pathlib.Path(args.input_root).resolve()
    base_dir = pathlib.Path(args.base_dir).resolve()
    comp_dir = pathlib.Path(args.comp_dir).resolve()
    working = pathlib.Path(args.working).resolve()
    output_zip = pathlib.Path(args.output_zip).resolve()

    input_root.mkdir(parents=True, exist_ok=True)
    working.mkdir(parents=True, exist_ok=True)
    output_zip.parent.mkdir(parents=True, exist_ok=True)

    # The public notebook locates the 6029 anchor by searching for this slug.
    slug_dir = input_root / "neurogolf-6029-submission-bundle"
    if slug_dir.is_symlink():
        slug_dir.unlink()
    if slug_dir.exists() and not (slug_dir / "task001.onnx").exists():
        shutil.rmtree(slug_dir)
    if not slug_dir.exists():
        shutil.copytree(base_dir, slug_dir)

    namespace: dict[str, object] = {"__name__": "__main__"}
    nb = json.loads(notebook.read_text())
    for index, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        source = patch_source(source, input_root, comp_dir, working, output_zip)
        code = compile(source, f"{notebook.name}:cell{index}", "exec")
        exec(code, namespace)

    print(f"built {output_zip}")


if __name__ == "__main__":
    main()
