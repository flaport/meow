import shutil
from collections.abc import Generator
from pathlib import Path

import pytest
from papermill.engines import papermill_engines
from papermill.execute import raise_for_execution_errors
from papermill.iorw import load_notebook_node
from papermill.utils import nb_kernel_name

TEST_DIR = Path(__file__).resolve().parent
NBS_DIR = TEST_DIR / "nbs"
EXAMPLES_DIR = TEST_DIR.parent / "nbs"
NBS_FAIL_DIR = TEST_DIR / "failed"

shutil.rmtree(NBS_FAIL_DIR, ignore_errors=True)
NBS_FAIL_DIR.mkdir(exist_ok=True)


def _find_notebooks(folder: Path) -> Generator[Path, None, None]:
    folder = Path(folder).resolve()
    for root, _, files in folder.walk():
        for file in files:
            if ("checkpoint" in file) or (not file.endswith(".ipynb")):
                continue
            yield Path(root) / file


TEST_NOTEBOOKS = [
    *sorted(_find_notebooks(NBS_DIR)),
    *sorted(_find_notebooks(EXAMPLES_DIR)),
]


@pytest.mark.parametrize("path", sorted(TEST_NOTEBOOKS))
def test_nbs(path: Path) -> None:
    fn = Path(path).name
    nb = load_notebook_node(str(path))
    nb = papermill_engines.execute_notebook_with_engine(
        engine_name=None,
        nb=nb,
        kernel_name=nb_kernel_name(nb, None),
        input_path=str(path),
        output_path=None,
    )
    raise_for_execution_errors(nb, str(NBS_FAIL_DIR / fn))
