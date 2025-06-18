import os
import shutil

import pytest
from papermill.engines import papermill_engines
from papermill.execute import raise_for_execution_errors
from papermill.iorw import load_notebook_node
from papermill.utils import nb_kernel_name

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
NBS_DIR = os.path.join(TEST_DIR, "nbs")
EXAMPLES_DIR = os.path.join(os.path.dirname(TEST_DIR), "examples")
NBS_FAIL_DIR = os.path.join(TEST_DIR, "nbs_fail")

shutil.rmtree(NBS_FAIL_DIR, ignore_errors=True)
os.mkdir(NBS_FAIL_DIR)


def _find_notebooks(dir):
    dir = os.path.abspath(os.path.expanduser(dir))
    for root, _, files in os.walk(dir):
        for file in files:
            if ("checkpoint" in file) or (not file.endswith(".ipynb")):
                continue
            yield os.path.join(root, file)


TEST_NOTEBOOKS = [
    *sorted(_find_notebooks(NBS_DIR)),
    *sorted(_find_notebooks(EXAMPLES_DIR)),
]


@pytest.mark.parametrize("path", sorted(TEST_NOTEBOOKS))
def test_nbs(path):
    fn = os.path.basename(path)
    nb = load_notebook_node(path)
    nb = papermill_engines.execute_notebook_with_engine(
        engine_name=None,
        nb=nb,
        kernel_name=nb_kernel_name(nb, None),
        input_path=path,
        output_path=None,
    )
    raise_for_execution_errors(nb, os.path.join(NBS_FAIL_DIR, fn))
