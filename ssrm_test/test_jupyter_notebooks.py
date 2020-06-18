import glob
import os
from typing import Any, Union

import papermill as pm
import pytest

JUPYTER_NOTEBOOK_DIR: Union[bytes, str] = os.path.join(os.getcwd(), "notebooks")
INCLUDED_NOTEBOOK_GLOB = os.path.join(JUPYTER_NOTEBOOK_DIR, "*.ipynb")
JUPYTER_NOTEBOOK_TESTING_OUTPUT_DIR: Union[bytes, str] = os.path.join(
    os.getcwd(), "ssrm_test", "jupyter_notebook_testing_output"
)


@pytest.fixture
def generate_papermill_output_dir(tmpdir_factory: object) -> object:
    """Ensures directory exists for output notebooks. This is one of the
       required parameters for papermill.execute_notebook()
    """
    try:
        os.makedirs(JUPYTER_NOTEBOOK_TESTING_OUTPUT_DIR, exist_ok=True)
        return JUPYTER_NOTEBOOK_TESTING_OUTPUT_DIR
    except OSError as err:
        raise err


def test_all_jupyter_notebook(generate_papermill_output_dir, caplog):
    caplog.set_level("INFO", logger="papermill")
    for notebook_file_path in glob.glob(INCLUDED_NOTEBOOK_GLOB):
        this_notebook_file_name: Union[Union[bytes, str], Any] = os.path.basename(
            notebook_file_path
        )
        output_file_path = os.path.join(
            generate_papermill_output_dir, this_notebook_file_name
        )
        pm.execute_notebook(
            notebook_file_path,
            output_file_path,
            cwd=JUPYTER_NOTEBOOK_DIR,
            log_output=True,
        )
