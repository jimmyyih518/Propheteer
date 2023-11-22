import os
import sys
import pytest
import logging

logging.basicConfig(level=logging.DEBUG)

# Run test on cpu only
os.environ['CUDA_VISIBLE_DEVICES'] = ''


@pytest.fixture(autouse=True)
def set_up_paths():
    project_root = os.path.dirname(os.path.dirname(__file__))
    parent_dir = os.path.dirname(project_root)

    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
