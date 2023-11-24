import os
import shutil
import sys
import pytest
import logging
import boto3
from moto import mock_s3

logging.basicConfig(level=logging.DEBUG)

# Run test on cpu only
os.environ["CUDA_VISIBLE_DEVICES"] = ""
DEFAULT_TEST_DIR = "./pytest_tmp/"


@pytest.fixture(autouse=True)
def set_up_paths():
    project_root = os.path.dirname(os.path.dirname(__file__))
    parent_dir = os.path.dirname(project_root)

    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)


@pytest.fixture(scope="session", autouse=True)
def cleanup(request):
    # Teardown code (runs after all tests in the session)
    def finalizer():
        if os.path.exists(DEFAULT_TEST_DIR):
            shutil.rmtree(DEFAULT_TEST_DIR)

    request.addfinalizer(finalizer)
