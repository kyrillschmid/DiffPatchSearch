"""
This modules contains the functions to apply a patch to a codebase and run tests on it.
"""

import time
import docker
import tempfile
import logging
import docker.errors
import sys
import stat
import os
import shutil
import subprocess
import xml.etree.ElementTree as ET

from . import config

logger = logging.getLogger("dockerconnector")


class MalformedPatchException(Exception):
    pass


class DockerConnector:
    """
    DockerConnector is a singleton class that connects to the Docker daemon and builds the Docker image if it does not exist.
    """

    def __init__(self):
        try:
            self.client = docker.from_env()
        except docker.errors.DockerException as e:
            print("Docker is not running")
            logger.critical("Docker is not running")
            sys.exit(1)
        self.build_image_if_not_exists()

    def build_image_if_not_exists(self, tag=config.DOCKER_TAG):
        try:
            logger.info("Getting docker image")
            image = self.client.images.get(tag)
        except docker.errors.ImageNotFound:
            logger.info("Image not found, building new image")
            image = self.client.images.build(path=".", tag=config.DOCKER_TAG)
        except Exception as e:
            logger.critical("Docker is not running", e)
            sys.exit(1)
        return image

    instance = None

    @staticmethod
    def get_instance():
        if DockerConnector.instance is None:
            DockerConnector.instance = DockerConnector()
        return DockerConnector.instance


class Container:
    """
    Container is a wrapper around a Docker container. It is used to run commands in the container and cleans up after itself.
    """

    def __init__(self, mount_dir: str):
        self.mount_dir = os.path.abspath(mount_dir)
        self.container = DockerConnector.get_instance().client.containers.run(
            image=config.DOCKER_TAG,
            detach=True,
            volumes={self.mount_dir: {"bind": "/repo", "mode": "rw"}},
            working_dir="/repo",
            tty=True,
            name=f"se_gym_container_{time.time()}",
        )

    def run_command(self, command: str):
        logger.debug(f"Running command {command}")
        return self.container.exec_run(cmd=command, stdout=True, stderr=True)

    def destroy(self):
        self.container.stop()
        self.container.remove(
            v={self.mount_dir: {"bind": "/repo", "mode": "rw"}}, force=True
        )


class CodeExecutor:
    """
    CodeExecutor is a wrapper around a Docker container. It copies the codebase for the container, and cleans up after itself.
    """

    def __init__(self, code_base_root: str, patch: str):
        self.code_base_root = code_base_root
        self.patch = patch
        self.temp_dir = tempfile.mkdtemp(prefix="se_gym_")
        logger.debug(f"Created temporary directory {self.temp_dir}")
        shutil.copytree(src=code_base_root, dst=f"{self.temp_dir}/", dirs_exist_ok=True)
        with open(f"{self.temp_dir}/file.patch", "w") as file:
            file.write(patch)
        self.container = Container(mount_dir=self.temp_dir)  # start a container

    def destroy(self):
        """
        Tear down the container and remove the temporary directory.
        """
        self.container.destroy()
        shutil.rmtree(self.temp_dir, onexc=CodeExecutor._shutil_onexc)

    @staticmethod
    def _shutil_onexc(func, path, exc_info):
        """Error handler for ``shutil.rmtree``."""
        os.chmod(path, stat.S_IWRITE)
        os.unlink(path)


def check_patch(code_base_root: str, patch: str):
    """
    Check if a patch can be applied to a codebase. The codebase will not be modified.

    Args:
        code_base_root (str): The root directory of the codebase.
        patch (str): The patch to apply to the codebase. This file might be corrupted, in which case the function will raise an MalformedPatchException.
    """
    with open(f"{code_base_root}/file.patch", "w") as file:
        file.write(patch)
    rand_path = f"./temp{str(time.time())}_file.patch"
    with open(rand_path, "w") as file:
        file.write(patch)
        logger.debug(f"writing patch to file {rand_path}")
    res = subprocess.check_output(args=config.GIT_CHECK_PATCH, cwd=code_base_root)
    if res.returncode != 0:
        logger.info(
            f"Failed to apply patch STDOUT:{res.stdout} STDERR:{res.stderr} PATCH:{patch}"
        )
        raise MalformedPatchException("Failed to apply patch", res.stdout)


def generate_patch(
    code_base_root: str, filename: str, old_code: str, new_code: str
) -> str:
    """
    Generate a patch file from the old and new code.
    """
    # discard current git changes in the codebase
    subprocess.run(config.GIT_DISCARD_CHANGES, cwd=code_base_root)
    # find the file to change
    file_path = os.path.join(code_base_root, filename)
    if not os.path.exists(file_path):
        logger.info(f"File {file_path} not found")
        raise ValueError(f"File {file_path} not found")
    # find the old code in the file
    with open(file_path, "r") as file:
        file_content = file.read()
    if old_code not in file_content:
        logger.info(f"Old code not found in the file {file_path}")
        raise ValueError(f"Old code not found in the {file_path}")
    # replace the old code with the new code
    new_file_content = file_content.replace(old_code, new_code)
    with open(file_path, "w") as file:
        file.write(new_file_content)
    # create a patch file running git diff
    patch = subprocess.run(config.GIT_DIFF, cwd=code_base_root, stdout=subprocess.PIPE)
    # discard the changes
    subprocess.run(config.GIT_DISCARD_CHANGES, cwd=code_base_root)
    return patch.stdout.decode("utf-8")


def apply_patch(code_base_root: str, patch: str):
    """
    Apply a patch to a codebase.

    Args:
        code_base_root (str): The root directory of the codebase. The directory will be copied and the patch will be applied to the copy. The original codebase will not be modified. The codebase should be a git repository.
        patch (str): The patch to apply to the codebase. This file might be corrupted, in which case the function will raise an MalformedPatchException.
    """
    executor = CodeExecutor(code_base_root, patch)
    apply_log = executor.container.run_command(config.GIT_APPLY_PATCH)  # Try to patch
    executor.destroy()
    # Check if the patch was applied successfully
    if apply_log.exit_code != 0:
        outp = apply_log.output.decode("utf-8")
        logger.info("Failed to apply patch", outp)
        raise MalformedPatchException("Failed to apply patch", outp)
    logger.info("Patch applied successfully")


def apply_patch_and_test(
    code_base_root: str, patch: str, command: str = "pytest --junitxml=testresults.xml"
) -> ET.Element:
    """
    Apply a patch to a codebase and run tests on it.

    Args:
        code_base_root (str): The root directory of the codebase.
        patch (str): The patch to apply to the codebase. This file should be a valid patch.
        command (str): The command to run in the container. Defaults to "pytest".

    Returns:
        ET.Element: The XML tree of the test results.
    """

    executor = CodeExecutor(code_base_root, patch)
    apply_log = executor.container.run_command(config.GIT_APPLY_PATCH)  # Try to patch
    if apply_log.exit_code != 0:  # this shouldn't be happening
        outp = apply_log.output.decode("utf-8")
        logger.error("Failed to apply patch", outp)
        raise MalformedPatchException("Failed to apply patch", outp)
    test_log = executor.container.run_command(command)  # Run the tests
    test_xml = executor.container.run_command("cat testresults.xml")
    executor.destroy()
    xml_str = test_xml.output.decode("utf-8")
    tree = ET.fromstring(xml_str)
    return tree


def parse_pytest_xml(tree: ET.Element) -> dict:
    """
    Parse the XML tree of a pytest test result.

    Args:
        tree (ET.Element): The XML tree of the test results.

    Returns:
        dict: A dictionary containing the test results. The keys are the test names and the values are dictionaries containing the status and the message of the test, if it failed or errored.
    """
    test_results = {}
    for testcase in tree.iter("testcase"):
        test_name = testcase.get("classname") + "." + testcase.get("name")
        test_results[test_name] = {}
        if testcase.find("failure") is not None:
            test_results[test_name]["status"] = "failed"
            test_results[test_name]["message"] = testcase.find("failure").text
        elif testcase.find("error") is not None:
            test_results[test_name]["status"] = "error"
            test_results[test_name]["message"] = testcase.find("error").text
        elif testcase.find("skipped") is not None:
            test_results[test_name]["status"] = "skipped"
            test_results[test_name]["message"] = testcase.find("skipped").text
        else:
            test_results[test_name]["status"] = "passed"
    return test_results
