from dataclasses import dataclass
import datasets
import os
import random
import typing
import subprocess
import logging
import regex as re

from . import utils
from . import config
from . import runner

random.seed(15)
logger = logging.getLogger("api")
__all__ = ["make"]

if not os.path.exists(config.DEFAULT_SAVE_PATH):
    os.makedirs(config.DEFAULT_SAVE_PATH)


def make(dataset: str = "princeton-nlp/SWE-bench_Lite_oracle/dev"):
    return Environment(get_ds(dataset))


def get_ds(dataset):
    if dataset == "dummy":
        import json

        with open("./dummy_dataset.json", "r") as f:
            return json.load(f)
    else:
        split = None
        if dataset.endswith("/dev") or dataset.endswith("/test"):
            split = dataset.split("/")[-1]
            dataset = "/".join(dataset.split("/")[:-1])
        return datasets.load_dataset(dataset, split=split)


def setup_repo(repo: str, environment_setup_commit: str, test_patch: str = ""):
    logger.debug(f"Setting up repo {repo} at commit {environment_setup_commit}")
    repo_slug = utils.slugify(repo)
    os.makedirs(config.DEFAULT_SAVE_PATH, exist_ok=True)
    target_path = f"{config.DEFAULT_SAVE_PATH}/{repo_slug}"
    if not os.path.exists(f"{config.DEFAULT_SAVE_PATH}/{repo_slug}"):
        subprocess.call(["git", "clone", f"https://github.com/{repo}.git", target_path])
    subprocess.call(config.GIT_DISCARD_CHANGES.split(" "), cwd=target_path)
    subprocess.call(["git", "checkout", environment_setup_commit], cwd=target_path)
    if test_patch:
        logger.debug("Applying test patch")
        with open(f"{target_path}/file.patch", "w") as f:
            f.write(test_patch)
        proc = subprocess.Popen(
            config.GIT_APPLY_PATCH.split(" "),
            cwd=target_path,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        out, err = proc.communicate()
        logger.debug(f"Applied test patch: out {out} err {err}")
        os.unlink(f"{target_path}/file.patch")
        subprocess.call(["git", "add", "."], cwd=target_path)
        subprocess.call(
            ["git", "commit", "-m", "Apply test patch"],
            cwd=target_path,
        )
    return target_path


@dataclass
class State:
    path: typing.Annotated[str, "Path to the repository"]
    issue: typing.Annotated[str, "Issue to be fixed"]
    logs: typing.Annotated[typing.Union[str, None], "Logs of previous steps"] = None
    fail_to_pass: typing.Annotated[typing.List[str], "Tests that currently fails"] = (
        None
    )


class InvalidState(State): ...


class Environment:
    def __init__(self, dataset: datasets.Dataset):
        """
        Initialize the environment with a dataset. If the dataset is not available, it will be downloaded lazily.
        """
        self.dataset = dataset
        self.current_index = None
        self.current_path = None
        self.current_issue = None
        self.test_patch = None
        self.fail_to_pass = None
        self.oracle_files = None

    def reset(self, index: typing.Optional[int] = None) -> State:
        """
        Return a new instance of the selected environment.
        """
        if index is None:
            len_ds = sum(1 for _ in self.dataset["instance_id"])
            index = random.randint(0, len_ds - 1)
        self.current_index = index
        self.current_path = setup_repo(
            self.dataset["repo"][self.current_index],
            self.dataset["environment_setup_commit"][self.current_index],
            self.dataset["test_patch"][self.current_index],
        )
        self.current_issue = self.dataset["problem_statement"][self.current_index]
        self.test_patch = self.dataset["test_patch"][self.current_index]
        self.fail_to_pass = self._parse_fail_to_pass(
            self.dataset["FAIL_TO_PASS"][self.current_index], self.current_path
        )
        try:
            self.oracle_files = self._parse_oracle_text(
                self.dataset["text"][self.current_index]
            )
        except Exception:
            logger.info("No oracle files found", exc_info=True)
            self.oracle_files = []
        return State(
            path=self.current_path,
            issue=self.current_issue,
            fail_to_pass=self.fail_to_pass,
        )

    def step(self, action: typing.Union[str, typing.List[str]]) -> State:
        """
        Perform an action in the environment.
        """
        if isinstance(action, list):
            return [self.step(a) for a in action]
        if not action:  # Sampler has produced invalid patch
            logger.info("Invalid patch, skipping")
            return InvalidState(
                path=self.current_path,
                issue=self.current_issue,
                fail_to_pass=self.fail_to_pass,
            )
        tree = runner.apply_patch_and_test(
            code_base_root=self.current_path, patch=action
        )
        log = runner.parse_pytest_xml(tree)
        return State(
            path=self.current_path,
            issue=self.current_issue,
            logs=log,
            fail_to_pass=self.fail_to_pass,
        )

    @staticmethod
    def _parse_oracle_text(text: str) -> typing.List[str]:
        pat = re.compile(r"\[start of (.*?)\]")
        return pat.findall(text)

    @staticmethod
    def _parse_fail_to_pass(fail_to_pass: str, current_path: str) -> typing.List[str]:
        """
        Parse the fail to pass string and return the list of tests that need to be fixed.
        E.g. "['test_boolean_expression_combined (expressions.tests.BasicExpressionsTests)', 'test_boolean_expression_combined_with_empty_Q (expressions.tests.BasicExpressionsTests)']"
        and current_path = "./temp/djangodjango" becomes
        ['temp/djangodjango/tests/expressions/tests.py']
        """
        tests = set()
        for test in eval(fail_to_pass):
            tests.add(
                "/".join(
                    test.split(" ")[-1]
                    .replace("(", "")
                    .replace(")", "")
                    .replace(".", "/")
                    .split("/")[:-1]
                )
                + ".py"
            )
        return [utils.find_file(root_dir=current_path, filename=test) for test in tests]
