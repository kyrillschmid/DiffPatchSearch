"""
This module allows to call a LLM to generate a patch file based on a given system prompt and context.
"""

import typing
import instructor
import instructor.retry
import openai
import logging
import tenacity
import time

from . import output_schema
from . import config
from . import utils
from . import client


logger = logging.getLogger("caller")


class SamplerInvalidPatchException(Exception):
    """Exception raised when the model fails to generate a valid response after MAX_RETRIES attempts"""


class SamplerTimeoutException(Exception):
    """Exception raised when the API call times out after TIMEOUT_SECONDS seconds"""


class Sampler:
    def __init__(
        self,
        code_base_root: str = None,
        output_class: output_schema.OutputSchema = output_schema.ChangePatchOutput,
    ):
        """
        Create a new Sampler for patch generation.

        Args:
            llm_client: OpenAI client object. It will be patched with instructor.
        """
        self.create_patch = self.__call__

        self.output_class = output_class
        self.output_class.code_base_root = code_base_root

    def __call__(self, system_prompt: str, context: str) -> str:
        """
        Generate a patch file based on the given context.

        Args:
            system_prompt: System prompt to instruct the model. The patch file format instructions will be appended to it.
            context: Context for the model to generate a patch file. The exact format depends on the observation.

        Returns:
            str: Patch file contents

        Raises:
            instructor.retry.InstructorRetryException: If the model fails to generate a valid response after MAX_RETRIES attempts
            openai.APITimeoutError: If the API call times out after TIMEOUT_SECONDS seconds


        TODO:
        change `max_retries=MAX_RETRIES,` to
        ```
                max_retries=tenacity.Retrying(
                    stop=tenacity.stop_after_attempt(MAX_RETRIES),
                    after=lambda _: logger.debug("invalid respone", _),
                ),
        ```
        to log the invalid responses
        """
        system_prompt_instruct = system_prompt + self.output_class.get_prompt()
        messages = [
            {"role": "system", "content": system_prompt_instruct},
            {"role": "user", "content": context},
        ]
        start_time = time.time()
        logger.debug(
            f"Calling LLM with message {messages} and model {config.MODEL_NAME}"
        )
        try:
            resp = client._Client.completions_create(
                messages=messages,
                response_model=self.output_class,
                field_name="patch_file",
                model=config.MODEL_NAME,
            )
            logger.debug(f"API call took {time.time() - start_time} seconds")
            return resp
        except instructor.retry.InstructorRetryException as e:
            logger.info(
                f"Failed to get a valid response after {config.MAX_RETRIES} attempts, last error: {e}"
            )
            raise SamplerInvalidPatchException(e)
        except openai.APITimeoutError as e:
            logger.info(
                f"API call timed out after {config.TIMEOUT_SECONDS} seconds \
                        (took {time.time() - start_time}), last error: {e}"
            )
            raise SamplerTimeoutException(e)
