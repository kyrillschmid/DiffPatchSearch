from . import runner
import pydantic
import typing
import json
import abc
import logging

logger = logging.getLogger("output_schema")


class OutputSchema(pydantic.BaseModel, abc.ABC):
    code_base_root: typing.ClassVar[str] = None
    prompt: typing.ClassVar[str] = ""
    patch_file: typing.Optional[str] = ""

    @classmethod
    def get_prompt(cls) -> str:
        schema = {k: v for k, v in cls.model_json_schema().items()}  # copy
        reduced_schema = schema
        if "title" in reduced_schema:
            del reduced_schema["title"]
        if "type" in reduced_schema:
            del reduced_schema["type"]
        schema_str = json.dumps(reduced_schema)
        return cls.prompt.replace("JSON_FORMAT_STRING", schema_str)


class GitPatchOutput(OutputSchema):
    patch_file: str = pydantic.Field(
        description="Contents of a .patch file to change the codebase. Starts with 'diff --git a/'"
    )

    prompt: typing.ClassVar[str] = """
The output should be formatted as a JSON instance that conforms to the JSON schema below.\n\n
As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}
the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.\n
Here is the output schema: \n\n
```
JSON_FORMAT_STRING
```
ONLY REPLY USING JSON FORMAT. DO NOT INCLUDE ANYTHING ELSE IN YOUR RESPONSE.

EXAMPLE:
Good patch files look like this: 
```
diff --git a/src/python_env/__main__.py b/src/python_env/__main__.py
index 2b39a9f..f1e21b3 100644
--- a/src/python_env/__main__.py
+++ b/src/python_env/__main__.py
@@ -2,8 +2,10 @@

 def main():
     print("hello world")
-    return 2
+    # return 2
+    return 3


 if __name__ == "__main__":
+    print("Calling main function...")
     main()
```
It modifies an existing file. The added lines start with a '+', the removed lines start with a '-'. In the header, starting with 'diff --git a/', the file paths are specified. In the section with @@ the line numbers are specified. \n
Embed good patch files in a JSON object with the key "patch_file".\n
"""

    @pydantic.field_validator("patch_file")
    @classmethod
    def ensure_valid_patch(cls, patch_str: str) -> str:
        if cls.code_base_root is None:
            logger.warn("No code base root provided, skipping patch validation")
            return patch_str
        patch_str = (
            patch_str.replace("\r\n", "\n").replace("&#34", "'").replace(r"\\n", "\n")
        )
        if not patch_str.startswith("diff --git a/"):
            logger.debug(f"Invalid patch file {patch_str}")
            raise ValueError("Patch file must start with 'diff --git a/'")
        if cls.code_base_root is not None:
            try:
                runner.check_patch(cls.code_base_root, patch_str)
            except Exception as e:
                logger.debug(f"Invalid patch file {patch_str} error {e}")
                raise e
        else:
            logger.debug("No code base root provided, skipping patch validation")
        return patch_str


class ChangePatchOutput(OutputSchema):
    filename: str = pydantic.Field(
        description="The filename of the file to be changed. Use the exact filename that was provided to you. Do not modify it. If you are modifying multiple files, only list one file here. You will be able to modify multiple files in the next step."
    )
    old_code: str = pydantic.Field(
        description="The original code. Use the exact code that was provided to you. Do not modify it."
    )
    new_code: str = pydantic.Field(
        description="The new code to replace the original code."
    )

    prompt: typing.ClassVar[str] = """
The output should be formatted as a JSON instance that conforms to the JSON schema below.
{
    'filename': The filename of the file to be changed. Use the exact filename that was provided to you. Do not modify it. If you are modifying multiple files, only list one file here. You will be able to modify multiple files in the next step.
    'old_code': This is the original code provided above. Do not modify it, just paste the part you want to replace.
    'new_code': This code will replace the original code. 
}

Make sure you use the proper filename. Do not filenames like `/home/user/scratch/`, but always use the exact filename that you see in the prompt. Do not modify the filename. 
The `/scratch` directory does not exist, use the known directories and files.

EXAMPLE: If you want to replace the code in the file `./src/main.py` from `Hello, World!` to `Hello, new World!`, the JSON object should look like this:

{
    'filename': './src/main.py',
    'old_code': '\nif __name__ == "__main__":\n    print("Hello, World!")\n',
    'new_code': '\nif __name__ == "__main__":\n    print("Hello, new World!")\n'
}

Only reply using this exact JSON format. Do not include anything else in your response.
Only include one change in your response. If you need to make multiple changes, you will be able to do so in the next step.
Your answer must be in JSON format. Make sure to include the keys "filename", "old_code", and "new_code". Refrence filenames exactly as they are provided to you. Only use the "filename", "old_code", and "new_code" keys.
REPLACE ONLY THE CODE THAT NEEDS TO BE CHANGED, NOT THE ENTIRE FILE. WRAP THE NEW CODE IN THE SAME FUNCTION OR CLASS AS THE OLD CODE.

"""

    @pydantic.root_validator(pre=True)
    def generate_patch(cls, v):
        logger.info(f"Validating {v}")
        try:
            # remove trailing whitespace and trailing `./` and `/`
            for f in "filename", "old_code", "new_code":
                if f not in v:
                    logger.error(f"Missing field {f} in {v}")
                    raise ValueError(f"Missing field {f} in {v}")
            v["filename"] = v["filename"].strip()
            if v["filename"].startswith("./"):
                v["filename"] = v["filename"][2:]
            if v["filename"].startswith("/"):
                v["filename"] = v["filename"][1:]
            if cls.code_base_root is None:
                logger.error("No code base root provided, cannot generate patch")
                raise ValueError("No code base root provided, cannot generate patch")
            patch_str = runner.generate_patch(
                code_base_root=cls.code_base_root,
                filename=v["filename"],
                old_code=v["old_code"],
                new_code=v["new_code"],
            )
            cls.patch_file = patch_str
            v["patch_file"] = patch_str
            logger.info(f"Patch generated successfully: {cls.patch_file}")
            return v
        except Exception as e:
            logger.error("Error generating patch", exc_info=True)
            raise e
