from functools import wraps
import pickle
import logging
from inspect import signature
import glob
import os
import openai
import pandas as pd

logger = logging.getLogger("utils")


def log_to_parqet(log_filename: str, **kwargs):
    df_ = pd.DataFrame({k: [v] for k, v in kwargs.items()})
    if os.path.exists(log_filename):
        df = pd.read_parquet(log_filename)
        df = pd.concat([df, df_], ignore_index=True)
    else:
        df = df_
    df.to_parquet(log_filename)


def slugify(value):
    """
    Makes any object a url and filename friendly slug.
    Taken from Django's https://github.com/django/django/blob/main/django/utils/text.py
    """
    import unicodedata
    import re

    value = str(value)
    value = (
        unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    )
    value = re.sub(r"[^\w\s-]", "", value.lower())
    return re.sub(r"[-\s]+", "-", value).strip("-_")


def find_file(root_dir: str, filename: str) -> str:
    """
    Find a file in a directory.
    """
    root_abs = os.path.abspath(root_dir)
    for f in glob.glob(f"{root_dir}/**/{filename}", recursive=True):
        f_abs = os.path.abspath(f)
        rel_path = os.path.relpath(f_abs, root_abs).replace("\\", "/")
        if not rel_path.startswith("./"):
            rel_path = "./" + rel_path
        return rel_path
    raise FileNotFoundError(f"File {filename} not found in {root_dir}")


def check_client(client):
    try:
        client.models.list()
    except openai._exceptions.APIStatusError:
        # model not found -> probably running on ollama
        pass
    except openai._exceptions.APITimeoutError:
        # timeout -> wrong ip
        raise ValueError("API Timeout. Check if you are connected to the VPN.")


def cached(ignore=None):
    if ignore is None:
        ignore = []

    def decorator(func):
        def get_key(args, kwargs):
            sig = signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            cache_key = tuple(
                (k, v) for k, v in bound_args.arguments.items() if k not in ignore
            )
            return slugify(cache_key)

        cache_file = f".cache.{func.__name__}.pkl"
        try:
            with open(cache_file, "rb") as f:
                func.cache = pickle.load(f)
                logger.debug(
                    f"Loaded cache from {cache_file}, keys: {list(func.cache.keys())}"
                )
        except FileNotFoundError:
            logger.debug(f"Cache file {cache_file} not found. Creating new cache.")
            func.cache = {}

        @wraps(func)
        def wrapper(*args, **kwargs):
            key = get_key(args, kwargs)
            if key in func.cache:
                return func.cache[key]
            else:
                func.cache[key] = result = func(*args, **kwargs)
                with open(cache_file, "wb") as f:
                    pickle.dump(func.cache, f)
                return result

        return wrapper

    return decorator
