# SEGym
SEGym allows you to simulate patches for Python repos in isolated environments.
You can use such an environment to let a solver (e.g. LLM) search for a patch for a given issue until the issue is resolved.


## Prerequisites

Docker installed and running. `poetry` installed.
Setup python environment using `poetry install`

## Installation

Currently, only usable as a module, not as a script. Check `demo.ipynb` for a usage example.

## Working with the project

### Models

Supply your own `openai.Client` compatible API.

FOR LMU: use `openai_lmu.get_lmu_openai_client()` to get a ready-to-use client.

### Running the gym
Drawing strong similarities to the OpenAI gym, the `SEGym` class is the main entry point for the library. It allows you to create a new environment, reset it, and step through it.
No LLM generated content will modify local files, instead `env` starts up a docker container for every patch generation, ensuring that the host system is not affected by any potential bugs in the generated code.
For example usage, see [demo.ipynb](demo.ipynb).

#### Docker
If you have a Docker related issues with the mounted volume on Mac, the following command might fix it: `sudo ln -s "$HOME/.docker/run/docker.sock" /var/run/docker.sock`
