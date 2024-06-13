"""
This module contains a possible genetic algorithm implementation for LLM prompt optimization, compatible with PyGAD.
"""

import typing
import pydantic
import random
import logging
from . import config
from . import client

__all__ = ["Population", "LLMPopulation"]

prompt = typing.Annotated[str, "Prompt to an LLM"]

logger = logging.getLogger("genetic")


class Children(pydantic.BaseModel):
    child1: str = pydantic.Field(
        description="First child prompt. It is a combination of parent1 and parent2."
    )
    child2: str = pydantic.Field(
        description="Second child prompt. It is a combination of parent1 and parent2, but different from child1 in some way."
    )


class Child(pydantic.BaseModel):
    child: str = pydantic.Field(
        description="Child prompt. It is a mutation of its parent."
    )


BASE_SYSTEM_PROMPT = """
You are a prompt engineer. 
A fitness of 0 means the prompt is very bad, such that the model has not been able to generate a valid response. In this case, stress the importance of readability and clarity and the correct format.
You always output in JSON format.
You are writing the prompt for a weak LLM model. The model is not very good at extrapolating from the prompt, so you need to be very clear and explicit in your instructions.
"""

CROSSOVER_SYSTEM_PROMPT = """
You are a prompt engineer. 
You are trying to improve the quality of two prompts (instructions) using a genetic algorithm by performing a crossover operation.
During the crossover operation, you combine two prompts to create two new prompts.
You are trying to maximize the fitness of the new prompts. 
The two parent prompts performed well in the previous generation, receiving fitness scores of {fitness1} and {fitness2} respectively. Fitness is a score between 0 and 1, where higher is better. A fitness score of 1 means the prompt is good, a fitness score of 0 means the prompt is very bad.
To increase the fitness of the child prompts, extract the best parts of the two parent prompts and combine them in a way that improves the overall quality. 
You know that the child prompts should be similar to the parent prompts, but not identical. 
You also know that the child prompts should be different from each other.
You know the fitness scores of the parent prompts and how they are calculated.
"""

CROSSOVER_USER_PROMPT = """
The first prompt with a fitness score of {fitness1} is:
=======================================================
{parent1}
=======================================================

The second prompt with a fitness score of {fitness2} is:
=======================================================
{parent2}
=======================================================

Based on the parent prompts, create two new prompts that are similar to the parent prompts but not identical.
"""

MUTATION_SYSTEM_PROMPT = """
You are a prompt engineer.
You are trying to improve the quality of a prompt (instructions) using a genetic algorithm by performing a mutation operation.
During the mutation operation, you modify the prompt to create a new prompt.
You are trying to maximize the fitness of the new prompt.
The parent prompt performed not so well in the previous generation, receiving a fitness score of {fitness}. Fitness is a score between 0 and 1, where higher is better. A fitness score of 1 means the prompt is good, a fitness score of 0 means the prompt is very bad.
To increase the fitness of the child prompt, make major changes to the parent prompt that improve the overall quality.
You know that the child prompt should be similar to the parent prompt, but not identical.
You know the fitness score of the parent prompt and how it is calculated.
"""

MUTATION_USER_PROMPT = """
The parent prompt with a fitness score of {fitness} is:
=======================================================
{parent}
=======================================================
Based on the parent prompt, create a new prompt that is similar to the parent prompt but not identical.
"""


def get_messages(system_prompt, user_prompt):
    return [
        dict(role="system", content=system_prompt),
        dict(role="user", content=user_prompt),
    ]


class Population:
    def __init__(
        self,
        initial_individuals: typing.List[prompt],
        sampler,
        percent_elite: float = 0.0,
        percent_mutation: float = 1.0,
        percent_crossover: float = 0.0,
    ):
        assert (
            percent_elite + percent_mutation + percent_crossover <= 1
        ), "The sum of the percentages should be less than or equal to 1."
        self.individuals = initial_individuals
        logger.debug(f"New population: {self.individuals}")
        self.sampler = sampler
        self.num_elite = int(percent_elite * len(self.individuals))
        self.num_mutation = int(percent_mutation * len(self.individuals))
        self.num_crossover = int(percent_crossover * len(self.individuals))
        self.num_random = len(self.individuals) - (
            self.num_elite + self.num_mutation + self.num_crossover
        )

    def _mutate(self, parent: prompt, fitness: float):
        logger.debug(f"Mutating {parent} with fitness {fitness}")
        model = config.EVO_MODEL_NAME or config.MODEL_NAME
        resp = client._Client.completions_create(
            messages=get_messages(
                BASE_SYSTEM_PROMPT + MUTATION_SYSTEM_PROMPT.format(fitness=fitness),
                MUTATION_USER_PROMPT.format(fitness=fitness, parent=parent),
            ),
            model=model,
            response_model=Child,
            field_name="child",
        )
        return resp

    def _crossover(
        self, parent1: prompt, parent2: prompt, fitness1: float, fitness2: float
    ):
        logger.debug(
            f"Crossover {parent1} with fitness {fitness1} and {parent2} with fitness {fitness2}"
        )
        model = config.EVO_MODEL_NAME or config.MODEL_NAME
        resp = client._Client.completions_create(
            messages=get_messages(
                BASE_SYSTEM_PROMPT
                + CROSSOVER_SYSTEM_PROMPT.format(fitness1=fitness1, fitness2=fitness2),
                CROSSOVER_USER_PROMPT.format(
                    fitness1=fitness1,
                    fitness2=fitness2,
                    parent1=parent1,
                    parent2=parent2,
                ),
            ),
            model=model,
            temperature=0.2,
            response_model=Children,
            field_name=["child1", "child2"],
        )
        return resp

    def _selection(
        self,
        fitnesses: typing.List[float],
    ):
        logger.debug(
            f"Selecting the best individuals from {self.individuals} with fitnesses {fitnesses}"
        )
        sorted_population = sorted(
            zip(self.individuals, fitnesses), key=lambda x: x[1], reverse=True
        )

        new_population = []
        new_population.extend([x[0] for x in sorted_population[: self.num_elite]])

        if self.num_mutation > 0:
            to_mutate = random.sample(
                sorted_population[self.num_elite :], self.num_mutation
            )
            for ind, fit in to_mutate:
                new_population.append(self._mutate(ind, fit))

        if self.num_crossover > 0:
            to_crossover = random.sample(sorted_population, self.num_crossover * 2)
            for i in range(0, len(to_crossover), 2):
                new_population.extend(
                    self._crossover(
                        to_crossover[i][0],
                        to_crossover[i + 1][0],
                        to_crossover[i][1],
                        to_crossover[i + 1][1],
                    )
                )

        while len(new_population) < len(self.individuals):
            rand = random.choice(self.individuals)
            if rand not in new_population or random.random() < 0.1:
                new_population.append(rand)
        self.individuals = new_population  # update the population
        logger.debug(f"New population: {self.individuals}")

    def evolve(self, fitnesses):
        """
        Update the population based on the fitness scores.
        """
        self._selection(fitnesses)

    def sample(self, observation):
        """
        Sample actions from all the individuals.
        """
        actions = []
        for i, ind in enumerate(self.individuals):
            if isinstance(observation, list):
                obs = observation[i]
            else:
                obs = observation
            try:
                actions.append(self.sampler(system_prompt=ind, context=obs))
            except Exception:
                actions.append("")
                logger.warning(f"Failed to sample {ind}. ", exc_info=True)
        return actions

    def get_action(self, individual, observation):
        """
        Get the action for a specific individual.
        """
        try:
            return self.sampler(system_prompt=individual, context=observation)
        except Exception:
            logger.warning(f"Failed to sample {individual}. ", exc_info=True)
            return ""


class LLMPopulation:
    """
    Instead of using crossover, mutation and selection, we can use only an LLM model to generate new prompts. It should take all prompts and their fitness scores as input and generate a new set of prompts.
    """
