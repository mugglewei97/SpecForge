"""Prompt budgets, independent of FSDP rank count and trajectory block count."""

import math
import random


def planned_steps(num_prompts, training):
    if num_prompts < 1:
        raise ValueError("on-policy training requires at least one prompt")
    if training.num_epochs is not None:
        return math.ceil(num_prompts / training.batch_size) * training.num_epochs
    return training.max_steps


def prompt_batches(prompts, training):
    """Visit each prompt once per epoch; never duplicate an epoch's tail.

    Step-budget mode retains the original indefinitely cycling full batches.
    An epoch's final batch can be smaller; sample weighting uses its actual N.
    """
    planned_steps(len(prompts), training)
    if training.num_epochs is not None:
        for epoch in range(training.num_epochs):
            indices = list(range(len(prompts)))
            random.Random(training.seed + epoch).shuffle(indices)
            for start in range(0, len(indices), training.batch_size):
                yield epoch + 1, [
                    prompts[index]
                    for index in indices[start : start + training.batch_size]
                ]
    else:
        indices = list(range(len(prompts)))
        random.Random(training.seed).shuffle(indices)
        cursor = 0
        for _ in range(training.max_steps):
            batch = [
                prompts[indices[(cursor + i) % len(indices)]]
                for i in range(training.batch_size)
            ]
            yield cursor // len(indices) + 1, batch
            cursor += training.batch_size
