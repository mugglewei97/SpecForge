"""Opt-in physical batch splitting without changing the logical data cursor.

Losses are sample-weighted means of microbatch objectives, NOT guaranteed to
equal a full-batch token-normalized/nonlinear objective. No no_sync is used:
ordinary FSDP backward frees/shards parameters after each microbatch.
"""


def microbatch_slices(batch_size, micro_batch_size):
    if batch_size < 1 or micro_batch_size < 1:
        raise ValueError("batch and microbatch sizes must be positive")
    return [(start, min(start + micro_batch_size, batch_size))
            for start in range(0, batch_size, micro_batch_size)]


def validate_microbatch_resume(saved_args, current_args):
    if saved_args is None:
        raise ValueError("Microbatch resume requires saved training args to validate the logical batch.")
    saved = saved_args if isinstance(saved_args, dict) else vars(saved_args)
    for name in ("batch_size", "accumulation_steps", "num_epochs", "seed",
                 "train_data_path", "target_model_path", "max_length", "block_size", "num_anchors"):
        if name not in saved or saved[name] != getattr(current_args, name):
            raise ValueError(f"Microbatch resume must preserve {name}: "
                             f"saved={saved.get(name)!r}, requested={getattr(current_args, name)!r}")


def forward_backward_microbatches(data, micro_batch_size, forward):
    """Forward/backward one slice at a time; caller steps optimizer once.

    forward(slice, index) returns the six OnlineDSparkModel outputs. Only
    detached scalar summaries survive a slice, so previous graphs are not kept.
    Counts are summed; other diagnostics and accuracy are sample-weighted means
    (not pooled conditional rates). All DP ranks must use matching slice sizes.
    """
    fields = ("input_ids", "attention_mask", "loss_mask", "sample_id", "source_id")
    size = len(data["input_ids"])
    if any(len(data[name]) != size for name in fields):
        raise ValueError("Microbatch fields must share their batch dimension")
    total_loss = None
    total_accuracy = None
    summaries = {}
    for index, (start, end) in enumerate(microbatch_slices(size, micro_batch_size)):
        micro = {name: data[name][start:end] for name in fields}
        result = forward(micro, index)
        loss, accuracy, components = result[0], result[1], result[5]
        weight = (end - start) / size
        (loss * weight).backward()
        detached_loss = loss.detach() * weight
        detached_accuracy = accuracy.detach() * weight
        total_loss = detached_loss if total_loss is None else total_loss + detached_loss
        total_accuracy = detached_accuracy if total_accuracy is None else total_accuracy + detached_accuracy
        for key, value in components.items():
            contribution = value.detach() if key.endswith("_count") else value.detach() * weight
            summaries[key] = summaries[key] + contribution if key in summaries else contribution
        del result, loss, accuracy, components, micro
    return total_loss, total_accuracy, summaries
