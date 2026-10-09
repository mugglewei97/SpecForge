"""Legacy epoch/step checkpoint discovery."""
import os
import re


def get_last_checkpoint(folder, prefix="epoch"):
    """
    Get the latest checkpoint directory along with its epoch and step information.

    Args:
        folder: The folder path containing checkpoints.
        prefix: The prefix for checkpoint directories, default is "epoch".

    Returns:
        tuple: (checkpoint_path, epoch, step)
               - Returns (None, None, None) if no checkpoint is found.
               - step is 0 if not present in the directory name.
    """
    content = os.listdir(folder)
    # Match: epoch_X or epoch_X_step_Y
    _re_checkpoint = re.compile(rf"^{re.escape(prefix)}_(\d+)(?:_step_(\d+))?$")

    checkpoints = [
        path
        for path in content
        if _re_checkpoint.search(path) is not None
        and os.path.isdir(os.path.join(folder, path))
    ]

    if len(checkpoints) == 0:
        return None, (0, 0)

    # Sort key: (epoch, step), step=0 when not present
    def sort_key(x):
        match = _re_checkpoint.search(x)
        epoch = int(match.group(1))
        step = int(match.group(2)) if match.group(2) else 0
        return (epoch, step)

    last_checkpoint = max(checkpoints, key=sort_key)
    match = _re_checkpoint.search(last_checkpoint)
    epoch = int(match.group(1))
    step = int(match.group(2)) if match.group(2) else 0

    return os.path.join(folder, last_checkpoint), (epoch, step)
