from typing import Optional

from support.save import save_full_model


def update_best_model(
    loss: float,
    best_loss: Optional[float],
    model,
    model_name: str,
    checkpoint_dir: str,
) -> float:
    """
    Check if validation improves -> call save_full_model

    loss: loss value of last training loop
    best_loss: previous best loss value or None
    model: Keras model
    checkpoint_path: file to save model weights

    returns: the latest best loss value found
    """

    # first epoch: no best yet
    if best_loss is None:
        save_full_model(model, checkpoint_dir, model_name)
        return loss
    else:
        if loss < best_loss:
            save_full_model(model, checkpoint_dir, model_name)
            print(f"Improved model saved as {model_name}")
            return loss

    return best_loss


def should_stop_early(
    history: list[float],
    patience: Optional[int],
) -> bool:
    """
    Check early stopping based on validation history.

    history: list of validation scores (e.g. RMSE) per epoch
    patience: number of epochs without improvement allowed

    returns: True if training should stop
    """
    if patience is None or patience <= 0:
        return False

    if len(history) <= patience:
        return False

    # best (lowest) so far
    best = min(history)

    # last `patience` values
    recent = history[-patience:]

    # if none of the recent values beat the best, stop
    if all(val > best for val in recent):
        return True

    return False
