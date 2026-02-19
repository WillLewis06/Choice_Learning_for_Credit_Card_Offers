import numpy as np
import tensorflow as tf  # only for type hints / ensuring TF is imported
from typing import Callable

from support.config import load_config
from support.validate_config import validate_eval_config
from support.data_prep import load_featureless_data, load_featurebased_data
from support.save import load_full_model_from_config
from support.losses import get_loss_fn
from training.loops import evaluate_raw, evaluate_freq


def layer_analysis(
    model: tf.keras.Model, X: tf.Tensor, Y: tf.Tensor, loss_fn: Callable
) -> None:
    """
    For the stacked model only
    the model returns a probability output for each layer
    we compute:

    accuracy per layer
    RMSE or NLL per layer (eval.loss)
    incremental improvement per layer
    for each layer ℓ>=2, rank items by how much their probabilities move
        relative to the previous layer (mean Δp across the test set)
    """
    # Forward pass once over the full test set
    probs_by_layer = model(X, training=False)
    if not isinstance(probs_by_layer, (list, tuple)):
        raise ValueError(
            "layer_analysis expects stacked model returning list of per-layer probabilities."
        )

    _, num_items = X.shape

    # num of most sensitive items at each layer to display
    top_picks = min(5, int(num_items))

    # index of actual choice - used to calculate accuracy
    y_true_idx = tf.argmax(Y, axis=1)

    # store for each layer-wise evaluations
    prev_metric = None
    prev_acc = None
    prev_p_np = None

    # Per-layer loop
    for layer, p in enumerate(probs_by_layer, start=1):
        print(f"Metrics for layer {layer}:")
        if layer == 1:
            print(
                "This first layer of the model shows the intrinsic value of the items"
            )

        # Accuracy
        y_pred_idx = tf.argmax(p, axis=1)
        correct = tf.cast(tf.equal(y_true_idx, y_pred_idx), tf.float32)
        acc = float(tf.reduce_mean(correct))
        print(f"Accuracy (% of correct predictions) for this layer is {acc:10.4f}")

        # Delta Accuracy
        if layer != 1:
            delta_acc = acc - prev_acc
            print(f"Accuracy improved by {delta_acc} compared to the previous layer")

        # Metric: average loss (consistent with training/evaluate_raw)
        metric_val = float(loss_fn(Y, p))
        print(f"avg_loss for this layer is {metric_val:10.6f}")

        # Delta loss metrics
        if layer != 1:
            delta_loss = prev_metric - metric_val
            print(f"avg_loss changed by {delta_loss} compared to the previous layer")

        # Per-layer item sensitivity: mean Δp per item versus previous layer
        p_np = p.numpy()
        if prev_p_np is not None:

            delta = p_np - prev_p_np  # (N, J)
            abs_delta = np.abs(delta)  # shows absolute change in predicted probs

            mean_abs = abs_delta.mean(axis=0)  # (J,)
            mean_delta = delta.mean(axis=0)  # shows directional change

            order = np.argsort(-mean_abs)  # descending
            top_items = order[:top_picks]  # select top ranked

            print(
                f"Top {top_picks} most affected items by the {layer} order interactions relative to the previous layer due to the Halo effect:"
            )
            for rank, j in enumerate(top_items, start=1):
                print(
                    f" #{rank}: item {int(j)} \t absolute effect: {mean_abs[j]:.4%} \t directional effects: {mean_delta[j]:.4%}"
                )

        prev_metric = metric_val
        prev_acc = acc
        prev_p_np = p_np


def main() -> None:
    """
    Evaluate a pre-trained Keras model by accuracy, RMSE and NLL

    To be run as python evaluate.py --config evaluate_config_.yaml
    See the evaluate_config_.yaml file for what arguments can be used

    This function:
    Parses CLI arguments - loads config
    Loads the dataset
    Loads a full Keras model from the config
    Evaluates the performance of this model
    If the model is of type stacked - evaluates per layer performance and improvements
    """

    # 1- Load config - parse the cli args and store in the cfg dict
    cfg = load_config()
    cfg = validate_eval_config(cfg)
    print("config loaded")

    # 2- Load data from the provided .csv files
    # this returns tensors - ds are batched tensor datasets
    # frequency data is also laoded to provide a smoother evaluation metric
    model_type = cfg["model"]["type"]
    print(f"model type is: {model_type}")
    batch_size = cfg["eval"]["batch_size"]

    # Load data depending on model type
    if model_type == "featurebased":
        (train_ds, X_train, Y_train, test_ds, X_test, Y_test, freq_data) = (
            load_featurebased_data(cfg, batch_size)
        )
    else:
        (train_ds, X_train, Y_train, test_ds, X_test, Y_test, freq_data) = (
            load_featureless_data(cfg, batch_size)
        )
    print("data loaded")

    # 3- Sanity checks on shapes
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(
            f"Train and test have different num_items: "
            f"{X_train.shape[1]} vs {X_test.shape[1]}"
        )

    num_items = int(X_train.shape[1])
    if num_items < 2:
        raise ValueError(f"num_items must be >= 2, got {num_items}")
    print(f"loaded data has {num_items} items to model choice")

    # 3- Load full Keras model from config
    model = load_full_model_from_config(cfg)
    print(f"model loaded")

    # check the load model shape matches our data
    # Extract model size J
    if hasattr(model, "num_items"):
        model_num_items = int(model.num_items)
    else:
        raise ValueError(f"Unable to get the models num_items")

    if model_num_items != num_items:
        raise ValueError(
            f"Mismatch: data has J={num_items}, but model was trained with J={model_num_items}."
        )

    # 4- Set loss function
    eval_loss_fn = get_loss_fn(cfg["eval"]["loss"])

    # 5- Run evaluation

    # if freq data is available -> assess
    if "train_freq" in freq_data:
        train_freq_metrics = evaluate_freq(
            model, model_type, X_train, freq_data["train_freq"], batch_size
        )
        print(f"train frequency data based RMSE is: {train_freq_metrics:.6f}")

    if "test_freq" in freq_data:
        test_freq_metrics = evaluate_freq(
            model, model_type, X_test, freq_data["test_freq"], batch_size
        )
        print(f"test frequency data based RMSE is: {test_freq_metrics:.6f}")

    # Evaluate using raw one-hot labels only
    train_loss = evaluate_raw(
        model, model_type, X_train, Y_train, eval_loss_fn, batch_size
    )
    print(f"raw train avg_loss is: {train_loss:.6f}")

    # same for test
    test_loss = evaluate_raw(
        model, model_type, X_test, Y_test, eval_loss_fn, batch_size
    )
    print(f"raw test avg_loss is: {test_loss:.6f}")

    # 6- if we have a stacked type model - perform layerwise evaluation
    if model_type == "stacked":
        layer_analysis(model, X_test, Y_test, eval_loss_fn)


if __name__ == "__main__":
    main()
