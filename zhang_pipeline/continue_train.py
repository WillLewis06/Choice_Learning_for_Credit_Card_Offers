import tensorflow as tf  # ensures TF is initialized and available

from support.config import load_config
from support.validate_config import validate_continue_config
from support.data_prep import load_featureless_data, load_featurebased_data
from support.save import load_full_model_from_config, make_continue_model_name
from support.losses import get_loss_fn
from training.callbacks import update_best_model, should_stop_early
from training.loops import train_one_epoch, evaluate_raw, evaluate_freq
from training.optim import get_optimizer


def main() -> None:
    """
    Continue training an already trained Keras model from a saved .keras file.

    To be run as python continue_train.py --config continue_training_config_.yaml
    See the continue_training_config_.yaml file for what arguments can be used

    This function:
    Parses CLI arguments - loads config
    Loads the dataset
    Loads a full Keras model from the confiig
    Runs a training loop with evaluation and early stopping
    Saves improved models
    """

    # 1- Load config - parse the cli args and store in the cfg dict
    cfg = load_config()
    cfg = validate_continue_config(cfg)
    print("config loaded")

    # 2- Load data from the provided .csv files
    # this returns tensors - ds are batched tensor datasets
    # frequency data is also laoded to provide a smoother evaluation metric
    model_type = cfg["model"]["type"]
    print(f"model type is: {model_type}")

    # Load data depending on model type
    if model_type == "featurebased":
        (
            train_ds,
            X_train,
            Y_train,
            test_ds,
            X_test,
            Y_test,
            freq_data,
        ) = load_featurebased_data(cfg, cfg["train"]["batch_size"])
    else:
        (
            train_ds,
            X_train,
            Y_train,
            test_ds,
            X_test,
            Y_test,
            freq_data,
        ) = load_featureless_data(cfg, cfg["train"]["batch_size"])
    print("data loaded")

    # show number of items J
    num_items = int(X_train.shape[1])
    if num_items < 2:
        raise ValueError(f"num_items must be >= 2, got {num_items}")
    print(f"loaded model has {num_items} items to model choice")

    # 3- Load full Keras model from config
    model = load_full_model_from_config(cfg)
    print("model loaded")

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

    # 4- Set the loss and optimizer functions
    train_loss_fn = get_loss_fn(cfg["train"]["loss"])
    eval_loss_fn = get_loss_fn(cfg["eval"]["loss"])
    optimizer = get_optimizer(cfg)
    print(
        f"using {cfg['train']['loss']} loss function to train with {cfg['train']['optimizer']} optimiser and {cfg['eval']['loss']} loss function for evaluations"
    )

    # 5- Create a name to save continued model as
    model_name = make_continue_model_name(cfg)
    print(f"model {model_name} is ready to be trained")

    # 6- Set Training hyperparameters
    epochs = cfg["train"]["epochs"]
    patience = cfg["train"]["patience"]
    eval_batch_size = cfg["eval"]["batch_size"]

    # variables to store validation metrics
    best_metrics = None
    val_history = []  # loss metric (RMSE) is stored for early stopping

    # 7- Main training loop
    print(f"training commencing for {epochs} epochs")
    for epoch in range(1, epochs + 1):
        print(f"Training epoch {epoch}")

        # train model one epoch
        train_loss = train_one_epoch(
            model, model_type, train_ds, train_loss_fn, optimizer, cfg["train"]["l2"]
        )
        print(f"train_loss={train_loss:.6f} ")

        # evaluate on freq data if available -> if not raw data
        if "test_freq" in freq_data:
            freq_loss = evaluate_freq(
                model, model_type, X_test, freq_data["test_freq"], eval_batch_size
            )
            print(f"frequency based rmse is: {freq_loss:.6f}")
        else:
            freq_metrics = None

        val_loss = evaluate_raw(
            model, model_type, X_test, Y_test, eval_loss_fn, eval_batch_size
        )
        print(f"avg {cfg['eval']['loss']} is {val_loss:.6f}")

        val_history.append(val_loss)

        # save best model
        best_metrics = update_best_model(
            val_loss, best_metrics, model, model_name, cfg["checkpoint_dir"]
        )

        # early stopping
        if patience is not None and should_stop_early(val_history, patience):
            print(
                f"model hasn't improved for {patience} epochs - early stopping at epoch {epoch}"
            )
            break

    print("Continued training complete.")
    print(f"Model best {cfg['eval']['loss']} on raw data: {best_metrics:.6f}")


if __name__ == "__main__":
    main()
