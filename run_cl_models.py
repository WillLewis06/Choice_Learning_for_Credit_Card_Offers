import tensorflow as tf
from typing import Any, Dict, Iterator, Tuple, Union

from choice_learn.datasets.base import load_swissmetro

# from choice_learn.datasets.expedia import load_expedia

from zhang.featureless import FeaturelessDeepHalo
from zhang.featurebased import FeatureBasedDeepHalo
from choice_learn.models.tastenet import TasteNet
from choice_learn.models.rumnet import GPURUMnet
from choice_learn.models.reslogit import ResLogit


def _as_float(x: Any) -> float:
    """
    Coverts to numpy if needed and returns a float.
    """
    if hasattr(x, "numpy"):
        return float(x.numpy())
    return float(x)


def data_block_width(x):
    """
    Takes a block from the dataset iterator and returns the width of that block.
    The Expedia load gives us tuple of blocks.
    """
    if isinstance(x, (tuple, list)):
        return int(sum(int(t.shape[-1]) for t in x))
    return int(x.shape[-1])


def get_batch_dims(ds, batch_size):
    """
    Returns the size of the batches produced by the dataset iterator.
    """
    shared, items, avail, choices = next(ds.iter_batch(batch_size=batch_size))
    return {
        "J": int(avail.shape[1]),
        "dx_shared": data_block_width(shared),
        "dx_items": data_block_width(items),
    }


class DenseViewChoiceDataset:
    """
    Proxy dataset: yields the same batches as ds, but concatenates tuple/list blocks into single dense tensors per batch (no full-dataset materialization).
    """

    def __init__(self, ds):
        self.ds = ds
        d = get_batch_dims(ds, batch_size=8)
        self._J = d["J"]
        self._dx_shared = d["dx_shared"]
        self._dx_items = d["dx_items"]

    @staticmethod
    def _concat_shared(shared):
        """
        Used to concat ds blocks into a tensori of shape (B, dx_shared)
        """
        if isinstance(shared, (tuple, list)):
            return tf.concat(list(shared), axis=1)
        return shared

    @staticmethod
    def _concat_items(items):
        """
        Used to concat ds blocks into a tensori of shape (B, dx_shared)
        """
        if isinstance(items, (tuple, list)):
            return tf.concat(list(items), axis=2)
        return items

    def iter_batch(self, batch_size=256, shuffle=False):
        """
        Yields batches of our class with the shared and items concatenated.
        """
        for shared, items, avail, choices in self.ds.iter_batch(
            batch_size=batch_size, shuffle=shuffle
        ):
            yield (
                self._concat_shared(shared),
                self._concat_items(items),
                avail,
                choices,
            )

    # Provide the getters some models call
    def get_n_items(self):
        return self._J

    def get_n_shared_features(self):
        return self._dx_shared

    def get_n_items_features(self):
        return self._dx_items

    def __len__(self):
        return len(self.ds)


def run_model(name, model, ds, batch_size, mode="optim"):
    """
    Takes a choice_learn model and dataset runs fit() and evaluate().
    """
    print(f"\nMODEL: {name}")
    try:
        model.fit(ds)
        val = model.evaluate(ds, batch_size=batch_size, mode=mode)
        print("  eval:", _as_float(val))
    except Exception as e:
        print("  FAILED:", repr(e))


def main():
    """
    Load the expedia dataset. For chosen models, define the hyperparameters, train and evalute the rsults.

    These models are: our produced featureless and featurebased deepHalo modes, Tastenet, Rumnet and Reslogit.
    """

    # Check tensorflow setup
    print("TensorFlow:", tf.__version__)
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    batch_size = 256
    lr = 1e-3

    # load expedia data using the buildin loader
    # ds_raw = load_expedia(as_frame=False, preprocessing="rumnet")
    ds_raw = load_swissmetro(as_frame=False)
    # transform to dense for easy fitting
    ds_dense = DenseViewChoiceDataset(ds_raw)

    # get the dimensions of the datasets
    dims_raw = get_batch_dims(ds_raw, batch_size=batch_size)
    dims_dense = {
        "J": ds_dense.get_n_items(),
        "dx_shared": ds_dense.get_n_shared_features(),
        "dx_items": ds_dense.get_n_items_features(),
    }

    print("Raw (tuple-capable) dims:", dims_raw)
    print("Dense-view dims:", dims_dense)

    J = dims_dense["J"]
    dx_shared = dims_dense["dx_shared"]
    dx_items = dims_dense["dx_items"]

    # Run and evalutate the featureless deepHalo model
    # Models that can handle tuple blocks: use ds_raw
    run_model(
        "FeaturelessDeepHalo",
        FeaturelessDeepHalo(
            num_items=dims_raw["J"],
            depth=2,
            width=max(8, dims_raw["J"]),
            optimizer="adam",
            lr=lr,
            batch_size=batch_size,
            epochs=50,
        ),
        ds_raw,
        batch_size,
    )

    # Run and evalutate the featurebased deepHalo model
    run_model(
        "FeatureBasedDeepHalo",
        FeatureBasedDeepHalo(
            num_items=dims_raw["J"],
            depth=2,
            width=16,
            heads=4,
            optimizer="adam",
            lr=lr,
            batch_size=batch_size,
            epochs=50,
        ),
        ds_raw,
        batch_size,
    )

    # Run and evalutate the Rumnet model (GPU version)
    # GPURUMnet: uses the raw dataset, but pass in the effective widths inferred from raw batches
    print("\nMODEL: GPURUMnet")
    try:
        m = GPURUMnet(
            num_products_features=dims_raw["dx_items"],
            num_customer_features=dims_raw["dx_shared"],
            width_eps_x=20,
            depth_eps_x=3,
            heterogeneity_x=10,
            width_eps_z=20,
            depth_eps_z=3,
            heterogeneity_z=10,
            width_u=20,
            depth_u=3,
            tol=0,
            optimizer="adam",
            lr=lr,
            batch_size=batch_size,
            epochs=5,
        )
        m.instantiate()
        m.fit(ds_raw)
        val = m.evaluate(ds_raw, batch_size=batch_size, mode="optim")
        print("  eval:", _as_float(val))
    except Exception as e:
        print("  FAILED:", repr(e))

    # Run and evalutate the Tastenet model
    # TasteNet: use raw dataset (it can concatenate tuple blocks internally),
    # but instantiate manually to the effective shared width to avoid the 13 vs 84 issue.
    print("\nMODEL: TasteNet")
    try:
        parametrization = [
            ["linear"] * dims_raw["dx_items"] for _ in range(dims_raw["J"])
        ]
        m = TasteNet(
            taste_net_layers=[],
            taste_net_activation="relu",
            items_features_by_choice_parametrization=parametrization,
            optimizer="adam",
            lr=lr,
            batch_size=batch_size,
            epochs=20,
        )
        m.instantiate(dims_raw["dx_shared"])
        m.instantiated = True
        m.fit(ds_raw)
        val = m.evaluate(ds_raw, batch_size=batch_size, mode="optim")
        print("  eval:", _as_float(val))
    except Exception as e:
        print("  FAILED:", repr(e))

    # Run and evalutate the Reslogit model
    # Models that require dense shared/items tensors: use ds_dense
    print("\nMODEL: ResLogit")
    try:
        m = ResLogit(
            intercept="item",
            n_layers=4,
            optimizer="adam",
            lr=lr,
            batch_size=batch_size,
            epochs=10,
        )
        m.instantiate(n_items=J, n_shared_features=dx_shared, n_items_features=dx_items)
        m.fit(ds_dense)
        val = m.evaluate(ds_dense, batch_size=batch_size, mode="optim")
        print("  eval:", _as_float(val))
    except Exception as e:
        print("  FAILED:", repr(e))

    print("\nDONE")


if __name__ == "__main__":
    main()
