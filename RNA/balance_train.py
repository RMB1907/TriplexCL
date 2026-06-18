# balance_train.py

import numpy as np
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)


def balance_data(X, y, positive_factor=3, random_state=42):
    """
    Oversample positives by positive_factor and
    undersample negatives to match.
    """
    rng = np.random.default_rng(random_state)

    pos_mask = (y == 1)
    neg_mask = (y == 0)

    X_pos = X[pos_mask]
    y_pos = y[pos_mask]

    X_neg = X[neg_mask]
    y_neg = y[neg_mask]

    # Oversample positives
    X_pos_balanced = np.concatenate(
        [X_pos] * positive_factor,
        axis=0
    )
    y_pos_balanced = np.concatenate(
        [y_pos] * positive_factor,
        axis=0
    )

    # Match number of negatives to positives
    target_negatives = len(y_pos_balanced)

    if target_negatives > len(y_neg):
        raise ValueError(
            f"Need {target_negatives} negatives but only "
            f"{len(y_neg)} available."
        )

    neg_indices = rng.choice(
        len(y_neg),
        size=target_negatives,
        replace=False
    )

    X_neg_balanced = X_neg[neg_indices]
    y_neg_balanced = y_neg[neg_indices]

    # Combine
    X_balanced = np.concatenate(
        [X_pos_balanced, X_neg_balanced],
        axis=0
    )
    y_balanced = np.concatenate(
        [y_pos_balanced, y_neg_balanced],
        axis=0
    )

    # Shuffle
    shuffle_idx = rng.permutation(len(y_balanced))

    return (
        X_balanced[shuffle_idx],
        y_balanced[shuffle_idx]
    )


def train_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
):


    callbacks = [
        EarlyStopping(
            monitor="val_pr_auc",
            mode="max",
            patience=20,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor="val_pr_auc",
            mode="max",
            factor=0.5,
            patience=10
        ),
        ModelCheckpoint(
            "output/rna_balanced.keras",
            monitor="val_pr_auc",
            mode="max",
            save_best_only=True,
            verbose=1
        )
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=200,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    return history