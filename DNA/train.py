from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from weights import compute_class_weights

def train_model(model, X_train, y_train, X_val, y_val):
    
    # Compute and clean class weights
    cw = compute_class_weights(y_train)
    class_weight = {int(k): float(v) for k, v in cw.items()}
    
    callbacks = [
        EarlyStopping(
            monitor='val_pr_auc',
            mode='max',
            patience=20,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_pr_auc',
            mode='max',
            factor=0.5,
            patience=10
        ),
        ModelCheckpoint(
            "best_dna.keras",
            monitor="val_pr_auc",
            mode="max",
            save_best_only=True,
            verbose=1
        )
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=200,
        batch_size=32,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1
    )
    
    return history
