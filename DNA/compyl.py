import tensorflow as tf
from tensorflow.keras.optimizers import Adam

def compile_model(model):
    
    optimizer = Adam(learning_rate=1e-3)
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            tf.keras.metrics.AUC(curve='PR', name='pr_auc'),
            tf.keras.metrics.AUC(name='roc_auc'),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall()
        ]
    )
    
    return model
