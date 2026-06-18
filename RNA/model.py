from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

def build_model():
    
    input_layer = Input(shape=(90, 1))
    
    x1 = Conv1D(32, 3, padding='same', activation='relu')(input_layer)
    x2 = Conv1D(32, 5, padding='same', activation='relu')(input_layer)
    x3 = Conv1D(32, 7, padding='same', activation='relu')(input_layer)
    
    concat = Concatenate(axis=-1)([x1, x2, x3])
    
    pool = MaxPooling1D(pool_size=2)(concat)
    
    lstm = LSTM(64, return_sequences=False)(pool)
    
    dense = Dense(64, activation='relu')(lstm)
    drop = Dropout(0.5)(dense)
    
    output = Dense(1, activation='sigmoid')(drop)
    
    model = Model(inputs=input_layer, outputs=output)
    
    return model

def compile_model(model):
    
    optimizer = Adam(learning_rate=1e-3)
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            tf.keras.metrics.AUC(curve='PR', name='pr_auc'),
            tf.keras.metrics.AUC(name='roc_auc'),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
        ]
    )
    
    return model

def binary_focal_loss(gamma=2.0, alpha=0.75):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)

        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        pt = tf.where(
            tf.equal(y_true, 1),
            y_pred,
            1.0 - y_pred
        )

        alpha_t = tf.where(
            tf.equal(y_true, 1),
            alpha,
            1.0 - alpha
        )

        focal_weight = alpha_t * tf.pow(1.0 - pt, gamma)

        return tf.reduce_mean(
            -focal_weight * tf.math.log(pt)
        )

    return loss


def compile_focal(model, lr=1e-3, gamma=2.0, alpha=0.75):

    optimizer = Adam(learning_rate=lr)
    
    model.compile(
        optimizer=optimizer,
        loss=binary_focal_loss(
            gamma=gamma,
            alpha=alpha
        ),
        metrics=[
            tf.keras.metrics.AUC(curve='PR', name='pr_auc'),
            tf.keras.metrics.AUC(name='roc_auc'),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
        ]
    )

    return model
