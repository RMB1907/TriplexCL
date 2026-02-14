import numpy as np
def compute_class_weights(y):
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(y)
    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y
    )
    
    return dict(zip(classes, weights))
