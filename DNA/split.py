from sklearn.model_selection import train_test_split

def stratified_split(X, y):
    
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )
    
    # Second split: train vs val
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=0.2,
        stratify=y_temp,
        random_state=42
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test
