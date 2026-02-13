def evaluate(model, X_test, y_test):
    results = model.evaluate(X_test, y_test, verbose=0)
    print(dict(zip(model.metrics_names, results)))
