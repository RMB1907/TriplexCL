def run_pipeline(posf, negf):

    sequences, labels = load.load_data(posf, negf)

    X = fea.generate_features(sequences)

    X_train, X_val, X_test, y_train, y_val, y_test = split.stratified_split(X, labels)

    model_instance = model.build_model()

    model_instance = compyl.compile_model(model_instance)

    train.train_model(model_instance, X_train, y_train, X_val, y_val)

    ival.evaluate(model_instance, X_test, y_test)

    return model_instance
