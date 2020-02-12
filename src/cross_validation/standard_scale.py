import sklearn.preprocessing

def standard_scale(X_train, X_valid):
    scaler = sklearn.preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)
    return X_train, X_valid
