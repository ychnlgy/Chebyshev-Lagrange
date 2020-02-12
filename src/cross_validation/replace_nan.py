import numpy

def with_mean(X_train, X_valid):
    assert len(X_train.shape) == 2
    keep = [
        _calc_mean_feature(X_train, X_valid, i)
        for i in range(X_train.shape[1])
    ]
    train = numpy.stack([k[0] for k in keep], axis=1)
    valid = numpy.stack([k[1] for k in keep], axis=1)
    return train, valid

def with_zero(X_train, X_valid):
    X_train[~numpy.isfinite(X_train)] = 0
    X_valid[~numpy.isfinite(X_valid)] = 0
    return X_train, X_valid

# === PRIVATE ===

def _calc_mean_feature(X, X_valid, i):
    "Replaces nan in X and X_valid with the mean of X (not X_valid)."
    column = X[:,i].copy()
    validc = X_valid[:,i].copy()
    finite = numpy.isfinite(column)
    check = column[finite]
    if not len(check):
        return numpy.zeros(len(X)), numpy.zeros(len(X_valid))
    else:
        miu = check.mean()
        column[~finite] = miu
        validc[~numpy.isfinite(validc)] = miu
        assert numpy.isfinite(column).all()
        return column, validc
