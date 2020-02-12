import numpy, torch, tqdm

import src

DATADIR = "../../data/"

def fit(model, X, Y):
    model.train()
    scorer = src.tensortools.Scorer(verbose=False)
    avg = src.util.MovingAverage(momentum=0.95)

    Y_miu = Y.float().mean().item()
    print("Y mean: %.3f" % Y_miu)

    std = X.std(dim=0).unsqueeze(0)
    dataloader = src.tensortools.dataset.create_loader([X, Y], batch_size=32, shuffle=True)
    
    lossf = torch.nn.CrossEntropyLoss()
    optim = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

    epochs = 300
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

    with tqdm.tqdm(range(epochs), ncols=80) as bar:
        for epoch in bar:
            for x, y in dataloader:
                
                yh = model(x)
                loss = lossf(yh, y)
                optim.zero_grad()
                loss.backward()
                optim.step()

                scorer.update(yh, y)
                avg.update(loss.item())
                bar.set_description("Loss|balanced-acc|acc: %.5f|%.5f|%.5f" % (avg.peek(), *scorer.peek()))

            sched.step()

def predict(model, X):
    model.eval()
    out = []
    with torch.no_grad():
        return model(X)

def select_features(X_in, Y, U, X_valid, seed):
    folds = 5
    print("Selecting features over %d training folds." % folds)
    threshold = 0.001
    D = X_in.size(1)
    weights = torch.ones(D).byte()
    for (X, Y), (X_test, Y_test) in src.cross_validation.k_fold(U.numpy(), X_in.numpy(), Y.numpy(), k=folds, seed=seed):
        X, Y, X_test, Y_test = tuple(map(torch.from_numpy, [X, Y, X_test, Y_test]))
        X = X.float()
        X_test = X_test.float()
        
        model = torch.nn.Linear(D, 2)
        fit(model, X, Y)
        weight_mask = model.weight[0].abs()/model.weight.abs().sum() > threshold
        print("Selected %d features." % weight_mask.long().sum().item())
        weights &= weight_mask
    return X_in[:,weights], X_valid[:,weights]

class Branches(torch.nn.ModuleList):

    def forward(self, X):
        if self.training:
            return sum(mod(X) for mod in self)/len(self)
        else:
            return sum(src.tensortools.onehot(mod(X).max(dim=1)[1], 2) for mod in self)

class Random(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1, 2))

    def forward(self, X):
        return self.w.repeat(len(X), 1)

def remove_columns_with_any_nans(X):
    I = ~numpy.isfinite(X)
    I = I.astype(int).sum(axis=0) > 0
    return X[:,~I]

@src.util.main(__name__)
def main(act_type, d, SEED=5):
    SEED = int(SEED)
    (
        (X_db_am, Y_db_am, U_db_am),
        (X_ha_am, Y_ha_am, U_ha_am),
        (X_ha_ca, Y_ha_ca, U_ha_ca),
        (X_uf_am, Y_uf_am, U_uf_am, S_uf_am)
    ) = src.data.adhc.get(DATADIR)
    
    K_FOLDS = 10
    
    d = int(d)
    
    if SEED is not None:
        torch.manual_seed(SEED)

    U = U_db_am
    X = numpy.concatenate([U_db_am.reshape(-1, 1), X_db_am], axis=1)
    Y = Y_db_am

    print(Y.astype(int).sum(), len(Y))
    
    print("Shape before removing columns with nans:", X.shape)
    X = remove_columns_with_any_nans(X)
    print("Shape after:", X.shape)
    
    total_scorer = src.tensortools.Scorer()

    for fold_i, (data, test) in enumerate(src.cross_validation.k_fold(U, X, Y, k=K_FOLDS, seed=SEED)):

        print(" === Fold %d ===" % fold_i)

        X, Y = data
        U = X[:,0]
        X = X[:,1:]
        
        X_valid, Y_valid = test
        X_valid = X_valid[:,1:]

        J = (X.std(axis=0) > 1e-8)
        X = X[:,J]
        X_valid = X_valid[:,J]
        
        X, X_valid = src.cross_validation.standard_scale(X, X_valid)

        X, Y, U, X_valid, Y_valid = tuple(map(
            torch.from_numpy,
            [X, Y, U, X_valid, Y_valid]
        ))

        X = X.float()
        Y = Y.long()
        U = U.long()
        X_valid = X_valid.float()
        Y_valid = Y_valid.long()

        D = X.size(1)

        print("Number of features: %d" % D)

        if act_type == "rand":
            model = Random()
        else:

            create_activation = {
                "relu": lambda: torch.nn.ReLU(),
                "tanh": lambda: torch.nn.Tanh(),
                "linkact": lambda: src.modules.polynomial.LinkActivation(2, d, n_degree=3, zeros=True),
                "regact": lambda: src.modules.polynomial.RegActivation(2, d, n_degree=3, zeros=True),
            }[act_type]
            
            model = torch.nn.Sequential(

                torch.nn.Linear(D, d),
                
                src.modules.ResNet(
                    src.modules.ResBlock(
                        torch.nn.Sequential(
                            create_activation(),
                            torch.nn.Linear(d, d),
                            create_activation(),
                            torch.nn.Linear(d, d),
                        ),
                    ),
                    src.modules.ResBlock(
                        torch.nn.Sequential(
                            create_activation(),
                            torch.nn.Linear(d, d),
                            create_activation(),
                            torch.nn.Linear(d, d),
                        ),
                    ),
                ),
                torch.nn.Linear(d, 2)
            )
        
        print("Parameters:", sum(torch.numel(p) for p in model.parameters() if p.requires_grad))

        fit(model, X, Y)
        
        Yh_valid = predict(model, X_valid)

        scorer = src.tensortools.Scorer()
        scorer.update(Yh_valid, Y_valid)
        total_scorer.update(Yh_valid, Y_valid)

        print("[Sample] Fold balanced-acc|acc|F1: %.5f|%.5f|%.5f" % (*scorer.peek(), scorer.calc_f1()))

    bal, acc = total_scorer.peek()
    f1 = total_scorer.calc_f1()
    print("[Final] Total balanced-acc|acc|F1: %.5f|%.5f|%.5f" % (bal, acc, f1))
    return acc, total_scorer.calc_sens(), total_scorer.calc_spec(), f1
    
