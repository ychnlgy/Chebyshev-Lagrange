from . import datasets, train, visual, polynomial
import src

def get_type(tier):
    return getattr(polynomial, "Tier_%s" % tier)

def main(
    dataname,
    results_savepath,
    tier,
    D,
    N,
    noise
):
    
    dataset, testset, viewset, lr = datasets.create_from_str(dataname, N, D, noise=noise)
    
    D = dataset[0].size(1)
    Model = get_type(tier)
    model = Model(D)
    print("Parameters: %d" % src.tensortools.paramcount(model))

    trainloss, testloss = train.main(dataset, testset, model, lr)
    print("Training/testing loss: %.5f/%.5f" % (trainloss, testloss))

    visual.visualize(model, results_savepath, *viewset, testloss, tier, noise)
