import os, json

import src

@src.util.main
def main(datanames, tiers, noise, N, repeat):

    datanames = datanames.split(",")
    tiers = tiers.split(",")
    noise = float(noise)
    N = int(N)
    repeat = int(repeat)

    for _ in range(repeat):
        for dataname in datanames:
            dataset = src.toy.datasets.search(dataname)
            for tier in tiers:
                D = dataset.get_num_true_features()
                
                savedir = "results/%s/" % dataname
                print("Output folder: %s" % savedir)
                if not os.path.isdir(savedir):
                    os.makedirs(savedir)
                i = 0
                make_savefile = lambda i: os.path.join(savedir, "%dN-%.2fn-%s-R%d.png" % (N, noise, tier, i))
                savefile = make_savefile(i)
                while os.path.isfile(savefile):
                    i += 1
                    savefile = make_savefile(i)
                src.toy.main(
                    dataname,
                    savefile,
                    tier=tier,
                    D=D,
                    N=N,
                    noise = noise
                )
                
