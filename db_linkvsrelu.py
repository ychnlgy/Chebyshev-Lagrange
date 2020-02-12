import collections, json

from db_adhc import main as single_main

import src

ACC = 0
SENS = 1
SPEC = 2
F1 = 3

@src.util.main
def main(repeats=30, seed0=5, outf="linkvsrelu.json"):

    repeats = int(repeats)
    seed0 = int(seed0)

    act_types = ["linkact", "relu", "rand", "tanh"]

    d_counts = [32, 64]

    out = [collections.defaultdict(list) for i in range(4)]

    for i in range(repeats):
        seed = seed0 + i
        for d in d_counts:
            for act_type in act_types:
                uid = "%s-%d" % (act_type, d)
                acc, sens, spec, f1 = single_main(act_type, d, seed)
                out[ACC][uid].append(acc)
                out[SENS][uid].append(sens)
                out[SPEC][uid].append(spec)
                out[F1][uid].append(f1)

    out = {
        "accuracy": dict(out[ACC]),
        "sensitivity": dict(out[SENS]),
        "specificity": dict(out[SPEC]),
        "f1": dict(out[F1])
    }
    json.dump(out, open(outf, "w"))
    print("Saved to link-vs-relu results to %s." % outf)

                
