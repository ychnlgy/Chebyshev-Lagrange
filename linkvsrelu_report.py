import json, statistics

data = json.load(open("linkvsrelu-visual.json"))

datatypes = ["accuracy", "sensitivity", "specificity", "f1"]

model_types = [
    "rand-32", "relu-32", "tanh-32", "linkact-32",
    "rand-64", "relu-64", "tanh-64", "linkact-64"
]

for model_type in model_types:
    print(" === %s ===" % model_type)
    for datatype in datatypes:
        scores = data[datatype][model_type]
        
        print("%s mean (std): %.4f (%.4f)" % (
            datatype,
            statistics.mean(scores) * 100,
            statistics.stdev(scores) * 100
        ))

import scipy.stats

tier32 = ["tanh-32", "linkact-32"]
tier64 = ["tanh-64", "linkact-64"]

for tier in [tier32, tier64]:
    print(tier)
    relu, link = tier
    reludata = data["f1"][relu]
    linkdata = data["f1"][link]

    print(scipy.stats.ttest_ind(reludata, linkdata, equal_var=False))
