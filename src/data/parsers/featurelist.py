import os, json

FEATURE_FILE = "dfd_features2.json"

def get(datadir):
    p = os.path.join(datadir, FEATURE_FILE)
    return json.load(open(p))
