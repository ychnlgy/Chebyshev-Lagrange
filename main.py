import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")

import src

HELP = '''

ERROR: Option %s is not recognized.

Please select from the following experiments:
'''

@src.util.main
def main(experiment, **kwargs):

    options = {
        "spectralgraph-mnist": src.experiments.spectralgraph_mnist,
    }

    if experiment not in options:
        print(HELP % experiment)
        for key, val in sorted(options.items()):
            print(key, val.main.__doc__)

        raise SystemExit(1)

    options[experiment].main(**kwargs)

    
            
