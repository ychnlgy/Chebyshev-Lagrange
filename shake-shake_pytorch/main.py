import os, sys

def main(repeat):
    for repeat in range(repeat):
        for mnist in [0, 1]:
            for act_type in ["linkact", "regact", "relu"]:
                os.system(
                    "python3 train.py --mnist %d --act_type %s --use_shakeshake %d --epochs 100" % (
                        mnist, act_type, 1-mnist
                    )
                )

if __name__ == "__main__":

    repeat = int(sys.argv[1])
    main(repeat)
