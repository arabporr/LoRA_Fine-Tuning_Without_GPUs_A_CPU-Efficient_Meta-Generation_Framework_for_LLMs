import argparse

from src.preprocessing.metrics.WD import WD_run
from src.preprocessing.metrics.KL import KL_run
from src.preprocessing.metrics.JS import JS_run
from src.preprocessing.metrics.MMD import MMD_run


def calculate_distances(metric: str) -> None:
    if metric == "WD":
        WD_run()
    elif metric == "KL":
        KL_run()
    elif metric == "JS":
        JS_run()
    elif metric == "MMD":
        MMD_run()
    else:
        raise Exception("Invalid distance metric!")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("metric", choices=["WD", "KL", "JS", "MMD"])
    args = p.parse_args()
    calculate_distances(args.metric)
