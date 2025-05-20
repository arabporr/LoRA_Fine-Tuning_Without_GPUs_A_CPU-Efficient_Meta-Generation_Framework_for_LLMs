import argparse

from src.preprocessing.dataset_tokenizer import tokenization_handler
from src.preprocessing.distance_calculator import calculate_distances


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenizing and calculating distances based on selected metric")
    parser.add_argument("-metric", choices=["WD", "KL", "JS", "MMD"])
    args = parser.parse_args()

    tokenization_handler()
    calculate_distances(args.metric)
    
