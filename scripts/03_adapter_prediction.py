import argparse

from src.adapter_prediction.adapters_generator import generate_adapters


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predicting adapters for the selected metric and model")
    parser.add_argument("-metric", choices=["WD", "KL", "JS", "MMD"])
    parser.add_argument(
        "-model", choices=["base_version", "normalized_version", "mlp_version"]
    )
    args = parser.parse_args()

    generate_adapters(metric=args.metric, model=args.model)
