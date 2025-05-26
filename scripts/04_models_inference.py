import argparse

from src.inference.load_and_inference import generate_outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generating model outputs for test set of selected dataset for base, fine tuned, and selected metric-model predicted models")
    parser.add_argument("-metric", choices=["WD", "KL", "JS", "MMD"])
    parser.add_argument(
        "-model", choices=["base_version", "normalized_version", "mlp_version"]
    )
    parser.add_argument("-dataset_index", type=int)
    args = parser.parse_args()

    generate_outputs(metric=args.metric, model=args.model,
                     dataset_index=args.dataset_index)
