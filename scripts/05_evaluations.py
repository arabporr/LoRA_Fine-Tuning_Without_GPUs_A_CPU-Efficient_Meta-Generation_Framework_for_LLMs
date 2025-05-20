import argparse

from src.evaluation.weights_biases_differences import numerical_evaluation
from src.evaluation.outputs_differences import outputs_evaluation



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generating evaluation results for the selected metric and model")
    parser.add_argument("-metric", choices=["WD", "KL", "JS", "MMD"])
    parser.add_argument(
        "-model", choices=["base_version", "normalized_version", "mlp_version"]
    )
    args = parser.parse_args()
    numerical_evaluation(metric=args.metric, model=args.model)
    outputs_evaluation(metric=args.metric, model=args.model)
