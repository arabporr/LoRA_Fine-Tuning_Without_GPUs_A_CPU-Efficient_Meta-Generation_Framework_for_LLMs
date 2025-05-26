import os

# Assuming that we want to put everything in the current directory
current_dir = os.getcwd()


all_distance_metrics = ["WD", "KL", "JS", "MMD"]
all_models = ["base_version", "normalized_version", "mlp_version"]

# Data Paths
parent_dir_data = os.path.join(current_dir, "data")
if not os.path.exists(parent_dir_data):
    os.makedirs(parent_dir_data)


raw_input_dir = os.path.join(parent_dir_data, "raw_input")
if not os.path.exists(raw_input_dir):
    os.makedirs(raw_input_dir)

raw_datasets_dir = os.path.join(raw_input_dir, "datasets")
if not os.path.exists(raw_datasets_dir):
    os.makedirs(raw_datasets_dir)

raw_adapters_dir = os.path.join(raw_input_dir, "adapters")
if not os.path.exists(raw_adapters_dir):
    os.makedirs(raw_adapters_dir)

all_adapters_file_location = os.path.join(raw_adapters_dir, "all_adapters.pt")


preprocessed_dir = os.path.join(parent_dir_data, "preprocessed")
if not os.path.exists(preprocessed_dir):
    os.makedirs(preprocessed_dir)

tokenized_datasets_dir = os.path.join(preprocessed_dir, "tokenized_datasets")
if not os.path.exists(tokenized_datasets_dir):
    os.makedirs(tokenized_datasets_dir)

distances_dir = os.path.join(preprocessed_dir, "distances")
if not os.path.exists(distances_dir):
    os.makedirs(distances_dir)

for metric in all_distance_metrics:
    distances_metric_dir = os.path.join(distances_dir, metric)
    if not os.path.exists(distances_metric_dir):
        os.makedirs(distances_metric_dir)


coefficients_dir = os.path.join(parent_dir_data, "coefficients")
if not os.path.exists(coefficients_dir):
    os.makedirs(coefficients_dir)

for metric in all_distance_metrics:
    coefficients_metric_dir = os.path.join(coefficients_dir, metric)
    if not os.path.exists(coefficients_metric_dir):
        os.makedirs(coefficients_metric_dir)

    for model in all_models:
        coefficients_metric_model_dir = os.path.join(
            coefficients_metric_dir, model)
        if not os.path.exists(coefficients_metric_model_dir):
            os.makedirs(coefficients_metric_model_dir)

predicted_adapters_dir = os.path.join(parent_dir_data, "predicted_adapters")
if not os.path.exists(predicted_adapters_dir):
    os.makedirs(predicted_adapters_dir)

for metric in all_distance_metrics:
    predicted_adapters_metric_dir = os.path.join(
        predicted_adapters_dir, metric)
    if not os.path.exists(predicted_adapters_metric_dir):
        os.makedirs(predicted_adapters_metric_dir)

    for model in all_models:
        predicted_adapters_metric_model_dir = os.path.join(
            predicted_adapters_metric_dir, model
        )
        if not os.path.exists(predicted_adapters_metric_model_dir):
            os.makedirs(predicted_adapters_metric_model_dir)


# Results Paths
parent_dir_results = os.path.join(current_dir, "results")
if not os.path.exists(parent_dir_results):
    os.makedirs(parent_dir_results)

models_outputs_dir = os.path.join(
    parent_dir_results, "models_generated_outputs")
if not os.path.exists(models_outputs_dir):
    os.makedirs(models_outputs_dir)

base_model_outputs_dir = os.path.join(
    models_outputs_dir, "Foundation_Model_Base")
if not os.path.exists(base_model_outputs_dir):
    os.makedirs(base_model_outputs_dir)

fine_tuned_model_outputs_dir = os.path.join(
    models_outputs_dir, "GPU_Fine_Tuned")
if not os.path.exists(fine_tuned_model_outputs_dir):
    os.makedirs(fine_tuned_model_outputs_dir)

for metric in all_distance_metrics:
    models_outputs_metric_dir = os.path.join(models_outputs_dir, metric)
    if not os.path.exists(models_outputs_metric_dir):
        os.makedirs(models_outputs_metric_dir)

    for model in all_models:
        models_outputs_metric_model_dir = os.path.join(
            models_outputs_metric_dir, model)
        if not os.path.exists(models_outputs_metric_model_dir):
            os.makedirs(models_outputs_metric_model_dir)


# numerical_results_dir = os.path.join(
#     parent_dir_results, "evaluation_numerical_results")
# if not os.path.exists(numerical_results_dir):
#     os.makedirs(numerical_results_dir)

# for metric in all_distance_metrics:
#     numerical_results_metric_dir = os.path.join(numerical_results_dir, metric)
#     if not os.path.exists(numerical_results_metric_dir):
#         os.makedirs(numerical_results_metric_dir)

#     for model in all_models:
#         numerical_results_metric_model_dir = os.path.join(
#             numerical_results_metric_dir, model
#         )
#         if not os.path.exists(numerical_results_metric_model_dir):
#             os.makedirs(numerical_results_metric_model_dir)


outputs_results_dir = os.path.join(
    parent_dir_results, "evaluation_outputs_results")
if not os.path.exists(outputs_results_dir):
    os.makedirs(outputs_results_dir)

# for metric in all_distance_metrics:
#     outputs_results_metric_dir = os.path.join(outputs_results_dir, metric)
#     if not os.path.exists(outputs_results_metric_dir):
#         os.makedirs(outputs_results_metric_dir)

#     for model in all_models:
#         outputs_results_metric_model_dir = os.path.join(
#             outputs_results_metric_dir, model
#         )
#         if not os.path.exists(outputs_results_metric_model_dir):
#             os.makedirs(outputs_results_metric_model_dir)
