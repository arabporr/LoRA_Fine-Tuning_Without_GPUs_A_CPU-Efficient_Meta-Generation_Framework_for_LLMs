import datetime


distance = "WD"  # WD, KL, JS, or MMD
sub_variant = "fewshot"  # nonlearnable, normalized, learnable, or fewshot
experiment_name = f"{distance}/{sub_variant}/"  # Folder path in results
experiment_time = datetime.datetime.now().__str__()
run_number = 1  # defult is -1, for testing
experiment_info = f"This is run number {run_number} at {experiment_time}, here we are using {distance} distance function and {sub_variant} sub-variant."

file_path = f"experiment_{run_number}.txt"

with open(file_path, "w") as file:
    file.write(f"Experiment info:\n{experiment_info}")


""" note: 
    1: I have set the distance function in the distances file! please go there and make sure
that that function and its is_symmetric value are correct! 
    2: check the distance processing file and check if everything is right too.
"""
