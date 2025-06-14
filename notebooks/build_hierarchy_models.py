import json
import os
import re
from datetime import datetime


def extract_timestamp(model_name):
    match = re.search(r"(\d{8}-\d{6})", model_name)
    if match:
        return datetime.strptime(match.group(1), "%Y%m%d-%H%M%S")
    return None


def load_existing_hierarchy(json_file):
    if os.path.exists(json_file):
        with open(json_file, "r") as file:
            try:
                return json.load(file).get("gym_models", {})
            except json.JSONDecodeError:
                return {}
    return {}


def update_model_hierarchy(
    root_dir,
    json_file,
    min_date=None,
    selected_algorithms=["PPO", "SAC", "DDPG", "TD3"],
    selected_features=["NoFeature", "CNN", "LSTM", "Transformer"],
):
    hierarchy = load_existing_hierarchy(json_file)
    print("Existing hierarchy loaded:", hierarchy)

    for algo in os.listdir(root_dir):
        if algo in selected_algorithms:
            algo_path = os.path.join(root_dir, algo)
            if os.path.isdir(algo_path):
                if algo not in hierarchy:
                    hierarchy[algo] = {}

                for feature in os.listdir(algo_path):
                    if feature in selected_features:
                        feature_path = os.path.join(algo_path, feature)
                        if os.path.isdir(feature_path):
                            if feature not in hierarchy[algo]:
                                hierarchy[algo][feature] = []

                            existing_models = set(hierarchy[algo][feature])
                            for model in os.listdir(feature_path):
                                timestamp = extract_timestamp(model)
                                if timestamp and (
                                    min_date is None or timestamp >= min_date
                                ):
                                    if model not in existing_models:
                                        hierarchy[algo][feature].append(model)

    return hierarchy


if __name__ == "__main__":
    selected_algorithms = ["TD3"]
    selected_features = ["CNNLSTM", "LSTM", "NoFeature"]
    root_directory = "../models"
    json_file_path = "../models/metadata/hierarchy.json"
    min_date_str = "20250416-000000"
    min_date = datetime.strptime(min_date_str, "%Y%m%d-%H%M%S")

    updated_hierarchy = update_model_hierarchy(
        root_directory, json_file_path, min_date, selected_algorithms, selected_features
    )

    print("Updated hierarchy:", updated_hierarchy)

    with open(json_file_path, "w") as json_file:
        json.dump({"gym_models": updated_hierarchy}, json_file, indent=2)

    print("JSON actualizado exitosamente: model_hierarchy.json")
