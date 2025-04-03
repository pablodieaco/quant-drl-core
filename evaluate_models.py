import argparse
import json
import re
import shutil
import time
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from loguru import logger

from quant_drl.configurations import get_complete_configuration
from quant_drl.utils.logging import setup_logger
from quant_drl.tester.tester import Tester


def extract_timestamp(model_name: str) -> str | None:
    match = re.search(r"_(\d{8}-\d{6})_", model_name)
    return match.group(1) if match else None


def detect_normalization(model_name: str) -> str:
    if "standard" in model_name:
        return "standard"
    elif "min_max" in model_name:
        return "min_max"
    return "unknown"


def parse_learning_rate(model_name: str) -> float:
    return float(model_name.split("lr_")[1].split("_")[0])


def parse_n_companies(model_name: str) -> int:
    return int(model_name.split("ncompanies_")[1].split("_")[0])

def parse_steps_override(step_list):
    override = {}
    for item in step_list:
        try:
            algo, steps = item.split("=")
            override[algo.upper()] = int(steps)
        except ValueError:
            logger.error(f"Formato inválido en steps_override: {item}")
    return override



def evaluate_all_models(args):
    logger.info(f"Cargando jerarquía desde: {args.hierarchy_path}")
    with open(args.hierarchy_path, "r") as f:
        models_data = json.load(f)
        hierarchy = models_data.get("gym_models", {})
        if not hierarchy:
            logger.error("No se encontró la clave 'gym_models' en el archivo de jerarquía.")
            return

    results = []
    total_models = sum(
        len(models)
        for algo in hierarchy.values()
        for models in algo.values()
    )

    logger.info(f"Modelos a evaluar: {total_models}")
    steps_override = parse_steps_override(args.steps_override)

    with tqdm(total=total_models, desc="Testing Models", unit="model") as pbar:
        for algo, features in hierarchy.items():
            for feature, model_names in features.items():
                for model_name in model_names:
                    normalization = detect_normalization(model_name)
                    n_companies = parse_n_companies(model_name)
                    learning_rate = parse_learning_rate(model_name)
                    timestamp = extract_timestamp(model_name)

                    logger.info(f"🔍 Testing {algo} | {feature} | {model_name}")

                    config = get_complete_configuration(len_portfolio=n_companies)
                    config["normalize"] = normalization

                    tester = Tester(config)
                    base_path = args.models_dir / algo / feature

                    if args.use_final_model:
                        model_file = "model_final.zip"
                        steps = None
                    else:
                        steps = steps_override.get(algo.upper(), 200_000)
                        model_file = f"{model_name}_{steps}_steps.zip"

                    try:
                        tester.load_model(base_path, model_file)
                    except Exception as e:
                        logger.warning(f"Error cargando modelo '{model_name}': {e}")
                        pbar.update(1)
                        continue

                    info_train, info_eval = tester.compare_train_eval(num_episodes=50)

                    for phase, info in zip(["train", "eval"], [info_train, info_eval]):
                        filtered_metrics = {
                            k: float(np.mean(v))
                            for k, v in info.items()
                            if not k.startswith("all_")
                        }

                        results.append({
                            "algorithm": algo,
                            "feature": feature,
                            "number_of_companies": n_companies,
                            "learning_rate": learning_rate,
                            "phase": phase,
                            "model_name": model_name,
                            "steps": steps if steps else "final",
                            "number_of_episodes": 50,
                            "timestamp": timestamp,
                            **filtered_metrics,
                        })

                    logger.success(f"Completado: {model_name}")
                    pbar.update(1)
                    time.sleep(0.1)

    df = pd.DataFrame(results)
    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.results_path, sep=";", decimal=",", index=False)
    logger.info(f"Resultados guardados en: {args.results_path}")

    if not args.no_move:
        timestamp_now = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_filename = f"hierarchy_{timestamp_now}.json"
        new_path = args.out_dir / new_filename
        args.out_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(args.hierarchy_path, new_path)
        logger.info(f"Archivo de jerarquía movido a: {new_path}")
    else:
        logger.info("Archivo de jerarquía no movido (flag --no_move activado)")

    logger.success("Evaluación completada con éxito.")



def parse_args():
    parser = argparse.ArgumentParser(description="Evaluación de modelos entrenados")
    parser.add_argument(
        "--hierarchy_path",
        type=Path,
        default=Path("models/metadata/hierarchy.json"),
        help="Ruta al archivo hierarchy.json",
    )
    parser.add_argument(
        "--results_path",
        type=Path,
        default=Path("results/evaluation_results.csv"),
        help="Ruta donde guardar el CSV con resultados",
    )
    parser.add_argument(
        "--models_dir",
        type=Path,
        default=Path("models"),
        help="Ruta a la carpeta de modelos",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("models/metadata/out"),
        help="Ruta de salida para la jerarquía movida",
    )
    parser.add_argument(
        "--no_move",
        action="store_true",
        help="No mover el archivo de jerarquía después de la evaluación",
    )
    parser.add_argument(
        "--use_final_model",
        action="store_true",
        help="Si se activa, carga 'model_final.zip' en vez de por steps",
    )
    parser.add_argument(
        "--steps_override",
        nargs="*",
        default=[],
        help="Override de steps por algoritmo en formato: PPO=800000 SAC=200000",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logger = setup_logger("logs/evaluate_models.log")

    args = parse_args()
    evaluate_all_models(args)