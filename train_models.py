import argparse
import os
import time
from datetime import datetime
from pathlib import Path
from itertools import product
import sys

from loguru import logger

from quant_drl.configurations import get_companies
from quant_drl.trainer.trainer import Trainer
from quant_drl.utils.hierarchy_builder import update_model_hierarchy, save_hierarchy
from quant_drl.utils.logging import setup_logger

def parse_args():
    parser = argparse.ArgumentParser(description="Entrenamiento masivo de modelos DRL")

    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=["SAC"],
        choices=["DDPG", "TD3", "SAC", "PPO"],
    )
    parser.add_argument("--features", nargs="+", default=["CNNLSTM"])
    parser.add_argument("--learning_rates", nargs="+", type=float, default=[1e-4])
    parser.add_argument("--n_companies", nargs="+", type=int, default=[10])

    parser.add_argument("--reward_type", type=str, default="log_reward")
    parser.add_argument("--initial_capital", type=float, default=10000)
    parser.add_argument(
        "--sectors", nargs="+", help="Filtra por sectores (opcional)", default=None
    )

    parser.add_argument("--logs_dir", type=str, default="logs/")
    parser.add_argument("--save_dir", type=str, default="models/")
    parser.add_argument(
        "--update_hierarchy",
        action="store_true",
        help="Actualizar el archivo hierarchy.json después de los experimentos",
    )
    parser.add_argument(
        "--trading_cost",
        type=float,
        default=0.0025,
        help="Coste de trading por operación",
    )
    parser.add_argument(
        "--window_length",
        type=int,
        default=50,
        help="Longitud de la ventana para el entrenamiento",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Número de pasos para el entrenamiento",
    )
    parser.add_argument(
        "--checkpoint_freq",
        type=int,
        default=5e4,
        help="Frecuencia de guardado de los checkpoints",
    )
    parser.add_argument(
        "--lstm_layers",
        type=int,
        default=4,
        help="Número de capas LSTM para el modelo",
    )
    parser.add_argument(
        "--length_train_data",
        type=int,
        default=10,
        help="Longitud en años de los datos de entrenamiento",
    )
    parser.add_argument(
        "--length_eval_data",
        type=int,
        default=4,
        help="Longitud en años de los datos de evaluación",
    )
    parser.add_argument(
        "--end_date_year",
        type=int,
        default=2022,
        help="Año de finalización de los datos",
    )
    
    return parser.parse_args()


def main():
    """Función principal que configura y lanza los experimentos."""
    args = parse_args()

    total_experiments = list(
        product(
            args.algorithms,
            args.features,
            args.learning_rates,
            args.n_companies,
        )
    )

    logger.info(f"Ejecutando {len(total_experiments)} experimentos...\n")

    for algo, feature, lr, n_comp in total_experiments:
        companies_abv, companies_names = get_companies(
            n=n_comp, sectors_filter=args.sectors, shuffle=False
        )

        config = {
            "reward_type": args.reward_type,
            "feature_index": 0,
            "initial_capital": args.initial_capital,
            "end_date": datetime(args.end_date_year, 12, 31),
            "length_train_data": args.length_train_data,
            "length_eval_data": args.length_eval_data,
            "features": ["Close", "High", "Low", "Open"],
            "indicators": [
                "SMA",
                "EMA",
                "RSI",
                "MACD",
                "Bollinger_High",
                "Bollinger_Low",
                "ATR",
            ],
            "normalize": "standard",
            "trading_cost": args.trading_cost,
            "window_length": args.window_length,
            "steps": args.steps,
            "scale_rewards": True,
            "total_timesteps": 1e6 if algo == "PPO" else 3e5,
            "checkpoint_freq": args.checkpoint_freq,
            "learning_rate": lr,
            "lstm_layers": args.lstm_layers,
            "algorithm": algo,
            "feature": feature,
            "companies": [
                {"abv": abv, "name": name}
                for abv, name in zip(companies_abv, companies_names)
            ],
        }

        logger.info(
            f"Ejecutando experimento con {n_comp} empresas, algoritmo {algo}, feature {feature}, y tasa de aprendizaje {lr}..."
        )
        logger.info(f"Configuración: {config}")

        checkpoint_time = time.time()

        trainer = Trainer(
            config,
            generate_default_name=True,
            run=True,
            logs_dir=args.logs_dir,
            save_dir=args.save_dir,
        )

        logger.info(
            f"Experimento completado. Tiempo de ejecución: {time.time() - checkpoint_time:.2f} segundos"
        )
        logger.info("#" * 50)
    
    if args.update_hierarchy:
        logger.info("Actualizando archivo hierarchy.json...")

        hierarchy_json_path = Path("models/metadata/hierarchy.json")
        models_root = Path("models")
        timestamp_now = datetime.now().strftime("%Y%m%d-%H%M%S")
        min_date = datetime.strptime(timestamp_now, "%Y%m%d-%H%M%S")

        hierarchy = update_model_hierarchy(
            root_dir=str(models_root),
            json_file=str(hierarchy_json_path),
            min_date=min_date,
            selected_algorithms=args.algorithms,
            selected_features=args.features,
        )

        save_hierarchy(str(hierarchy_json_path), hierarchy)
        logger.success("Archivo hierarchy.json actualizado.")

if __name__ == "__main__":
    logger = setup_logger("logs/train.log")

    main()
