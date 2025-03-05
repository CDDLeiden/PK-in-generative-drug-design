"""Hyperparameter optimization for QSPR models for A2AR pKi, VDSS, CL, and FU datasets"""

import argparse
import datetime
import json
import os
from os.path import join

import git
import pandas as pd
import stopit
from qsprpred.data import QSPRDataset
from qsprpred.logs.utils import enable_file_logger, export_conda_environment
from qsprpred.models import CrossValAssessor, QSPRModel, SklearnModel
from qsprpred.models.hyperparam_optimization import GridSearchOptimization
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

hyperparams_dict = {
    "RandomForestRegressor": {
        "n_estimators": [100, 300, 500, 1000],
        "max_depth": [5, 10, 20],
    },
    "KNeighborsRegressor": {"n_neighbors": [5, 7, 10]},
    "SVR": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"], "cache_size": [1000]},
    "PLSRegression": {"n_components": [5, 7, 10, 20, 50]},
}


@stopit.threading_timeoutable(default="timeout")
def model_training(model: QSPRModel, dataset: QSPRDataset):
    """Run cross-validation and hyperparameter optimization for a given model and dataset."""
    hyperparams = hyperparams_dict[model.alg.__name__]

    # remove n_components that are larger than the number of features
    if model.alg.__name__ == "PLSRegression":
        max_components = dataset.X.shape[1]
        hyperparams["n_components"] = [
            i for i in hyperparams["n_components"] if i <= max_components
        ]
        if hyperparams["n_components"] == []:
            hyperparams["n_components"] = [max_components]

    logger.info(f"Training model {model.name} started at {datetime.datetime.now()}.")
    gs = GridSearchOptimization(
        param_grid=hyperparams_dict[model.alg.__name__],
        model_assessor=CrossValAssessor(scoring="r2", round=7),
    )
    best_params = gs.optimize(model, dataset)
    logger.info(f"Best hyperparameters: {best_params} for model {model.name}.")
    logger.info(f"Training model {model.name} finished at {datetime.datetime.now()}.\n")

    model_result_df = gs.monitor.scores
    model_result_df["hyperparameters"] = gs.monitor.parameters
    model_result_df["assessment"] = "grid_search"
    model_result_df["fold_scores"] = model_result_df["fold_scores"].apply(
        lambda x: ";".join([str(i) for i in x])
    )
    model_result_df.to_csv(
        join(model.outDir, f"{model.name}_results.tsv"), sep="\t", index=False
    )

    return model_result_df


def main(qspr_dir: str, overwrite: bool, n_proc: int, timeout: int):
    """Main function to create models for all datasets in the qspr_dir.

    Args:
        qspr_dir (str): Path to the directory with the QSPR datasets.
        overwrite (bool): Overwrite existing models.
        n_proc (int): Number of processors to use.
        timeout (int): Timeout for model training in seconds.
    """
    dataset_folders = os.listdir(join(qspr_dir, "data"))
    dataset_folders = [
        folder
        for folder in dataset_folders
        if folder.startswith("A2AR")
        or folder.startswith("FU")
        or folder.startswith("VDSS")
        or folder.startswith("CL")
    ]
    result_df = pd.DataFrame(
        columns=[
            "aggregated_score",
            "fold_scores",
            "assessment",
            "hyperparameters",
            "dataset_settings",
            "model_path",
            "dataset_path",
        ]
    )
    for folder in dataset_folders:
        # load data settings from json
        with open(join(qspr_dir, "data", folder, "prep_settings.json"), "r") as f:
            data_settings = json.load(f)
        logger.info(f"Loading dataset {folder}.")
        dataset = QSPRDataset.fromFile(
            join(qspr_dir, "data", folder, f"{folder}_meta.json")
        )

        for alg in [PLSRegression, RandomForestRegressor, KNeighborsRegressor, SVR]:
            # skip if model already exists and overwrite is False
            model_name = f"{alg.__name__}_{dataset.name}"

            if not overwrite and os.path.exists(join(qspr_dir, "models", model_name)):
                logger.info(
                    f"Model {model_name} already exists and overwrite is False. Skipping.\n"
                )
                continue

            # Create the model
            model = SklearnModel(
                base_dir=join(qspr_dir, "models"),
                alg=alg,
                name=model_name,
                parameters=(
                    {"n_jobs": n_proc}
                    if alg.__name__ in ["RandomForestRegressor", "KNeighborsRegressor"]
                    else {}
                ),
            )
            # Train the model
            model_result_df = model_training(model, dataset, timeout=timeout)
            if isinstance(model_result_df, str) and model_result_df == "timeout":
                logger.info(
                    f"Model {model_name} training timed out after {timeout} seconds. Skipping.\n"
                )
                # remove model folder if it exists and is empty
                if os.path.exists(model.outDir) and not os.listdir(model.outDir):
                    logger.info(f"Removing empty model folder {model.outDir}")
                    os.rmdir(model.outDir)
                continue
            model_result_df["dataset_settings"] = [data_settings] * len(model_result_df)
            model_result_df["algorithm"] = alg.__name__
            model_result_df["model_path"] = model.metaFile
            model_result_df["dataset_path"] = dataset.metaFile
            result_df = pd.concat([result_df, model_result_df])
            result_df.to_csv(
                join(qspr_dir, "models", f"hyperparamopt_results_{now}.tsv"),
                sep="\t",
                index=False,
            )
    logger.info(f"Model creation finished at {datetime.datetime.now()}.")


def save_best_models_to_config(qspr_dir: str, config_file: str):
    """Save the best models to the config file."""
    model_files = os.listdir(join(qspr_dir, "models"))
    model_result_files = [
        f for f in model_files if f.startswith("hyperparamopt_results")
    ]
    model_results = pd.concat(
        [pd.read_csv(join(qspr_dir, "models", f), sep="\t") for f in model_result_files]
    )
    model_results["target"] = model_results["model_path"].apply(
        lambda x: os.path.basename(x).split("_")[1]
    )
    with open(config_file, "r") as f:
        config = json.load(f)

    for target in ["A2AR", "FU", "VDSS", "CL"]:
        best_model = (
            model_results[model_results["target"] == target]
            .sort_values(by="aggregated_score", ascending=False)
            .head(1)
        )
        best_model_path = best_model.model_path.values[0]
        best_data_path = best_model.dataset_path.values[0]

        config[f"{target}_DATA"] = os.path.basename(best_data_path).removesuffix(
            "_meta.json"
        )
        config[f"{target}_MODEL"] = os.path.basename(best_model_path).removesuffix(
            "_meta.json"
        )

    with open(config_file, "w") as f:
        json.dump(config, f, indent=4)


if __name__ == "__main__":
    # get data path and output directory from user args
    parser = argparse.ArgumentParser(
        description="Hyperparameter optimization for QSPR models"
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="config.json",
        help="Path to the configuration file",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing models",
    )
    parser.add_argument(
        "--n_proc",
        type=int,
        default=10,
        help="Number of processors to use",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60 * 60 * 8,
        help="Timeout for model training in seconds",
    )
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        config = json.load(f)

    # set data paths
    qspr_dir = join(config["BASE_DIR"], config["PROCESSED_DATA_DIR"], "QSPR")

    # save conda environment to outdir
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(join(qspr_dir, "models"), exist_ok=True)
    env_file = join(qspr_dir, f"models/conda_env_{now}.yml")
    export_conda_environment(env_file)

    # Set up logging
    logSettings = enable_file_logger(
        log_folder=f"{qspr_dir}/models",
        filename=f"HyperparamOpt_{now}.log",
        log_name=__name__,
        debug=False,
        disable_existing_loggers=False,
        init_data={
            "QSPR_DIR": join(config["PROCESSED_DATA_DIR"], "QSPR"),
            "N_PROC": args.n_proc,
            "OVERWRITE": args.overwrite,
            "TIMEOUT": args.timeout,
        },
    )
    logger = logSettings.log

    # Save the current git commit
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    logger.info(f"Git commit: {sha}")

    main(qspr_dir, args.overwrite, args.n_proc, args.timeout)

    save_best_models_to_config(qspr_dir, args.config_file)
