import argparse
import datetime
import json
import logging
import os
from itertools import product
from os.path import join

import git
import pandas as pd
from qsprpred.data import QSPRDataset, RandomSplit
from qsprpred.data.descriptors.sets import SmilesDesc
from qsprpred.extra.gpu.models.chemprop import ChempropModel
from qsprpred.logs import setLogger
from qsprpred.logs.utils import enable_file_logger, export_conda_environment
from qsprpred.models import CrossValAssessor, EarlyStoppingMode, QSPRModel
from qsprpred.models.hyperparam_optimization import GridSearchOptimization
from s02_hyperparam_optimization import save_best_models_to_config


def dataset_preparation(
    data_path_a2ar: str,
    data_path_pk: str,
    output_dir: str,
    seed: int = 42,
    overwrite: bool = False,
    target_prop: str = "A2AR",
):
    """Prepare datasets for ChemProp models

    Prepares datasets for the A2AR, CL, FU, and VDSS target properties
    by transforming the target properties and setting SMILES as
    the compound representation.

    Args:
        data_path_a2ar (str): Path to the A2AR dataset.
        data_path_pk (str): Path to the PK dataset.
        output_dir (str): Path to the output directory.
        seed (int, optional): Random seed. Defaults to 42.
        n_proc (int, optional): Number of processors to use. Defaults to 5.
        overwrite (bool, optional): Overwrite existing datasets. Defaults to False.
        target_prop (str, optional): Target property to optimize. Defaults to "A2AR".
    """
    # Define the transformers for the target properties
    transformer_dict = {
        "CL": lambda x: (__import__("numpy").log10(x)),
        "FU": lambda x: (__import__("numpy").sqrt(x)),
        "VDSS": lambda x: (__import__("numpy").log10(x)),
        "A2AR": None,
    }

    assert (
        target_prop in transformer_dict.keys()
    ), f"Invalid target property {target_prop}"

    data_path = data_path_a2ar if target_prop == "A2AR" else data_path_pk

    # Skip if dataset already exists
    if (
        os.path.exists(join(output_dir, "data", f"{target_prop}_chemprop"))
        and not overwrite
    ):
        logger.info(f"Dataset for {target_prop} already exists. Skipping. \n")
    else:
        # Create dataset
        dataset = QSPRDataset.fromTableFile(
            name=f"{target_prop}_chemprop",
            filename=data_path,
            store_dir=f"{output_dir}/data",
            target_props=[
                {
                    "name": target_prop,
                    "task": "REGRESSION",
                    "transformer": transformer_dict[target_prop],
                }
            ],
            random_state=seed,
        )

        # calculate compound features and split dataset into train and test
        dataset.prepareDataset(
            split=RandomSplit(test_fraction=0.2, seed=seed),
            feature_calculators=[SmilesDesc()],
            recalculate_features=True,
        )
        dataset.save()
        logger.info(f"Prepared dataset {target_prop} at {datetime.datetime.now()}")


def model_training(model: QSPRModel, dataset: QSPRDataset):
    """Optimize hyperparameters for ChemProp models"""

    hyperparam_dict = {
        "depth": [3, 5],  # number of message passing steps (default 3)
        "hidden_size": [128, 256, 512],  # dim of the hidden layers in MPN (300)
        "ffn_num_layers": [1, 2, 3],  # num layers in FFN after MPN encoding (2)
        "dropout": [0.0, 0.1, 0.2],  # dropout rate (0.0)
    }

    logger.info(f"Training model {model.name} started at {datetime.datetime.now()}.")
    gs = GridSearchOptimization(
        param_grid=hyperparam_dict,
        model_assessor=CrossValAssessor(
            scoring="r2", round=7, mode=EarlyStoppingMode.RECORDING
        ),
    )

    best_params = gs.optimize(model, dataset)

    # log the best epoch for each fold and parameter setting
    log_message = []
    i = 0
    for params in product(*hyperparam_dict.values()):
        params = dict(zip(hyperparam_dict.keys(), params))
        epochs = ",".join(map(str, model.earlyStopping.trainedEpochs[i : i + 5]))
        log_message.append(f"Parameters: {params} => {epochs}")
        i += 5
    logger.info("Best epochs:\n%s", "\n".join(log_message))

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


def hyperparam_optimization(
    qspr_dir: str, overwrite: bool, gpu: int, target_prop: str = "A2AR"
):
    """Main function to create models for all datasets in the qspr_dir.

    Args:
        qspr_dir (str): Path to the directory with the QSPR datasets.
        overwrite (bool): Overwrite existing models.
        gpu (int): GPU to use.
        target_prop (str, optional): Target property to optimize. Defaults to "A2AR".
    """
    dataset_folders = os.listdir(join(qspr_dir, "data"))
    # add data settings for consistency with other models
    data_settings = {"feature_calculators": ["SMILES"]}
    dataset_folders = [
        folder for folder in dataset_folders if folder.endswith("chemprop")
    ]
    dataset_folders = sorted(dataset_folders)
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
    folder = [folder for folder in dataset_folders if target_prop in folder][0]
    logger.info(f"Loading dataset {folder}.")
    dataset = QSPRDataset.fromFile(
        join(qspr_dir, "data", folder, f"{folder}_meta.json")
    )

    # skip if model already exists and overwrite is False
    model_name = f"Chemprop_{dataset.name}"

    if not overwrite and os.path.exists(join(qspr_dir, "models", model_name)):
        logger.info(
            f"Model {model_name} already exists and overwrite is False. Skipping.\n"
        )
    else:
        # Create the model
        model = ChempropModel(
            base_dir=join(qspr_dir, "models"),
            name=model_name,
            parameters={"epochs": 200, "batch_size": 128},
        )
        model.setGPUs([gpu])
        # Train the model
        model_result_df = model_training(model, dataset)
        model_result_df["dataset_settings"] = [data_settings] * len(model_result_df)
        model_result_df["algorithm"] = "Chemprop"
        model_result_df["model_path"] = model.metaFile
        model_result_df["dataset_path"] = dataset.metaFile
        result_df = pd.concat([result_df, model_result_df])
        result_df.to_csv(
            join(qspr_dir, "models", f"hyperparamopt_results_chemprop_{now}.tsv"),
            sep="\t",
            index=False,
        )
        logger.info(f"Model creation finished at {datetime.datetime.now()}.")


if __name__ == "__main__":
    # Parse the command line arguments
    parser = argparse.ArgumentParser(description="ChemProp hyperparameter optimization")
    parser.add_argument(
        "--config_file",
        type=str,
        default="config.json",
        help="Path to the configuration file",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing data sets",
    )
    parser.add_argument(
        "--property",
        type=str,
        default="A2AR",
        help="Target property to optimize",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU to use",
    )
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        config = json.load(f)

    # set data paths
    data_path_a2ar = join(
        config["BASE_DIR"],
        config["PROCESSED_DATA_DIR"],
        "A2ARDataset",
        "A2AR_dataset.tsv",
    )
    data_path_pk = join(
        config["BASE_DIR"],
        config["PROCESSED_DATA_DIR"],
        "PKDataset",
        "pk_dataset.tsv",
    )
    qspr_dir = join(config["BASE_DIR"], config["PROCESSED_DATA_DIR"], "QSPR")

    # save conda environment to outdir
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(join(qspr_dir, "data"), exist_ok=True)
    os.makedirs(join(qspr_dir, "models"), exist_ok=True)
    env_file = join(qspr_dir, f"models/conda_env_{now}.yml")
    export_conda_environment(env_file)

    # Set up logging
    logSettings = enable_file_logger(
        log_folder=f"{qspr_dir}/models",
        filename=f"ChemProp_{args.property}_{now}.log",
        log_name=__name__,
        debug=False,
        disable_existing_loggers=False,
        init_data={
            "TASK": args.property,
            "A2AR_DATA_DIR": join(config["PROCESSED_DATA_DIR"], "A2ARDataset"),
            "PK_DATA_DIR": join(config["PROCESSED_DATA_DIR"], "PKDataset"),
            "QSPR_DIR": join(config["PROCESSED_DATA_DIR"], "QSPR"),
            "SEED": args.seed,
            "GPU": args.gpu,
            "OVERWRITE": args.overwrite,
        },
    )
    logger = logSettings.log

    # Change the format of the root logger
    root_logger = logging.getLogger()
    new_formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    root_logger.handlers[1].setFormatter(new_formatter)

    # Propagate qsprpred logger
    qsprpred_logger = logging.getLogger("qsprpred")
    qsprpred_logger.setLevel(logging.INFO)
    qsprpred_logger.propagate = True
    setLogger(qsprpred_logger)

    # Save the current git commit
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    logger.info(f"Git commit: {sha}")

    dataset_preparation(
        data_path_a2ar, data_path_pk, qspr_dir, args.seed, args.overwrite, args.property
    )
    hyperparam_optimization(qspr_dir, args.overwrite, args.gpu, args.property)

    save_best_models_to_config(qspr_dir, args.config_file)
