"""Prepare datasets for QSPR models by transforming the target properties and calculating and filtering features."""

import argparse
import datetime
import json
import os
from copy import deepcopy
from os.path import join

import git
import tqdm
from boruta import BorutaPy
from qsprpred.data import QSPRDataset, RandomSplit
from qsprpred.data.descriptors.fingerprints import MorganFP
from qsprpred.data.descriptors.sets import RDKitDescs
from qsprpred.data.processing.feature_filters import (
    BorutaFilter,
    HighCorrelationFilter,
    LowVarianceFilter,
)
from qsprpred.logs.utils import enable_file_logger, export_conda_environment
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


class PrepSettings:
    """Class to store settings for data preparation for the QSPR datasets."""

    def __init__(
        self,
        feature_calculators: list,
        low_var_th: float = None,
        high_corr_th: float = None,
        boruta_perc: float = None,
        seed: int = 42,
        n_proc: int = 5,
    ):
        self.feature_calculators = feature_calculators
        self.low_var_th = low_var_th
        self.high_corr_th = high_corr_th
        self.boruta_perc = boruta_perc
        self.seed = seed
        self.n_proc = n_proc

    def getSettings(self):
        filters = []
        if self.low_var_th:
            filters.append(LowVarianceFilter(self.low_var_th))
        if self.high_corr_th:
            filters.append(HighCorrelationFilter(self.high_corr_th))
        if self.boruta_perc:
            filters.append(
                BorutaFilter(
                    BorutaPy(
                        estimator=RandomForestRegressor(n_jobs=self.n_proc),
                        perc=self.boruta_perc,
                    ),
                    self.seed,
                )
            )
        return {
            "feature_calculators": self.feature_calculators,
            "split": RandomSplit(test_fraction=0.2, seed=self.seed),
            "feature_filters": filters,
            "feature_standardizer": StandardScaler(),
            "random_state": self.seed,
        }

    def __dict__(self):
        return {
            "low_var_th": self.low_var_th,
            "high_corr_th": self.high_corr_th,
            "boruta_perc": self.boruta_perc,
            "feature_calculators": [str(f) for f in self.feature_calculators],
        }

    def __str__(self):
        feature_names = "_".join([str(f) for f in self.feature_calculators])
        if self.boruta_perc:
            return f"{feature_names}_B_{self.boruta_perc}_HC_{self.high_corr_th}"
        else:
            return f"{feature_names}_LV_{self.low_var_th}_HC_{self.high_corr_th}"


def main(
    data_path_a2ar: str,
    data_path_pk: str,
    output_dir: str,
    seed: int = 42,
    n_proc: int = 5,
    overwrite: bool = False,
):
    """Prepare datasets for QSPR models

    Prepares datasets for the A2AR, CL, FU, and VDSS target properties
    by transforming the target properties and calculating and filtering features.

    Args:
        data_path_a2ar (str): Path to the A2AR dataset.
        data_path_pk (str): Path to the PK dataset.
        output_dir (str): Path to the output directory.
        seed (int, optional): Random seed. Defaults to 42.
        n_proc (int, optional): Number of processors to use. Defaults to 5.
        overwrite (bool, optional): Overwrite existing datasets. Defaults to False.
    """

    # Define the transformers for the target properties
    transformer_dict = {
        "CL": lambda x: (__import__("numpy").log10(x)),
        "FU": lambda x: (__import__("numpy").sqrt(x)),
        "VDSS": lambda x: (__import__("numpy").log10(x)),
        "A2AR": None,
    }

    # Create different pre-processing settings
    descriptors = [
        [MorganFP(radius=3, nBits=2048), RDKitDescs()],
        [MorganFP(radius=3, nBits=2048)],
        [RDKitDescs()],
    ]
    prep_settings_list = [
        PrepSettings(feature_calculators, low_var_th, high_corr_th, None)
        for feature_calculators in descriptors
        for low_var_th in [0.01, 0.05]
        for high_corr_th in [0.9, 0.95, 0.99]
    ] + [
        PrepSettings(feature_calculators, None, high_corr_th, boruta_perc)
        for feature_calculators in descriptors
        for boruta_perc in [80, 100]
        for high_corr_th in [0.9, 0.95, 0.99]
    ]
    log.info(f"Created {len(prep_settings_list)} pre-processing settings.")

    # Prepare the datasets for the different target properties and pre-processing settings
    for target_prop in ["A2AR", "CL", "FU", "VDSS"]:
        for prep_settings in tqdm.tqdm(prep_settings_list):
            data_path = data_path_a2ar if target_prop == "A2AR" else data_path_pk

            # Skip if dataset already exists
            if (
                os.path.exists(
                    join(output_dir, "data", f"{target_prop}_{prep_settings}")
                )
                and not overwrite
            ):
                log.info(
                    f"Dataset with settings {prep_settings} for {target_prop} already "
                    f"exists. Skipping. \n"
                )
                continue
            # Initialize and prepare the dataset
            dataset = QSPRDataset.fromTableFile(
                name=f"{target_prop}_{prep_settings}",
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
                n_jobs=n_proc,
            )
            try:
                dataset.prepareDataset(**deepcopy(prep_settings.getSettings()))
            except Exception as e:
                log.error(
                    f"Failed to prepare dataset {target_prop} with settings "
                    f"{prep_settings}: {e} \n"
                )
                continue
            dataset.save()
            log.info(
                f"Prepared dataset {target_prop} with settings {prep_settings} at "
                f"{datetime.datetime.now()}"
            )
            log.info(f"Number of features after filtering: {dataset.X.shape[1]} \n")
            # Also save the settings
            with open(
                join(
                    output_dir,
                    "data",
                    f"{target_prop}_{prep_settings}",
                    "prep_settings.json",
                ),
                "w",
            ) as f:
                json.dump(prep_settings.__dict__(), f)

    log.info(f"Finished at {datetime.datetime.now()}")


if __name__ == "__main__":
    # Parse the command line arguments
    parser = argparse.ArgumentParser(description="Prepare datasets for QSPR models")
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
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--n_proc",
        type=int,
        default=5,
        help="Number of processors to use",
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
        args.data_path_pk,
        "pk_dataset.tsv",
    )
    output_dir = join(config["BASE_DIR"], config["PROCESSED_DATA_DIR"], "QSPR")

    # save conda environment to outdir
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(join(output_dir, "data"), exist_ok=True)
    env_file = join(output_dir, f"data/conda_env_{now}.yml")
    export_conda_environment(env_file)

    # Set up logging
    logSettings = enable_file_logger(
        log_folder=f"{output_dir}/data",
        filename=f"DataPrep_{now}.log",
        log_name=__name__,
        debug=False,
        disable_existing_loggers=False,
        init_data={
            "A2AR_DATA_DIR": join(config["PROCESSED_DATA_DIR"], "A2ARDataset"),
            "PK_DATA_DIR": join(config["PROCESSED_DATA_DIR"], "PKDataset"),
            "OUTPUT_DIR": join(config["PROCESSED_DATA_DIR"], "QSPR"),
            "SEED": args.seed,
            "N_PROC": args.n_proc,
            "OVERWRITE": args.overwrite,
        },
    )
    log = logSettings.log

    # Save the current git commit
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    log.info(f"Git commit: {sha}")

    main(
        data_path_a2ar, data_path_pk, output_dir, args.seed, args.n_proc, args.overwrite
    )
