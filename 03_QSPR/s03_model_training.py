import argparse
import datetime
import json
import os
from copy import deepcopy
from os.path import join

import git
import numpy as np
import pandas as pd
from mlchemad import TopKatApplicabilityDomain
from qsprpred.benchmarks import BenchmarkRunner, BenchmarkSettings, DataPrepSettings
from qsprpred.data import ClusterSplit, MoleculeTable, QSPRDataset, RandomSplit
from qsprpred.data.processing.applicability_domain import MLChemADWrapper
from qsprpred.data.sources import DataSource
from qsprpred.logs.utils import enable_file_logger, export_conda_environment
from qsprpred.models import CrossValAssessor, QSPRModel, SklearnModel, TestSetAssessor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.preprocessing import StandardScaler


def model_training(model: QSPRModel, dataset: QSPRDataset):
    """Run cross-validation and hyperparameter optimization for a given model and dataset."""
    scores = CrossValAssessor(scoring="r2", round=7)(model, dataset)
    logger.info(f"Cross-validated scores: {scores}")
    score = TestSetAssessor(scoring="r2", round=7)(model, dataset)
    logger.info(f"Test set score: {score}")

    # attach applicability domain
    ap = MLChemADWrapper(TopKatApplicabilityDomain())
    dataset.applicabilityDomain = ap

    model.fitDataset(dataset)
    logger.info(f"Model training finished at {datetime.datetime.now()}.")

    model = QSPRModel.fromFile(model.metaFile)

    applicability = model.applicabilityDomain.contains(
        dataset.getFeatures(concat=True, ordered=True)
    )
    logger.info(f"Inliers: {sum(applicability.values)}")
    logger.info(f"Outliers: {len(applicability.values) - sum(applicability.values)}")


class DataSourceTesting(DataSource):
    """Wrapper around the data set for benchmarking."""

    def __init__(self, name: str, base_dir: str, df: pd.DataFrame):
        self.name = name  # name of the created data set
        self.baseDir = base_dir  # where to save it and all its derived data sets
        self.df = df

    def getData(self, name: str | None = None, **kwargs) -> MoleculeTable:
        """Returns `MoleculeTable` with dataset for benchmarking."""
        name = name or self.name
        mol_table = MoleculeTable(
            df=deepcopy(self.df), name=name, store_dir=self.baseDir, **kwargs
        )
        mol_table.df.drop(columns=["Split_IsTrain"], inplace=True)
        return mol_table


def bootstrapping(
    model: QSPRModel, dataset: QSPRDataset, save_dir: str, seed: int, n_proc: int
):
    """Run bootstrapping for uncertainty estimation with clustersplit and randomsplit."""
    source = DataSourceTesting(
        name=dataset.name, base_dir=join(save_dir, "data"), df=dataset.df
    )

    settings = BenchmarkSettings(
        name=dataset.targetPropertyNames[0],
        n_replicas=50,
        random_seed=seed,
        data_sources=[source],
        descriptors=[
            [
                deepcopy(descriptortable.calculator)
                for descriptortable in dataset.descriptors
            ]
        ],
        target_props=[deepcopy(dataset.targetProperties)],
        prep_settings=[
            DataPrepSettings(
                split=RandomSplit(test_fraction=0.2),
                feature_standardizer=StandardScaler(),
            ),
            DataPrepSettings(
                split=ClusterSplit(test_fraction=0.2),
                feature_standardizer=StandardScaler(),
            ),
        ],
        models=[
            SklearnModel(
                name=model.name,
                alg=model.alg,
                base_dir=f"{save_dir}/models",
                parameters=model.parameters,
            )
        ],
        assessors=[
            TestSetAssessor(scoring="r2"),
            TestSetAssessor(scoring="neg_root_mean_squared_error"),
        ],
    )

    runner = BenchmarkRunner(settings, data_dir=save_dir, n_proc=n_proc)
    logger.info(f"Benchmarking {model.name} started at {datetime.datetime.now()}.")
    runner.run(raise_errors=True)
    logger.info(f"Benchmarking {model.name} finished at {datetime.datetime.now()}.")
    return runner


def applicability_domain_bootstrapping(
    runner: BenchmarkRunner, save_dir: str, property: str
):
    """Calculate applicability domain for bootstrapping results."""
    ap = TopKatApplicabilityDomain()

    applicabiltiliy_df = pd.DataFrame(
        columns=[
            "ReplicaFile",
            "explained_var",
            "split",
            "Domain",
            "MAE",
            "R2",
            "RMSE",
            "num_samples",
        ]
    )

    for replica in runner.iterReplicas():
        # load in replica and prepare data as in the model
        file_name = f"{replica.model.outPrefix}_replica.json"
        replica.initData()
        replica.addDescriptors()
        replica.prepData()
        split = str(replica.prepSettings.split.__class__.__name__)
        train, test = replica.ds.getFeatures()
        # Get mean inlier and outlier error with TOPKAT applicability domain
        # fit applicability domain on training data and predict on test data
        ap.fit(train)
        pred_ap = ap.contains(test)
        # calculate statistics for inliers and outliers
        replica_results = pd.read_csv(f"{replica.model.outPrefix}.ind.tsv", sep="\t")

        def get_stats(preds, labels, domain):
            if len(preds) > 0:
                MAE = mean_absolute_error(labels, preds)
                RMSE = root_mean_squared_error(labels, preds)
                R2 = r2_score(labels, preds)
            else:
                MAE, RMSE, R2 = np.nan, np.nan, np.nan
            return pd.DataFrame(
                {
                    "ReplicaFile": [file_name],
                    "split": [split],
                    "Domain": [domain],
                    "MAE": [MAE],
                    "RMSE": [RMSE],
                    "R2": [R2],
                    "num_samples": [len(preds)],
                }
            )

        preds, labels = (
            replica_results[f"{property}_Prediction"],
            replica_results[f"{property}_Label"],
        )
        outlier_stats = get_stats(preds[~pred_ap], labels[~pred_ap], "outlier")
        inlier_stats = get_stats(preds[pred_ap], labels[pred_ap], "inlier")
        all_stats = get_stats(preds, labels, "all")
        # save results to dataframe
        applicabiltiliy_df = pd.concat(
            [applicabiltiliy_df, outlier_stats, inlier_stats, all_stats],
            ignore_index=True,
        )
        applicabiltiliy_df.reset_index(drop=True).to_csv(
            f"{save_dir}/applicability_domain_bootstrapping.tsv", sep="\t", index=False
        )


def feature_importance(dataset: QSPRDataset, model: QSPRModel, seed: int, n_proc: int):
    """Calculate feature importance using permutation importance."""
    logger.info(
        f"Calculating feature importance for {model.name} started at "
        f"{datetime.datetime.now()}."
    )
    _, X_test = dataset.getFeatures()
    _, y_test = dataset.getTargetPropertiesValues()
    results = permutation_importance(
        deepcopy(model.estimator),
        X_test,
        y_test,
        n_repeats=30,
        scoring=["r2", "neg_mean_squared_error"],
        n_jobs=n_proc,
        random_state=seed,
    )

    # save results to file
    importance_df = pd.DataFrame(
        np.array(
            [
                results["r2"].importances_mean,
                results["r2"].importances_std,
                results["neg_mean_squared_error"].importances_mean,
                results["neg_mean_squared_error"].importances_std,
            ]
        ).T,
        columns=[
            "importances_mean_r2",
            "importances_std_r2",
            "importances_mean_neg_mean_squared_error",
            "importances_std_neg_mean_squared_error",
        ],
        index=dataset.getFeatureNames(),
    )
    importance_df.sort_values(by="importances_mean_r2", ascending=False, inplace=True)
    importance_df.to_csv(f"{model.outPrefix}_feature_importance.csv")
    logger.info(
        f"Feature importance for {model.name} saved to "
        f"{model.outPrefix}_feature_importance.csv at {datetime.datetime.now()}."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train QSPR models.")
    parser.add_argument(
        "--config_file",
        type=str,
        default="config.json",
        help="Path to the configuration file",
    )
    parser.add_argument(
        "--n_proc",
        type=int,
        default=10,
        help="Number of processors to use",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        config = json.load(f)

    # set data paths
    qspr_dir = join(config["BASE_DIR"], config["PROCESSED_DATA_DIR"], "QSPR")
    model_dir = join(qspr_dir, "models")
    data_dir = join(qspr_dir, "data")
    model_a2ar = join(
        model_dir, config["A2AR_MODEL"], f"{config['A2AR_MODEL']}_meta.json"
    )
    data_a2ar = join(data_dir, config["A2AR_DATA"], f"{config['A2AR_DATA']}_meta.json")
    model_fu = join(model_dir, config["FU_MODEL"], f"{config['FU_MODEL']}_meta.json")
    data_fu = join(data_dir, config["FU_DATA"], f"{config['FU_DATA']}_meta.json")
    model_vdss = join(
        model_dir, config["VDSS_MODEL"], f"{config['VDSS_MODEL']}_meta.json"
    )
    data_vdss = join(data_dir, config["VDSS_DATA"], f"{config['VDSS_DATA']}_meta.json")
    model_cl = join(model_dir, config["CL_MODEL"], f"{config['CL_MODEL']}_meta.json")
    data_cl = join(data_dir, config["CL_DATA"], f"{config['CL_DATA']}_meta.json")

    # save conda environment to outdir
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(join(qspr_dir, "models"), exist_ok=True)
    env_file = join(qspr_dir, f"models/conda_env_{now}.yml")
    export_conda_environment(env_file)

    # Set up logging
    logSettings = enable_file_logger(
        log_folder=f"{qspr_dir}/models",
        filename=f"ModelTraining_{now}.log",
        log_name=__name__,
        debug=False,
        disable_existing_loggers=False,
        init_data={
            "PK_QSPR_DIR": join(config["PROCESSED_DATA_DIR"], "QSPR"),
            "MODEL_A2AR": config["A2AR_MODEL"],
            "DATA_A2AR": config["A2AR_DATA"],
            "MODEL_FU": config["FU_MODEL"],
            "DATA_FU": config["FU_DATA"],
            "MODEL_VDSS": config["VDSS_MODEL"],
            "DATA_VDSS": config["VDSS_DATA"],
            "MODEL_CL": config["CL_MODEL"],
            "DATA_CL": config["CL_DATA"],
        },
    )
    logger = logSettings.log

    # Save the current git commit
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    logger.info(f"Git commit: {sha}")

    for model, data, prop in zip(
        [model_a2ar, model_fu, model_vdss, model_cl],
        [data_a2ar, data_fu, data_vdss, data_cl],
        ["A2AR", "FU", "VDSS", "CL"],
    ):
        logger.info(f"Training model: {model}")

        #     # read the best model and data
        model = QSPRModel.fromFile(model)
        dataset = QSPRDataset.fromFile(data)

        # Train the model
        model_training(model, dataset)

        # Repeat the training with bootstrapping for uncertainty estimation
        benchmarkrunner = bootstrapping(
            model,
            dataset,
            join(qspr_dir, f"bootstrapping_{prop}"),
            seed=args.seed,
            n_proc=args.n_proc,
        )

        # Calculate applicability domain
        applicability_domain_bootstrapping(
            benchmarkrunner, join(qspr_dir, f"bootstrapping_{prop}"), property=prop
        )

        # Calculate feature importance
        feature_importance(dataset, model, seed=args.seed, n_proc=args.n_proc)

    logger.info("Training finished.")
