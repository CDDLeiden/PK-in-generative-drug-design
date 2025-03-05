"""Make UMAPs from generated molecules and datasets"""

import argparse
import datetime
import json
import os
from os.path import join

import git
import pandas as pd
import umap.umap_ as umap_
from qsprpred.data import MoleculeTable, QSPRDataset
from qsprpred.data.descriptors.fingerprints import MorganFP
from qsprpred.logs.utils import enable_file_logger, export_conda_environment


def calculate_umap(smiles_list: list, umap_settings: dict, n_proc) -> dict:
    """Calculate UMAP coordinates for a list of SMILES"""
    df = pd.DataFrame(smiles_list, columns=["SMILES"])
    mt = MoleculeTable(
        df=df,
        store_dir=".",
        n_jobs=n_proc,
        name="AllSmiles",
        overwrite=True,
        random_state=42,
    )
    mt.addDescriptors([MorganFP(radius=3, nBits=2048)])
    morganfp = mt.getDescriptors().values
    umap = umap_.UMAP(
        n_components=2,
        metric="jaccard",
        n_jobs=n_proc,
        low_memory=False,
        **umap_settings,
    )
    umap_coords = umap.fit_transform(morganfp)
    return {smiles: coords for smiles, coords in zip(mt.df.SMILES, umap_coords)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Make UMAP for generated molecules and datasets"
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="config.json",
        help="Path to the configuration file",
    )
    parser.add_argument(
        "--n_proc",
        type=int,
        default=5,
        help="Number of processors to use",
    )
    parser.add_argument(
        "--n_reps",
        type=int,
        default=3,
        help="Number of repetitions",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=10000,
        help="Number of samples to generate",
    )
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        config = json.load(f)

    # set paths
    FINETUNING_DIR = join(
        config["BASE_DIR"], config["PROCESSED_DATA_DIR"], "DNDD", "finetuned"
    )
    GENERATED_DIR = join(
        config["BASE_DIR"], config["PROCESSED_DATA_DIR"], "DNDD", "generated"
    )

    # QSPR models and data
    QSPR_DIR = join(config["BASE_DIR"], config["PROCESSED_DATA_DIR"], "QSPR")
    data_dir = join(QSPR_DIR, "data")
    data_a2ar = join(data_dir, config["A2AR_DATA"], f"{config['A2AR_DATA']}_meta.json")
    data_fu = join(data_dir, config["FU_DATA"], f"{config['FU_DATA']}_meta.json")
    data_vdss = join(data_dir, config["VDSS_DATA"], f"{config['VDSS_DATA']}_meta.json")
    data_cl = join(data_dir, config["CL_DATA"], f"{config['CL_DATA']}_meta.json")

    # load in smiles for all datasets
    A2AR_smiles = QSPRDataset.fromFile(data_a2ar).df.SMILES.to_list()
    CL_smiles = QSPRDataset.fromFile(data_cl).df.SMILES.to_list()
    FU_smiles = QSPRDataset.fromFile(data_fu).df.SMILES.to_list()
    VDSS_smiles = QSPRDataset.fromFile(data_vdss).df.SMILES.to_list()
    prop_smiles = {
        "A2AR": A2AR_smiles,
        "CL": CL_smiles,
        "FU": FU_smiles,
        "VDSS": VDSS_smiles,
    }

    # save conda environment to outdir
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(join(GENERATED_DIR), exist_ok=True)
    env_file = join(GENERATED_DIR, f"conda_env_{now}.yml")
    export_conda_environment(env_file)

    # Set up logging
    logSettings = enable_file_logger(
        log_folder=GENERATED_DIR,
        filename=f"Umap_{now}.log",
        log_name=__name__,
        debug=False,
        disable_existing_loggers=False,
        init_data={
            "GENERATED_DIR": join(config["PROCESSED_DATA_DIR"], "DNDD", "generated"),
            "QSPR_DIR": join(config["PROCESSED_DATA_DIR"], "QSPR"),
            "A2AR_MODEL": config["A2AR_MODEL"],
            "FU_MODEL": config["FU_MODEL"],
            "VDSS_MODEL": config["VDSS_MODEL"],
            "CL_MODEL": config["CL_MODEL"],
            "N_PROC": args.n_proc,
            "N_REPETITIONS": args.n_reps,
            "N_SAMPLES": args.n_samples,
        },
    )
    logger = logSettings.log

    # Save the current git commit
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    logger.info(f"Git commit: {sha}")

    ####------ Make UMAP for A2AR optimized molecules and dataset -------###
    ## collect all unique and valid molecules from all generated datasets
    for rep in range(args.n_reps):
        A2AR_files = [
            f
            for f in os.listdir(GENERATED_DIR)
            if "A2AR" in f and os.path.isdir(join(GENERATED_DIR, f))
        ]
        A2AR_files = [f for f in A2AR_files if f"_{rep}" in f or "finetuned" in f]
        smiles_list = []
        for name in A2AR_files:
            df = pd.read_csv(
                f"{GENERATED_DIR}/{name}/generated_{args.n_samples}.tsv", sep="\t"
            )
            df = df[df["Unique & Valid & Applicable & Novel"] == 1]
            smiles_list.extend(df.SMILES.to_list())
        smiles_list.extend(A2AR_smiles)  # add dataset SMILES
        smiles_list = list(set(smiles_list))

        umap_settings = {"n_neighbors": 150, "min_dist": 0.6}
        umap_coords = calculate_umap(smiles_list, umap_settings, args.n_proc)
        for name in A2AR_files:
            # add UMAP coordinates to generated molecules
            df = pd.read_csv(
                f"{GENERATED_DIR}/{name}/generated_{args.n_samples}.tsv", sep="\t"
            )
            df[f"UMAP1_{rep}"] = df.SMILES.map(
                lambda x: umap_coords[x][0] if x in umap_coords else None
            )
            df[f"UMAP2_{rep}"] = df.SMILES.map(
                lambda x: umap_coords[x][1] if x in umap_coords else None
            )
            df.to_csv(
                f"{GENERATED_DIR}/{name}/generated_{args.n_samples}.tsv",
                sep="\t",
                index=False,
            )

            # add UMAP coordinates to datasets
            name = name.replace("max", "").replace(f"_{rep}", "")
            if os.path.isfile(f"{FINETUNING_DIR}/{name}/train_df.tsv"):
                df = pd.read_csv(f"{FINETUNING_DIR}/{name}/train_df.tsv", sep="\t")
                df[f"UMAP1_{rep}"] = df.SMILES.map(lambda x: umap_coords[x][0])
                df[f"UMAP2_{rep}"] = df.SMILES.map(lambda x: umap_coords[x][1])
                df.to_csv(
                    f"{FINETUNING_DIR}/{name}/train_df.tsv", sep="\t", index=False
                )
        logger.info(
            f"Finished UMAP for A2AR molecules and datasets for repetition {rep} at "
            f"{datetime.datetime.now()}."
        )

    ####------ Make UMAP for PK optimized molecules and datasets -------###
    for rep in range(args.n_reps):
        for prop in ["CL", "FU", "VDSS"]:
            generated_PK = [
                f"{FINETUNING_DIR}/{prop}/train_df.tsv",
                f"{GENERATED_DIR}/{prop}_finetuned/generated_{args.n_samples}.tsv",
                f"{GENERATED_DIR}/{prop}max_{rep}/generated_{args.n_samples}.tsv",
                f"{GENERATED_DIR}/{prop}min_{rep}/generated_{args.n_samples}.tsv",
            ]
            # calculate UMAP for generated PK molecules and datasets
            smiles_list = []
            for filename in generated_PK:
                df = pd.read_csv(filename, sep="\t")
                if "Unique & Valid & Applicable & Novel" in df.columns:
                    df = df[df["Unique & Valid & Applicable & Novel"] == 1]
                smiles_list.extend(df.SMILES.to_list())
            smiles_list.extend(prop_smiles[prop])  # add dataset SMILES
            smiles_list = list(set(smiles_list))
            umap_settings = {"n_neighbors": 50, "min_dist": 0.001}
            umap_coords = calculate_umap(smiles_list, umap_settings, args.n_proc)
            # add UMAP coordinates to generated molecules and datasets
            for filename in generated_PK:
                df = pd.read_csv(filename, sep="\t")
                df[f"UMAP1_{rep}"] = df.SMILES.map(
                    lambda x: umap_coords[x][0] if x in umap_coords else None
                )
                df[f"UMAP2_{rep}"] = df.SMILES.map(
                    lambda x: umap_coords[x][1] if x in umap_coords else None
                )
                df.to_csv(filename, sep="\t", index=False)
            logger.info(
                f"Finished UMAP for {prop} molecules and datasets for repetition {rep} at "
                f"{datetime.datetime.now()}."
            )
