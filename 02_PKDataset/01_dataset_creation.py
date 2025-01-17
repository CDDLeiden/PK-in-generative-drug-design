"""Create PK dataset from Lombardo et al. (2018) dataset"""

import argparse
import datetime
import json
import os
from os.path import join

import git
import pandas as pd
from chembl_structure_pipeline import get_parent_mol, standardizer
from qsprpred.logs.utils import enable_file_logger, export_conda_environment
from rdkit import Chem


def main(data_path: str, output_dir: str):
    """Create PK dataset from Lombardo data

    Filters out molecules with MW > 900 Da, invalid SMILES, metals Pt or Gd.
    Removes stereochemistry and takes mean of duplicates.
    Saves removed molecules to tsv.
    """

    # Read in Lombardo data
    df = pd.read_excel(data_path, header=8)

    # filter out all rows with a MW > 900 Da (upper molecular-weight limit for
    # small molecule Wang et al. https://pubs.acs.org/doi/10.1021/acs.jcim.9b00300)
    logger.info(f"Removing {df[df['MW'] > 900].shape[0]} molecules with MW > 900 Da")
    df_removed = df[df["MW"] > 900].copy()
    df_removed["reason"] = "MW > 900 Da"
    df = df[df["MW"] < 900]

    # convert all smiles to molecules and filter out any that are not valid
    df["molecule"] = df["SMILES"].apply(lambda x: Chem.MolFromSmiles(x))
    logger.info(f"Removing {df['molecule'].isna().sum()} invalid SMILES")
    removed = df[df["molecule"].isna()].copy()
    removed["reason"] = "Invalid SMILES"
    df_removed = pd.concat([df_removed, removed])
    df = df[df["molecule"].notna()]

    # filter out any molecules with metals Pt (platinum), Gd(gadolinium)
    METALS = ["Pt", "Gd"]
    df["metals"] = df["molecule"].apply(
        lambda x: any([atom.GetSymbol() in METALS for atom in x.GetAtoms()])
    )
    logger.info(f"Removing {df['metals'].sum()} molecules with metals Pt or Gd")
    removed = df[df["metals"]].copy()
    removed["reason"] = "Metals Pt or Gd"
    df_removed = pd.concat([df_removed, removed])
    df = df[~df["metals"]]

    # Rename columns to easier names
    df = df.rename(
        columns={
            "human VDss (L/kg)": "VDSS",
            "human CL (mL/min/kg)": "CL",
            "fraction unbound \nin plasma (fu)": "FU",
        }
    )

    # remove salts and standardize with chembl_standardizer
    df["SMILES"] = df["molecule"].apply(
        lambda x: Chem.MolToSmiles(get_parent_mol(standardizer.standardize_mol(x))[0])
    )

    # save removed molecules to tsv
    logger.info(
        f"Saving {df_removed.shape[0]} removed molecules to "
        f"{output_dir}/removed_molecules.tsv"
    )
    df_removed.drop(["molecule", "metals"], axis=1).to_csv(
        f"{output_dir}/removed_molecules.tsv", sep="\t", index=False
    )

    # remove stereochemistry and take mean of duplicates
    df["SMILES_withStereo"] = df["SMILES"]
    df["SMILES"] = df["SMILES"].apply(
        lambda x: Chem.MolToSmiles(
            Chem.MolFromSmiles(x), isomericSmiles=False, canonical=True
        )
    )
    logger.info(f"Combining {df.duplicated('SMILES', keep=False).sum()} duplicates")
    df = df[["Name", "SMILES", "SMILES_withStereo", "VDSS", "CL", "FU"]]
    df[["VDSS_raw", "CL_raw", "FU_raw"]] = df[["VDSS", "CL", "FU"]].astype(str)
    df = (
        df.groupby("SMILES")
        .agg(
            {
                "Name": lambda x: "; ".join(x),
                "SMILES_withStereo": lambda x: "; ".join(x),
                "VDSS": "mean",
                "CL": "mean",
                "FU": "mean",
                "VDSS_raw": lambda x: "; ".join(x),
                "CL_raw": lambda x: "; ".join(x),
                "FU_raw": lambda x: "; ".join(x),
            }
        )
        .reset_index()
    )

    # Get some statistics (number of molecules, mean, min, max, std of VDSS, CL, FU)
    stats = df[["VDSS", "CL", "FU"]].describe().T
    stats = stats.reset_index()
    stats = stats.rename(columns={"index": "parameter"})
    stats = stats.round(3)
    stats["count"] = stats["count"].astype(int)

    logger.info(f"Statistics of VDSS, CL, FU:\n{stats}")
    logger.info(f"Saving statistics to {output_dir}/statistics.tsv")
    stats.to_csv(f"{output_dir}/statistics.tsv", sep="\t", index=False)

    # save to tsv
    logger.info(f"Saving {df.shape[0]} molecules to {output_dir}/pk_dataset.tsv")
    df.to_csv(f"{output_dir}/pk_dataset.tsv", sep="\t", index=False)


if __name__ == "__main__":
    # get data path and output directory from user args
    parser = argparse.ArgumentParser(description="Create PK dataset from Lombardo data")
    parser.add_argument(
        "--config_file",
        type=str,
        default="config.json",
        help="Path to the configuration file",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="Supplemental_82966_revised_corrected.xlsx",
        help="Name of the Lombardo data file",
    )
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        config = json.load(f)

    # set data paths
    data_path_path = join(config["BASE_DIR"], config["RAW_DATA_DIR"], args.data_path)
    output_dir = join(config["BASE_DIR"], config["PROCESSED_DATA_DIR"], "PKDataset")
    os.makedirs(output_dir, exist_ok=True)

    # save conda environment to outdir
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    env_file = join(output_dir, f"conda_env_{now}.yml")
    export_conda_environment(env_file)

    # Set up logging
    logSettings = enable_file_logger(
        log_folder=output_dir,
        filename=f"DatasetCreation_{now}.log",
        log_name=__name__,
        debug=False,
        disable_existing_loggers=False,
        init_data={
            "data_path": join(config["RAW_DATA_DIR"], args.data_path),
            "output_dir": join(config["PROCESSED_DATA_DIR"], "PKDataset"),
        },
    )
    logger = logSettings.log

    # Save the current git commit
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    logger.info(f"Git commit: {sha}")

    main(data_path_path, output_dir)
