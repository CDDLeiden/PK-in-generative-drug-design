"""Create A2AR dataset from Papyrus data"""

import argparse
import datetime
import json
import os
import os.path
from os.path import join
from typing import Tuple

import pandas as pd
from papyrus_scripts.download import download_papyrus
from papyrus_scripts.preprocess import (
    consume_chunks,
    keep_accession,
    keep_quality,
    keep_type,
)
from papyrus_scripts.reader import read_papyrus
from qsprpred.logs.utils import enable_file_logger, export_conda_environment
from rdkit import Chem
from rdkit.Chem import Draw


def substructure_filter(
    data: pd.DataFrame,
    substructure_smiles: str,
    save: bool = False,
    dir_name: str = None,
) -> Tuple[pd.DataFrame, int]:
    """Filter out compounds containing a specific substructure.

    Args:
        data (pd.DataFrame): Dataframe with SMILES.
        substructure_smiles (str): SMILES of substructure to filter out.
        save (bool, optional): If True, save images of removed compounds. Defaults to False.
        dir_name (str, optional): Directory to save images. Defaults to None.

    Returns:
        pd.DataFrame: Dataframe without compounds containing the substructure.
        int: Number of removed compounds.
    """

    substructure_mol = Chem.MolFromSmarts(substructure_smiles)
    num_substructures = 0
    for i, row in data.iterrows():
        mol = Chem.MolFromSmiles(row.SMILES)
        if mol.HasSubstructMatch(substructure_mol):
            SubstructMatch = mol.GetSubstructMatch(substructure_mol)
            data = data.drop(i)
            num_substructures += 1
            if save and dir_name is not None:
                img = Draw.MolToImage(
                    mol,
                    size=(600, 600),
                    fitImage=True,
                    highlightAtoms=SubstructMatch,
                    highlightBonds=SubstructMatch,
                )
                img.save("%s/%s.png" % (dir_name, row.SMILES))
    return data, num_substructures


def create_dataset(
    output_dir,
    data_path,
    version,
    plusplus,
    target_id_dict,
    substructure_dict,
    activity_types,
    min_quality,
    save_images=True,
):
    """Creates dataset from Papyrus data.

    Args:
        output_dir (str): Path to output directory.
        data_path (str): Path to Papyrus data.
        version (str): Version of Papyrus data.
        plusplus (bool): If True, Papyrus++ data is used.
        target_id_dict (dict): Dictionary with target ids as keys and target names as
            values.
        substructure_dict (dict): Dictionary with target ids as keys and substructure
            smiles as values.
        activity_types (list): List of activity types.
        min_quality (str): Minimum quality of data points.
        save_images (bool, optional): If True, images of removed compounds are saved.
            Defaults to True.
    """
    download_papyrus(data_path, version, only_pp=plusplus, descriptors=None)

    # read in and filter papyrus set for adenosine receptor data
    papyrus_data = read_papyrus(
        is3d=False,
        chunksize=1000000,
        source_path=data_path,
        version=version,
        plusplus=plusplus,
    )

    filter = keep_accession(data=papyrus_data, accession=list(target_id_dict.values()))
    filter2 = keep_type(data=filter, activity_types=activity_types)
    filter3 = keep_quality(data=filter2, min_quality=min_quality)
    AR_data = consume_chunks(
        filter3, total=60
    )  # 60 = 59,763,781 aggregated points / 1,000,000 chunksize
    AR_data = AR_data.reset_index(drop=True)
    print(AR_data.head())

    logger.info(f"\nNumber of activity points: {AR_data.shape[0]}")

    # Remove molecules containing specific substructures
    for sub in substructure_dict:
        save_dir = None
        if save_images:
            save_dir = "%s/A2AR_dataset_substr/%s" % (output_dir, sub)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        AR_data, removed = substructure_filter(
            AR_data, substructure_dict[sub], save=save_images, dir_name=save_dir
        )
        logger.info(
            "%s molecules containing %s <%s> were removed."
            % (removed, sub, substructure_dict[sub])
        )

    logger.info(f"\nFinal number of activity points: {AR_data.shape[0]}")

    # rename pchembl_value_Mean to A2AR for convenience
    AR_data.rename(columns={"pchembl_value_Mean": "A2AR"}, inplace=True)

    AR_data.to_csv(os.path.join(output_dir, "A2AR_dataset.tsv"), index=False, sep="\t")


def EnvironmentArgParser():
    """Define and read command line arguments."""

    parser = argparse.ArgumentParser(description="Create adenosine receptor dataset")
    parser.add_argument(
        "--config_file",
        type=str,
        default="config.json",
        help="Path to the configuration file",
    )
    parser.add_argument(
        "-v", "--version", type=str, default="05.6", help="Papyrus version to use"
    )
    parser.add_argument(
        "-pp", "--plusplus", action="store_true", help="If included, use Papyrus++ data"
    )
    parser.add_argument(
        "-at",
        "--activity_types",
        nargs="+",
        default=["IC50", "Ki", "EC50", "Kd"],
        help="Activity types to include",
    )
    parser.add_argument(
        "-mq",
        "--min_quality",
        type=str,
        default="High",
        help="Minimum quality of data points to include",
    )
    parser.add_argument(
        "-s",
        "--save_images",
        action="store_true",
        help="Save images of the filtered out structures",
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = EnvironmentArgParser()

    with open(args.config_file, "r") as f:
        config = json.load(f)

    data_path = join(config["BASE_DIR"], config["RAW_DATA_DIR"])
    output_dir = os.path.join(
        config["BASE_DIR"], config["PROCESSED_DATA_DIR"], "A2ARDataset"
    )

    # set data paths
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # save conda environment to output_dir
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    env_file = os.path.join(output_dir, f"conda_env_{now}.yml")
    export_conda_environment(env_file)

    # Set up logging
    logSettings = enable_file_logger(
        log_folder=output_dir,
        filename=f"DatasetCreation_{now}.log",
        log_name=__name__,
        debug=False,
        disable_existing_loggers=False,
        init_data={
            "DATA_PATH": join(config["RAW_DATA_DIR"], "papyrus"),
            "OUTPUT_DIR": join(config["PROCESSED_DATA_DIR"], "A2ARDataset"),
            "PAPYRUS VERSION": args.version,
            "PLUSPLUS": args.plusplus,
            "ACTIVITY_TYPES": args.activity_types,
            "MIN_QUALITY": args.min_quality,
            "SAVE_IMAGES": args.save_images,
        },
    )
    logger = logSettings.log

    # targets of interest
    target_id_dict = {"A2AR": "P29274"}

    # Remove molecules containing ribose from dataset, because those are usually agonists
    # 'Amiloride', 'Thiophene' are filtered out because allosteric modulators
    # remove molecules containing selenium, because not all descriptors can be calculated for them
    substructure_dict = {
        "Amiloride": "Cl-c:1:c(-[NX3]):[nX2]:c(-[ND1H2]):c(-C(-[ND2H1]-C(=[ND1H1])-[NX3])=[OX1]):[nX2]1",
        "Thiophene": "c1ccsc1",
        "Ribose": "OC1COCC1O",
        "Selenium": "[Se]",
    }

    create_dataset(
        output_dir,
        data_path,
        args.version,
        args.plusplus,
        target_id_dict,
        substructure_dict,
        args.activity_types,
        args.min_quality,
        args.save_images,
    )
