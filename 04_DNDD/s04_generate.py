"""Generate n molecules for all scenarios and repetitions."""

import argparse
import datetime
import json
import os
from os.path import join

import git
import pandas as pd
from _drugex_environment import *
from drugex.data.corpus.vocabulary import VocSmiles
from drugex.logs import logger
from drugex.training.generators import SequenceRNN
from drugex.training.scorers.properties import Property
from qsprpred.data import QSPRDataset
from qsprpred.data.chem.clustering import FPSimilarityLeaderPickerClusters
from qsprpred.data.descriptors.fingerprints import MorganFP
from qsprpred.data.descriptors.sets import RDKitDescs, TanimotoDistances
from qsprpred.logs.utils import enable_file_logger, export_conda_environment
from rdkit import Chem
from rdkit.SimDivFilters import rdSimDivPickers


### overwrite FPSimilarityLeaderPickerClusters _get_centroids method to access the centroids
class FPSimilarityLeaderPickerClusters(FPSimilarityLeaderPickerClusters):
    def _get_centroids(self, fps: list) -> list:
        """Get cluster centroids with LeaderPicker algorithm."""
        picker = rdSimDivPickers.LeaderPicker()
        self.centroid_indices = picker.LazyBitVectorPick(
            fps, len(fps), self.similarityThreshold
        )

        return self.centroid_indices


class GenerateAndScore:
    def __init__(self, num_samples, batch_size, n_proc, all_scorers_env):
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.n_proc = n_proc
        self.all_scorers_env = all_scorers_env

    def __call__(
        self,
        name,
        agent,
        environment=None,
    ):
        os.makedirs(join(GENERATED_DIR, name), exist_ok=True)
        logger.info(f"Starting generating molecules for {name}")
        sample = agent.generate(
            evaluator=environment,
            num_samples=self.num_samples,
            batch_size=self.batch_size,
            n_proc=self.n_proc,
            raw_scores=False,
            drop_duplicates=False,
            drop_invalid=False,
        )

        if environment is not None:
            scorer_keys = environment.getScorerKeys()
            sample = sample.rename(
                columns={key: f"modified_{key}" for key in scorer_keys}
            )

        # calculate all scores from all scorers
        sample = pd.concat(
            [sample, self.all_scorers_env.getUnmodifiedScores(sample.SMILES.to_list())],
            axis=1,
        )

        sample["Valid"] = [
            1 if Chem.MolFromSmiles(smiles) is not None else 0
            for smiles in sample.SMILES
        ]
        sample["Generated_SMILES"] = sample.SMILES
        sample["SMILES"] = [
            (
                Chem.MolToSmiles(Chem.MolFromSmiles(row.SMILES), canonical=True)
                if row.Valid != 0
                else None
            )
            for _, row in sample.iterrows()
        ]

        sample["Unique"] = (~sample.SMILES.duplicated(keep="first")).astype(int)

        # check if present in datasets
        sample["in_A2AR"] = sample.SMILES.isin(A2AR_smiles).astype(int)
        sample["in_CL"] = sample.SMILES.isin(CL_smiles).astype(int)
        sample["in_FU"] = sample.SMILES.isin(FU_smiles).astype(int)
        sample["in_VDSS"] = sample.SMILES.isin(VDSS_smiles).astype(int)
        sample["Novel"] = (
            sample[["in_A2AR", "in_CL", "in_FU", "in_VDSS"]].sum(axis=1) == 0
        ).astype(int)

        # check if applicable according to relevant scorers
        tasks = [part for part in name.split("_") if not part.isdigit()]
        scorer_columns = [
            col
            for col in sample.columns
            if ("applicability_scorer" in col) and not ("modified" in col)
        ]
        applicability_cols = [
            col for col in scorer_columns if any(task in col for task in tasks)
        ]
        sample["Applicable"] = (
            sample[applicability_cols].sum(axis=1) == len(applicability_cols)
        ).astype(int)

        # reduce sample to unique to valid, unique, applicable and novel samples
        sample_unique = sample[
            (sample.Valid == 1)
            & (sample.Unique == 1)
            & (sample.Applicable == 1)
            & (sample.Novel == 1)
        ].copy()

        # calculate RDkit descriptors
        rdkit_descs = RDKitDescs()
        sample_unique = pd.concat(
            [
                sample_unique,
                rdkit_descs(
                    sample_unique.SMILES.to_list(),
                    props={"QSPRID": sample_unique.index},
                ),
            ],
            axis=1,
        )

        # cluster dataset
        clusterer = FPSimilarityLeaderPickerClusters(similarity_threshold=0.8)
        cluster_dict = clusterer.get_clusters(sample_unique["SMILES"].to_list())

        # make dataframe of centroid nr and smiles
        pd.DataFrame(
            {
                "cluster": list(cluster_dict.keys()),
                "SMILES": [
                    sample_unique.iloc[idx]["SMILES"]
                    for idx in clusterer.centroid_indices
                ],
            }
        ).to_csv(
            join(GENERATED_DIR, f"{name}/generated_{self.num_samples}_centroids.tsv"),
            sep="\t",
            index=False,
        )

        # add cluster nr to dataframe
        sample_unique["cluster"] = -1
        for cluster_nr, cluster_idx in cluster_dict.items():
            sample_unique.iloc[
                cluster_idx, sample_unique.columns.get_loc("cluster")
            ] = cluster_nr

        sascore = Property("SA")
        sample_unique["SA"] = sascore(
            [Chem.MolFromSmiles(smiles) for smiles in sample_unique["SMILES"]]
        )

        # calculate tanimoto distances to all datasets and self
        def calculate_tanimoto_distances(list_of_smiles, dist_to_smiles, suff):
            tanimoto = TanimotoDistances(
                list_of_smiles=dist_to_smiles,
                fingerprint_type=MorganFP(radius=3, nBits=2048),
            )
            tan_dists = tanimoto(list_of_smiles, props={"QSPRID": sample_unique.index})
            tan_dists.index = sample_unique.SMILES.to_list()

            # fill diagonal with nan if comparing to self
            if suff == "self":
                np.fill_diagonal(tan_dists.values, np.nan)

            sample_unique[f"min_tanimoto_dist_{suff}"] = np.nanmin(tan_dists, axis=1)
            sample_unique[f"closest_molecule_{suff}"] = tan_dists.idxmin(axis=1).values
            sample_unique[f"mean_tanimoto_dist_{suff}"] = np.nanmean(tan_dists, axis=1)
            sample_unique[f"median_tanimoto_dist_{suff}"] = np.nanmedian(
                tan_dists, axis=1
            )

        calculate_tanimoto_distances(
            sample_unique.SMILES.to_list(), sample_unique.SMILES.to_list(), "self"
        )
        calculate_tanimoto_distances(
            sample_unique.SMILES.to_list(), A2AR_smiles, "A2AR"
        )
        calculate_tanimoto_distances(sample_unique.SMILES.to_list(), CL_smiles, "CL")
        calculate_tanimoto_distances(sample_unique.SMILES.to_list(), FU_smiles, "FU")
        calculate_tanimoto_distances(
            sample_unique.SMILES.to_list(), VDSS_smiles, "VDSS"
        )

        # merge with original sample
        sample = sample.reindex(columns=sample_unique.columns)
        sample.loc[sample_unique.index] = sample_unique

        # add columns for combinations of conditions
        sample["Unique & Valid"] = ((sample.Unique == 1) & (sample.Valid == 1)).astype(
            int
        )
        sample["Unique & Valid & Applicable"] = (
            (sample["Unique & Valid"] == 1) & (sample.Applicable == 1)
        ).astype(int)
        sample["Unique & Valid & Applicable & Novel"] = (
            (sample["Unique & Valid & Applicable"] == 1) & (sample.Novel == 1)
        ).astype(int)

        # save generated molecules with all scores
        sample.to_csv(
            join(GENERATED_DIR, f"{name}/generated_{self.num_samples}.tsv"),
            sep="\t",
            index=False,
        )

        # make statistics
        stats = sample.describe()
        stats.to_csv(
            join(GENERATED_DIR, f"{name}/generated_{self.num_samples}_stats.tsv"),
            sep="\t",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DrugEx reinforcement learning for all scenarios"
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="config.json",
        help="Path to the configuration file",
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default="drugex_pretrained/Papyrus05.5_smiles_rnn_PT.pkg",
        help="Path to the pretrained model",
    )
    parser.add_argument(
        "--vocab_file",
        type=str,
        default="drugex_pretrained/Papyrus05.5_smiles_rnn_PT.vocab",
        help="Path to the vocabulary file",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=1,
        help="GPU to use",
    )
    parser.add_argument(
        "--n_proc",
        type=int,
        default=5,
        help="Number of processors to use",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch size",
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
    # raw data
    PRETRAINED_MODEL = join(
        config["BASE_DIR"], config["RAW_DATA_DIR"], args.pretrained_model
    )
    VOCAB_FILE = join(config["BASE_DIR"], config["RAW_DATA_DIR"], args.vocab_file)

    # QSPR models and data
    QSPR_DIR = join(config["BASE_DIR"], config["PROCESSED_DATA_DIR"], "QSPR")
    data_dir = join(QSPR_DIR, "data")
    data_a2ar = join(data_dir, config["A2AR_DATA"], f"{config['A2AR_DATA']}_meta.json")
    data_fu = join(data_dir, config["FU_DATA"], f"{config['FU_DATA']}_meta.json")
    data_vdss = join(data_dir, config["VDSS_DATA"], f"{config['VDSS_DATA']}_meta.json")
    data_cl = join(data_dir, config["CL_DATA"], f"{config['CL_DATA']}_meta.json")

    # DNDD folders
    FINETUNING_DIR = join(
        config["BASE_DIR"], config["PROCESSED_DATA_DIR"], "DNDD", "finetuned"
    )
    REINFORCE_DIR = join(
        config["BASE_DIR"], config["PROCESSED_DATA_DIR"], "DNDD", "reinforced"
    )
    GENERATED_DIR = join(
        config["BASE_DIR"], config["PROCESSED_DATA_DIR"], "DNDD", "generated"
    )

    # save conda environment to outdir
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(GENERATED_DIR, exist_ok=True)
    env_file = join(GENERATED_DIR, f"conda_env_{now}.yml")
    export_conda_environment(env_file)

    # Set up logging
    logSettings = enable_file_logger(
        log_folder=GENERATED_DIR,
        filename=f"Generated_{now}.log",
        log_name=__name__,
        debug=False,
        disable_existing_loggers=False,
        init_data={
            "PRETRAINED_MODEL": join(config["RAW_DATA_DIR"], args.pretrained_model),
            "QSPR_DIR": join(config["PROCESSED_DATA_DIR"], "QSPR"),
            "A2AR_model": config["A2AR_MODEL"],
            "FU_model": config["FU_MODEL"],
            "VDSS_model": config["VDSS_MODEL"],
            "CL_model": config["CL_MODEL"],
            "FINETUNING_DIR": join(config["PROCESSED_DATA_DIR"], "DNDD", "finetuned"),
            "REINFORCE_DIR": join(config["PROCESSED_DATA_DIR"], "DNDD", "reinforced"),
            "GENERATED_DIR": join(config["PROCESSED_DATA_DIR"], "DNDD", "generated"),
            "GPU": args.gpu,
            "N_PROC": args.n_proc,
            "N_REPETITIONS": args.n_reps,
            "BATCH_SIZE": args.batch_size,
            "N_SAMPLES": args.n_samples,
        },
    )
    logger = logSettings.log

    # Save the current git commit
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    logger.info(f"Git commit: {sha}")

    voc = VocSmiles.fromFile(VOCAB_FILE, encode_frags=False)

    environ_dict, all_scorers_env = get_environ_dict(
        QSPR_PATH=join(QSPR_DIR, "models"),
        A2AR_model=config["A2AR_MODEL"],
        FU_model=config["FU_MODEL"],
        VDSS_model=config["VDSS_MODEL"],
        CL_model=config["CL_MODEL"],
        N_CPU=args.n_proc,
    )

    # load in smiles for all datasets
    A2AR_smiles = QSPRDataset.fromFile(data_a2ar).df.SMILES.to_list()
    CL_smiles = QSPRDataset.fromFile(data_cl).df.SMILES.to_list()
    FU_smiles = QSPRDataset.fromFile(data_fu).df.SMILES.to_list()
    VDSS_smiles = QSPRDataset.fromFile(data_vdss).df.SMILES.to_list()

    generate_and_score = GenerateAndScore(
        args.n_samples, args.batch_size, args.n_proc, all_scorers_env
    )

    pretrained = SequenceRNN(voc, is_lstm=True, use_gpus=[args.gpu])
    pretrained.loadStatesFromFile(PRETRAINED_MODEL)
    generate_and_score("pretrained", pretrained)

    for name in ["A2AR", "CL", "FU", "VDSS", "A2AR_CL", "A2AR_FU", "A2AR_VDSS"]:
        finetuned = SequenceRNN(voc, is_lstm=True, use_gpus=[args.gpu])
        finetuned.loadStatesFromFile(f"{FINETUNING_DIR}/{name}/finetuned.pkg")
        generate_and_score(f"{name}_finetuned", finetuned)

    # Generate for reinforced models for each environment and N repetitions
    for rep in range(args.n_reps):
        for name, environment in environ_dict.items():
            logger.info(
                f"Starting generating for {name} environment,"
                f" repetition {rep + 1}/{args.n_reps} at {datetime.datetime.now()}"
            )

            agent = SequenceRNN(voc, is_lstm=True, use_gpus=[args.gpu])
            agent.loadStatesFromFile(f"{REINFORCE_DIR}/{name}_{rep}/{name}_agent.pkg")
            generate_and_score(f"{name}_{rep}", agent, environment)

            logger.info(
                f"Finished generating for {name} environment,"
                f" repetition {rep + 1}/{args.n_reps} at {datetime.datetime.now()}"
            )

    logger.info(f"Finished generating molecules at {datetime.datetime.now()}")
