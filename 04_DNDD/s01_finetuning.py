# import packages
import argparse
import datetime
import json
import os
from os.path import join

import git
import pandas as pd
from drugex.data.corpus.corpus import SequenceCorpus
from drugex.data.corpus.vocabulary import VocSmiles
from drugex.data.datasets import SmilesDataSet
from drugex.data.processing import CorpusEncoder, RandomTrainTestSplitter
from drugex.logs import logger
from drugex.training.generators import SequenceRNN
from drugex.training.monitors import FileMonitor
from qsprpred.data import QSPRDataset
from qsprpred.logs.utils import enable_file_logger, export_conda_environment
from qsprpred.models import QSPRModel


def main(
    vocab_file: str,
    pretrained_model: str,
    finetuning_dir: str,
    gpus: list[int],
    batch_size: int,
    epochs: int,
    n_proc: int,
    chunk_size: int,
    lr: float,
    model_data_dict: dict,
    environ_list: list[list[str]],
):

    voc = VocSmiles.fromFile(vocab_file, encode_frags=False)

    # Run the reinforcement learning loop for each environment and N repetitions
    for environ in environ_list:
        logger.info(
            f"Starting finetuning for {environ} environment at {datetime.datetime.now()}"
        )

        # load in the dataset
        df_train = QSPRDataset.fromFile(model_data_dict[environ[0]][1]).df
        smiles_train = df_train.SMILES.values

        # remove smiles outside of applicability domain environ
        for prop in environ:
            model = QSPRModel.fromFile(model_data_dict[prop][0])
            scores, applicability = model.predictMols(
                smiles_train,
                use_applicability_domain=True,
            )
            applicability = applicability.flatten()
            smiles_train = smiles_train[applicability]
            df_train = df_train.iloc[applicability]
            df_train[f"{prop}_pred"] = scores[applicability]
            logger.info(
                f"Removed {len(applicability) - sum(applicability)} smiles by applicability domain for {prop} model"
            )
        logger.info(f"Remaining smiles: {len(smiles_train)}")
        smiles_train = list(smiles_train)

        # make file prefix from environ
        prefix = "_".join(environ)
        os.makedirs(finetuning_dir, exist_ok=True)
        os.makedirs(join(finetuning_dir, f"{prefix}"), exist_ok=True)
        df_train.to_csv(join(finetuning_dir, f"{prefix}/train_df.tsv"), sep="\t")

        # encode the smiles
        encoder = CorpusEncoder(
            SequenceCorpus,
            {"vocabulary": voc, "update_voc": False, "throw": True},
            n_proc=n_proc,
            chunk_size=chunk_size,
        )
        data_collector = SmilesDataSet(
            join(finetuning_dir, f"{prefix}/ligand_corpus.tsv"), rewrite=True
        )
        encoder.apply(smiles_train, collector=data_collector)

        # Split the data into train and validation sets
        splitter = RandomTrainTestSplitter(0.1, 1e4)
        train, test = splitter(data_collector.getData())
        for data, name in zip([train, test], ["train", "test"]):
            pd.DataFrame(data, columns=data_collector.getColumns()).to_csv(
                join(finetuning_dir, f"{prefix}/ligand_{name}.tsv"),
                header=True,
                index=False,
                sep="\t",
            )

        data_set_train = SmilesDataSet(
            join(finetuning_dir, f"{prefix}/ligand_train.tsv"), voc=voc
        )
        data_set_train.voc = voc
        train_loader = data_set_train.asDataLoader(batch_size=batch_size)

        data_set_test = SmilesDataSet(
            join(finetuning_dir, f"{prefix}/ligand_test.tsv"), voc=voc
        )
        data_set_test.voc = voc
        valid_loader = data_set_test.asDataLoader(batch_size=batch_size)

        # finetune the model
        ft_path = join(finetuning_dir, f"{prefix}/finetuned")
        finetuned = SequenceRNN(voc, is_lstm=True, use_gpus=gpus, lr=lr)
        finetuned.loadStatesFromFile(pretrained_model)
        monitor = FileMonitor(ft_path, save_smiles=True, reset_directory=True)
        finetuned.fit(train_loader, valid_loader, epochs=epochs, monitor=monitor)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune the DrugEx model")
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
        "--lr",
        type=float,
        default=0.0001,
        help="Learning rate",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="Number of epochs",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1000,
        help="Chunk size",
    )
    parser.add_argument(
        "--n_proc",
        type=int,
        default=4,
        help="Number of processors to use",
    )
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        config = json.load(f)

    # set paths
    pretrained_model = join(
        config["BASE_DIR"], config["RAW_DATA_DIR"], args.pretrained_model
    )
    vocab_file = join(config["BASE_DIR"], config["RAW_DATA_DIR"], args.vocab_file)
    finetuning_dir = join(
        config["BASE_DIR"], config["PROCESSED_DATA_DIR"], "DNDD", "finetuned"
    )
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

    # model_dict
    model_data_dict = {
        "A2AR": (model_a2ar, data_a2ar),
        "FU": (model_fu, data_fu),
        "VDSS": (model_vdss, data_vdss),
        "CL": (model_cl, data_cl),
    }

    # fine-tuning environments
    environ_list = [
        ["A2AR"],
        ["FU"],
        ["VDSS"],
        ["CL"],
        ["A2AR", "FU"],
        ["A2AR", "VDSS"],
        ["A2AR", "CL"],
    ]

    # save conda environment to outdir
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(join(finetuning_dir), exist_ok=True)
    env_file = join(finetuning_dir, f"conda_env_{now}.yml")
    export_conda_environment(env_file)

    # Set up logging
    logSettings = enable_file_logger(
        log_folder=f"{finetuning_dir}",
        filename=f"Finetuning_{now}.log",
        log_name=__name__,
        debug=False,
        disable_existing_loggers=False,
        init_data={
            "PRETRAINED_MODEL": join(config["RAW_DATA_DIR"], args.pretrained_model),
            "FINETUNING_DIR": join(config["PROCESSED_DATA_DIR"], "DNDD", "finetuned"),
            "MODEL_A2AR": config["A2AR_MODEL"],
            "DATA_A2AR": config["A2AR_DATA"],
            "MODEL_FU": config["FU_MODEL"],
            "DATA_FU": config["FU_DATA"],
            "MODEL_VDSS": config["VDSS_MODEL"],
            "DATA_VDSS": config["VDSS_DATA"],
            "MODEL_CL": config["CL_MODEL"],
            "DATA_CL": config["CL_DATA"],
            "GPU": args.gpu,
            "BATCH_SIZE": args.batch_size,
            "EPOCHS": args.epochs,
            "N_PROC,": args.n_proc,
            "CHUNK_SIZE": args.chunk_size,
            "LR": args.lr,
        },
    )
    logger = logSettings.log
    logger.info

    # Save the current git commit
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    logger.info(f"Git commit: {sha}")

    main(
        vocab_file,
        pretrained_model,
        finetuning_dir,
        [args.gpu],
        args.batch_size,
        args.epochs,
        args.n_proc,
        args.chunk_size,
        args.lr,
        model_data_dict,
        environ_list,
    )
