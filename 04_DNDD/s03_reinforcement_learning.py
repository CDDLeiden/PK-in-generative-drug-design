"""Reinforcement learning for all scenarios"""

import argparse
import datetime
import json
import os
from os.path import join

import git
from _drugex_environment import environ_finetune_dict, get_environ_dict
from drugex.data.corpus.vocabulary import VocSmiles
from drugex.logs import logger
from drugex.training.explorers import SequenceExplorer
from drugex.training.generators import SequenceRNN
from drugex.training.monitors import FileMonitor
from qsprpred.logs.utils import enable_file_logger, export_conda_environment


def reinforcement_learning(
    voc,
    name,
    environment,
    rep,
    PRETRAINED_MODEL,
    FINETUNING_DIR,
    REINFORCE_DIR,
    GPUS,
    LR,
    BATCH_SIZE,
    N_SAMPLES,
    EPOCHS,
    PATIENCE,
    MIN_EPOCHS,
    RELOAD_INTERVAL,
    EPSILON,
    MUTATE,
):
    finetuned = join(FINETUNING_DIR, environ_finetune_dict[name], "finetuned.pkg")
    agent = SequenceRNN(voc, is_lstm=True, use_gpus=GPUS, lr=LR)
    agent.loadStatesFromFile(finetuned)

    mutate = SequenceRNN(voc, is_lstm=True, use_gpus=GPUS)
    if MUTATE == "pretrained":
        mutate.loadStatesFromFile(PRETRAINED_MODEL)
    else:
        mutate.loadStatesFromFile(finetuned)

    # Set up the explorer (agent, environment, etc.)
    explorer = SequenceExplorer(
        agent=agent,
        env=environment,
        mutate=mutate,
        epsilon=EPSILON,
        use_gpus=GPUS,
        batch_size=BATCH_SIZE,
        n_samples=N_SAMPLES,
    )
    monitor = FileMonitor(
        join(REINFORCE_DIR, f"{name}_{rep}/{name}_agent"),
        save_smiles=True,
        reset_directory=True,
    )
    # Reinforcement learning
    explorer.fit(
        monitor=monitor,
        epochs=EPOCHS,
        patience=PATIENCE,
        min_epochs=MIN_EPOCHS,
        criteria="avg_amean",
        reload_interval=RELOAD_INTERVAL,
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
        "--epochs",
        type=int,
        default=2000,
        help="Number of epochs",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=300,
        help="Patience",
    )
    parser.add_argument(
        "--min_epochs",
        type=int,
        default=200,
        help="Minimum number of epochs",
    )
    parser.add_argument(
        "--reload_interval",
        type=int,
        default=200,
        help="Reload interval",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=2000,
        help="Number of samples",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="Learning rate",
    )
    parser.add_argument(
        "--n_reps",
        type=int,
        default=3,
        help="Number of repetitions",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="Epsilon value",
    )
    parser.add_argument(
        "--mutate",
        type=str,
        default="finetuned",
        help="Mutation network, pretrained or finetuned",
        choices=["pretrained", "finetuned"],
    )
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        config = json.load(f)

    # set paths
    PRETRAINED_MODEL = join(
        config["BASE_DIR"], config["RAW_DATA_DIR"], args.pretrained_model
    )
    VOCAB_FILE = join(config["BASE_DIR"], config["RAW_DATA_DIR"], args.vocab_file)
    FINETUNING_DIR = join(
        config["BASE_DIR"], config["PROCESSED_DATA_DIR"], "DNDD", "finetuned"
    )
    QSPR_MODEL_DIR = join(
        config["BASE_DIR"], config["PROCESSED_DATA_DIR"], "QSPR", "models"
    )
    REINFORCE_DIR = join(
        config["BASE_DIR"], config["PROCESSED_DATA_DIR"], "DNDD", "reinforced"
    )

    # save conda environment to outdir
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(REINFORCE_DIR, exist_ok=True)
    env_file = join(REINFORCE_DIR, f"conda_env_{now}.yml")
    export_conda_environment(env_file)

    # Set up logging
    logSettings = enable_file_logger(
        log_folder=f"{REINFORCE_DIR}",
        filename=f"ReinforcementLearning_{now}.log",
        log_name=__name__,
        debug=False,
        disable_existing_loggers=False,
        init_data={
            "PRETRAINED_MODEL": join(config["RAW_DATA_DIR"], args.pretrained_model),
            "FINETUNING_DIR": join(config["PROCESSED_DATA_DIR"], "DNDD", "finetuned"),
            "REINFORCE_DIR": join(config["PROCESSED_DATA_DIR"], "DNDD", "reinforced"),
            "GPU": args.gpu,
            "N_PROC": args.n_proc,
            "BATCH_SIZE": args.batch_size,
            "EPOCHS": args.epochs,
            "PATIENCE": args.patience,
            "MIN_EPOCHS": args.min_epochs,
            "RELOAD_INTERVAL": args.reload_interval,
            "N_SAMPLES": args.n_samples,
            "LR": args.lr,
            "N_REPETITIONS": args.n_reps,
            "EPSILON": args.epsilon,
            "MUTATE": args.mutate,
        },
    )
    logger = logSettings.log

    # Save the current git commit
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    logger.info(f"Git commit: {sha}")

    voc = VocSmiles.fromFile(VOCAB_FILE, encode_frags=False)

    environ_dict, _ = get_environ_dict(
        QSPR_PATH=QSPR_MODEL_DIR,
        A2AR_model=config["A2AR_MODEL"],
        FU_model=config["FU_MODEL"],
        VDSS_model=config["VDSS_MODEL"],
        CL_model=config["CL_MODEL"],
        N_CPU=args.n_proc,
    )

    # Run the reinforcement learning loop for each environment and N repetitions
    for rep in range(args.n_reps):
        for name, environment in environ_dict.items():
            if os.path.exists(join(REINFORCE_DIR, f"{name}_{rep}")):
                logger.info(
                    f"Repetition {rep + 1}/{args.n_reps} for {name} environment already exists, skipping"
                )
                continue
            logger.info(
                f"Starting reinforcement learning for {name} environment,"
                f" repetition {rep + 1}/{args.n_reps} at {datetime.datetime.now()}"
            )
            reinforcement_learning(
                voc,
                name,
                environment,
                rep,
                PRETRAINED_MODEL,
                FINETUNING_DIR,
                REINFORCE_DIR,
                [args.gpu],
                args.lr,
                args.batch_size,
                args.n_samples,
                args.epochs,
                args.patience,
                args.min_epochs,
                args.reload_interval,
                args.epsilon,
                args.mutate,
            )
            logger.info(
                f"Finished reinforcement learning for {name} environment,"
                f" repetition {rep + 1}/{args.n_reps} at {datetime.datetime.now()}"
            )
