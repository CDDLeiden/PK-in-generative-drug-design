import argparse
import datetime
import json
import os
from os.path import join

import git
from _drugex_environment import get_environ_dict
from drugex.data.corpus.vocabulary import VocSmiles
from drugex.logs import logger
from drugex.training.explorers import SequenceExplorer
from drugex.training.generators import SequenceRNN
from drugex.training.monitors import FileMonitor
from qsprpred.logs.utils import enable_file_logger, export_conda_environment


def reinforce_agent(
    epsilon,
    mutate_file,
    GRID_SEARCH_DIR,
    GPUS,
    LR,
    BATCH_SIZE,
    N_SAMPLES,
    EPOCHS,
    PATIENCE,
    MIN_EPOCHS,
    RELOAD_INTERVAL,
    GENERATE_N_SAMPLES,
    N_PROC,
):
    type_mutate = "pretrained" if mutate_file == PRETRAINED_MODEL else "finetuned"
    name = f"epsilon{epsilon}_mutate{type_mutate}"

    logger.info(
        f"Starting reinforcement learning for {name} environment,"
        f" at {datetime.datetime.now()}"
    )

    mutate = SequenceRNN(voc, is_lstm=True, use_gpus=GPUS)
    mutate.loadStatesFromFile(mutate_file)

    agent = SequenceRNN(voc, is_lstm=True, use_gpus=GPUS, lr=LR)
    agent.loadStatesFromFile(FINETUNED_MODEL)

    # Set up the explorer (agent, environment, etc.)
    explorer = SequenceExplorer(
        agent=agent,
        env=environment,
        mutate=mutate,
        epsilon=epsilon,
        use_gpus=GPUS,
        batch_size=BATCH_SIZE,
        n_samples=N_SAMPLES,
    )
    monitor = FileMonitor(
        join(GRID_SEARCH_DIR, f"{name}/{name}_agent"),
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
    logger.info(
        f"Finished reinforcement learning for {name} environment,"
        f" at {datetime.datetime.now()}"
    )

    os.makedirs(join(GRID_SEARCH_DIR, name), exist_ok=True)
    sample = agent.generate(
        evaluator=environment,
        num_samples=GENERATE_N_SAMPLES,
        batch_size=BATCH_SIZE,
        n_proc=N_PROC,
        raw_scores=False,
        drop_duplicates=False,
        drop_invalid=False,
    )
    sample.to_csv(
        join(GRID_SEARCH_DIR, f"{name}/{name}_sample.tsv"), sep="\t", index=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DrugEx reinforcement learning grid search"
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
        "--generate_n_samples",
        type=int,
        default=10000,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="Learning rate",
    )
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        config = json.load(f)

    # set paths
    PRETRAINED_MODEL = join(
        config["BASE_DIR"], config["RAW_DATA_DIR"], args.pretrained_model
    )
    VOCAB_FILE = join(config["BASE_DIR"], config["RAW_DATA_DIR"], args.vocab_file)
    FINETUNED_MODEL = join(
        config["BASE_DIR"],
        config["PROCESSED_DATA_DIR"],
        "DNDD",
        "finetuned",
        "A2AR_FU/finetuned.pkg",
    )
    GRID_SEARCH_DIR = join(
        config["BASE_DIR"], config["PROCESSED_DATA_DIR"], "DNDD", "grid_search"
    )

    QSPR_MODEL_DIR = join(
        config["BASE_DIR"], config["PROCESSED_DATA_DIR"], "QSPR", "models"
    )

    # save conda environment to outdir
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(join(GRID_SEARCH_DIR), exist_ok=True)
    env_file = join(GRID_SEARCH_DIR, f"conda_env_{now}.yml")
    export_conda_environment(env_file)

    # Set up logging
    logSettings = enable_file_logger(
        log_folder=f"{GRID_SEARCH_DIR}",
        filename=f"GridSearch_{now}.log",
        log_name=__name__,
        debug=False,
        disable_existing_loggers=False,
        init_data={
            "PRETRAINED_MODEL": join(config["RAW_DATA_DIR"], args.pretrained_model),
            "FINETUNING_MODEL": join(
                config["PROCESSED_DATA_DIR"],
                "DNDD",
                "finetuned",
                "A2AR_FU",
                "finetuned.pkg",
            ),
            "GRID_SEARCH_DIR": join(
                config["PROCESSED_DATA_DIR"], "DNDD", "grid_search"
            ),
            "GPU": args.gpu,
            "N_PROC": args.n_proc,
            "BATCH_SIZE": args.batch_size,
            "EPOCHS": args.epochs,
            "PATIENCE": args.patience,
            "MIN_EPOCHS": args.min_epochs,
            "RELOAD_INTERVAL": args.reload_interval,
            "N_SAMPLES": args.n_samples,
            "GENERATE_N_SAMPLES": args.generate_n_samples,
            "LR": args.lr,
        },
    )
    logger = logSettings.log

    # Save the current git commit
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    logger.info(f"Git commit: {sha}")
    logger.info("GPU devices: {}".format([args.gpu]))
    logger.info("Batch size: {}".format(args.batch_size))

    voc = VocSmiles.fromFile(VOCAB_FILE, encode_frags=False)

    environ_dict, _ = get_environ_dict(
        QSPR_PATH=QSPR_MODEL_DIR,
        A2AR_model=config["A2AR_MODEL"],
        FU_model=config["FU_MODEL"],
        N_CPU=args.n_proc,
    )
    environment = environ_dict["A2AR_FUmax"]

    # Define grid search
    grid_search = {
        "epsilon": [0.01, 0.05, 0.1, 0.2, 0.3],
        "mutate_file": [PRETRAINED_MODEL, FINETUNED_MODEL],
    }

    # Run grid search
    for epsilon in grid_search["epsilon"]:
        for mutate_file in grid_search["mutate_file"]:
            reinforce_agent(
                epsilon,
                mutate_file,
                GRID_SEARCH_DIR,
                [args.gpu],
                args.lr,
                args.batch_size,
                args.n_samples,
                args.epochs,
                args.patience,
                args.min_epochs,
                args.reload_interval,
                args.generate_n_samples,
                args.n_proc,
            )
