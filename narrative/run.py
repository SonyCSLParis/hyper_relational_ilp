# -*- coding: utf-8 -*-

"""Run all"""
import os
import subprocess
import argparse
from loguru import logger

# DS VERSIONS
VP = os.path.expanduser("~/.data/ilp/NarrativeInductiveDataset/inductive/statements/")
VERSIONS = sorted(os.listdir(VP))

# LOGS PATH
PYKEEN_LOGS_P = os.path.expanduser("~/.data/pykeen/logs/")
EXPS_LOGS_P = "./narrative/experiments"

# FIXED PARAMS
HIDDEN_DIMENSION = 704  # --transformer-hidden-dimension
CREATE_INVERSE_TRIPLES = True  # --create-inverse-triples
USE_BIAS = False  # --use-bias
TRIPLE_QUAL_WEIGHT = 0.8  # --triple-qual-weight
QUALIFIER_AGGREGATION = "attn"  # --qualifier-aggregation
NUM_TRANSFORMER_LAYERS = 2  # --transformer-num-layers
NUM_TRANSFORMER_HEADS = 4  # --transformer-num-heads
NUM_LAYERS = 2  # --num-layers
NUM_EPOCHS = 1_000  # --num-epochs
NUM_ATTENTION_HEADS = 2  # --attention-num-heads
MAX_NUM_QUALIFIER_PAIRS = 6  # --max-num-qualifier-pairs
LABEL_SMOOTHING = 0.1  # --label-smoothing
HID_DROP = 0.3  # --hidden-dropout
GCN_DROP = 0.3  # --gcn-dropout
EVALUATION_BATCH_SIZE = 10  # --eval-batch-size
ATTENTION_SLOPE = 0.4  # --attention-slope
ATTENTION_DROP = 0.4  # --attention-dropout


# DIFFERENT VALUES
LEARNING_RATE = [0.001, 0.002, 0.0001, 0.0005]  # --learning-rate
EMBEDDING_DIM = [160, 192, 224]  # --embedding-dimension
BATCH_SIZE = [192, 256, 896, 960]  # --batch-size

# COMMAND
COMMAND = f"""
ilp run stare --dataset-name narrativeinductive \
    --dataset-version <v> --learning-rate <lr> \
    --embedding-dimension <ed> --batch-size <bs> \
    --transformer-hidden-dimension {HIDDEN_DIMENSION} \
    --create-inverse-triples {CREATE_INVERSE_TRIPLES} \
    --use-bias {USE_BIAS} \
    --triple-qual-weight {TRIPLE_QUAL_WEIGHT} \
    --qualifier-aggregation {QUALIFIER_AGGREGATION} \
    --transformer-num-layers {NUM_TRANSFORMER_LAYERS} \
    --transformer-num-heads {NUM_TRANSFORMER_HEADS} \
    --num-layers {NUM_LAYERS} \
    --num-epochs {NUM_EPOCHS} \
    --attention-num-heads {NUM_ATTENTION_HEADS} \
    --max-num-qualifier-pairs {MAX_NUM_QUALIFIER_PAIRS} \
    --label-smoothing {LABEL_SMOOTHING} \
    --hidden-dropout {HID_DROP} \
    --gcn-dropout {GCN_DROP} \
    --eval-batch-size {EVALUATION_BATCH_SIZE} \
    --attention-slope {ATTENTION_SLOPE} \
    --attention-dropout {ATTENTION_DROP}
"""

def parse_args():
    parser = argparse.ArgumentParser(description="Run NarrativeInductive experiments")
    parser.add_argument("--gpu", type=str, default=None, help="GPU to use (e.g., 0 or 1)")
    # Parameters for grid search
    parser.add_argument("--versions", type=str, default=None, help="Comma-separated list of versions to run")
    parser.add_argument("--learning-rates", type=str, default=None, 
                        help="Comma-separated list of learning rates to try (e.g., 0.001,0.0001)")
    parser.add_argument("--embedding-dims", type=str, default=None,
                        help="Comma-separated list of embedding dimensions to try (e.g., 160,192)")
    parser.add_argument("--batch-sizes", type=str, default=None,
                        help="Comma-separated list of batch sizes to try (e.g., 192,256)")
    
    return parser.parse_args()


def cp_results_file(ftc_p, cf_p):
    command = f"cp {ftc_p} {cf_p}"
    subprocess.run(command, shell=True, check=False)

def prep_temp_dir(temp_dir, env, version):
    os.makedirs(temp_dir, exist_ok=True)
    folder = "ilp/NarrativeInductiveDataset"
    orig_dir = os.path.expanduser("~/.data/")
    os.makedirs(os.path.join(temp_dir, folder, "inductive/statements"), exist_ok=True)
    for fn in ["index.txt", "embeddings.pkl"]:
        command = f"cp {os.path.join(orig_dir, folder, fn)} {os.path.join(temp_dir, folder, fn)}"
        subprocess.run(command, shell=True, check=False, env=env)
    command = f"cp -r {os.path.join(orig_dir, folder, 'inductive/statements', version)} {os.path.join(temp_dir, folder, 'inductive/statements')}"
    subprocess.run(command, shell=True, check=False, env=env)

def main():
    args = parse_args()

    if args.versions:
        versions_to_run = args.versions.split(",")
        # Validate versions
        for v in versions_to_run:
            if v not in VERSIONS:
                logger.error(f"Version {v} not found in {VP}")
                return
    else:
        versions_to_run = VERSIONS

    if args.learning_rates:
        learning_rates = [float(lr) for lr in args.learning_rates.split(",")]
    else:
        learning_rates = LEARNING_RATE
    
    if args.embedding_dims:
        embedding_dims = [int(ed) for ed in args.embedding_dims.split(",")]
    else:
        embedding_dims = EMBEDDING_DIM
    
    if args.batch_sizes:
        batch_sizes = [int(bs) for bs in args.batch_sizes.split(",")]
    else:
        batch_sizes = BATCH_SIZE

    for lr in learning_rates:
        for ed in embedding_dims:
            for bs in batch_sizes:
                for v in versions_to_run:
                    fp = os.path.join(EXPS_LOGS_P, f"lr_{lr}_ed_{ed}_bs_{bs}_{v}.csv")
                    if not os.path.exists(fp):
                        logger.info(f"Running params: lr ({lr}), ed ({ed}), bs ({bs}) on {v}")

                        # environment variables
                        env = os.environ.copy()
                        if args.gpu is not None:
                            env["CUDA_VISIBLE_DEVICES"] = args.gpu
                        
                        # command
                        command = COMMAND.replace("<v>", v).replace("<lr>", str(lr)) \
                            .replace("<ed>", str(ed)).replace("<bs>", str(bs))
                        print(command)

                        # Use a temporary custom directory for PyKEEN to avoid conflicts
                        temp_dir = f"/tmp/pykeen_run_{v}_{lr}_{ed}_{bs}"
                        prep_temp_dir(temp_dir, env, v)
                        env["PYSTOW_HOME"] = temp_dir

                        subprocess.run(command, shell=True, check=False, env=env)
                        logger.info(f"Finished running {command}, saving to {fp}")
                        cp_results_file(sorted(os.listdir(os.path.join(temp_dir, "pykeen/logs")))[-1], fp)
                        logger.info(f"Copied results to {fp}")
                        subprocess.run(f"rm -rf {temp_dir}", shell=True, check=False, env=env)
                    else:
                        logger.info(f"File {fp} already exists, skipping.")


if __name__ == "__main__":
    main()