# -*- coding: utf-8 -*-

"""Running some first models as tests (to see if runs out of memory).
Default settings are used."""
import os
import subprocess
from loguru import logger

VP = os.path.expanduser("~/.data/ilp/NarrativeInductiveDataset/inductive/statements/")
VERSIONS = os.listdir(VP)
MODELS = ["qblp", "stare"]
FP = "./narrative/terminal_output"

def main():
    """Runs ILP models on NarrativeInductive dataset versions.
    Logs output to console and files in the FP directory."""
    for m in MODELS:
        for v in VERSIONS:
            logger.info(f"Running {m} on {v}")
            command = f"ilp run {m} --dataset-name narrativeinductive --dataset-version {v} 2>&1 | tee {os.path.join(FP, m + '_' + v + '.log')}"
            subprocess.run(command, shell=True, check=False)
            logger.info(f"Finished running {m} on {v}")
    print(VERSIONS)

if __name__ == "__main__":
    main()
