# -*- coding: utf-8 -*-

""" Analyse experiments 

- AMR: adjusted arithmetic mean rank (lower is better)
- MRR: mean reciprocal rank / inverse harmonic mean rank (higher is better)

"""
import os
import re
import click
from tqdm import tqdm
import pandas as pd

PATTERN = re.compile(
    r"lr_(?P<lr>\d+\.\d+)_ed_(?P<embedding_dim>\d+)_bs_(?P<batch_size>\d+)_kg_base_"
    r"prop_(?P<prop>\d+)_subevent_(?P<subevent>\d+)_role_(?P<role>\d+)_"
    r"causation_(?P<causation>\d+)_syntax_hyper_relational_rdf_star\.csv"
)
METRICS = [
    "validation.both.realistic.adjusted_arithmetic_mean_rank",  # lower is better
    "validation.both.realistic.inverse_harmonic_mean_rank",
    "validation.both.realistic.hits_at_1",
    "validation.both.realistic.hits_at_3",
    "validation.both.realistic.hits_at_5",
    "validation.both.realistic.hits_at_10"
]

def get_info_df(df_p):
    df = pd.read_csv(df_p, header=None)
    df.columns = ["metric", "epoch", "name", "value"]
    best_epoch = df[df.name==METRICS[0]].sort_values(by="value").epoch.values[0]
    params = {"epoch": df.epoch.max(), "best_epoch": best_epoch}
    params.update({m: df[(df.epoch==best_epoch) & (df.name==m)]["value"].values[0] for m in METRICS})
    return params


@click.command()
@click.option("--folder-in", default="narrative/experiments", help="Folder containing the experiment logs")
@click.option("--folder-out", default="narrative/results", help="Folder to store results")
def main(folder_in, folder_out):
    exps = [x for x in os.listdir(folder_in) if x.startswith("lr") and x.endswith(".csv")]
    exps = [x for x in exps if "role_0" in x]
    data = []
    for exp in tqdm(exps):
        params = PATTERN.match(exp).groupdict()
        params = {k: int(v) if k.isdigit() else float(v) for k, v in params.items()}
        params.update({"exp": exp})
        params.update(get_info_df(os.path.join(folder_in, exp)))
        data.append(params)
    df = pd.DataFrame(data)
    if not os.path.exists(folder_out):
        os.makedirs(folder_out)
    df.to_csv(os.path.join(folder_out, "results.csv"))



if __name__ == "__main__":
    main()
