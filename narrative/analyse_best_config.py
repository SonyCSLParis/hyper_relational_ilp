# -*- coding: utf-8 -*-

"""Analysing best configs on fully inductive experiments"""
import os
import json
from collections import Counter

FILES = [x for x in os.listdir("best_configs/fully_inductive") if x.startswith("StarE")]

def load_json(fp):
    with open(fp, "r", encoding='utf-8') as f:
        data = json.load(f)
    return data

def main():
    res = {}
    for fp in FILES:
        data = load_json(os.path.join("best_configs/fully_inductive", fp))
        for k, v in data.items():
            if k != "Name" and not any(k.startswith(x) for x in ["test.", "inference_stats."]):
                if k in res:
                    res[k].append(v)
                else:
                    res[k] = [v]
    for k, v in res.items():
        print(k, Counter(v))



if __name__ == '__main__':
    main()
