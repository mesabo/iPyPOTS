#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2025/07/21

🚀 Welcome to the Awesome Python Script 🚀

User: Messou Franck Junior Aboya
Email: messouaboya17@gmail.com
Github: https://github.com/mesabo
Univ: Hosei University - IIST - (Tokyo, Japan)
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

import argparse
from pypots.utils.random import set_random_seed
import benchpots

# Pipelines
from pipeline.imputations.llm4imp import train_and_evaluate_llm4imp
from pipeline.imputations.saits import train_and_evaluate_saits
from pipeline.imputations.timellm import train_and_evaluate_timellm  # ✅ NEW

def prepare_physionet_dataset(missing_rate=0.1):
    set_random_seed()
    dataset = benchpots.datasets.preprocess_physionet2012(
        subset="set-a",
        rate=missing_rate,
    )
    print("✅ Dataset loaded with keys:", dataset.keys())
    return dataset


def main(args):
    print(f"🚀 Running imputation pipeline for model: {args.model}")
    dataset = prepare_physionet_dataset(args.missing_rate)

    model_name = args.model.lower()
    if model_name == "llm4imp":
        train_and_evaluate_llm4imp(dataset, saving_path=args.saving_path, device=args.device)
    elif model_name == "saits":
        train_and_evaluate_saits(dataset, saving_path=args.saving_path, device=args.device)
    elif model_name == "timellm":
        train_and_evaluate_timellm(dataset, saving_path=args.saving_path, device=args.device)
    else:
        raise ValueError(f"❌ Unknown model: {args.model}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run imputation models via PyPOTS pipeline")
    parser.add_argument("--model", type=str, default="llm4imp", help="Model name: llm4imp | saits | timellm")
    parser.add_argument("--missing_rate", type=float, default=0.1, help="Artificial missing rate in dataset")
    parser.add_argument("--saving_path", type=str, default="output/imputation/llm4imp", help="Path to save results")
    parser.add_argument("--device", type=str, default=None, help="Device: 'cpu', 'cuda', or None (auto)")

    args = parser.parse_args()
    main(args)