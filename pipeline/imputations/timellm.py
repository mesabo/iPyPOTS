#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2025/07/23

🚀 Welcome to the Awesome Python Script 🚀

User: Messou Franck Junior Aboya
Email: messouaboya17@gmail.com
Github: https://github.com/mesabo
Univ: Hosei University - IIST - (Tokyo, Japan)
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

# pipeline/imputations/timellm.py

import numpy as np
from pypots.nn.functional import calc_mae
from pypots.optim import Adam
from pypots.imputation import TimeLLM


def train_and_evaluate_timellm(dataset: dict, saving_path: str = None, device: str = None):
    # 1. Assemble train/val/test datasets
    dataset_for_IMPU_training = {"X": dataset["train_X"]}
    dataset_for_IMPU_validating = {
        "X": dataset["val_X"],
        "X_ori": dataset["val_X_ori"],
    }
    dataset_for_IMPU_testing = {"X": dataset["test_X"]}

    test_X_ori = np.nan_to_num(dataset["test_X_ori"])
    test_X_indicating_mask = (
        np.isnan(dataset["test_X_ori"]) ^ np.isnan(dataset["test_X"])
    )

    # 2. Initialize TimeLLM model
    timellm = TimeLLM(
        n_steps=dataset["n_steps"],
        n_features=dataset["n_features"],
        llm_model_type="GPT2",
        n_layers=1,
        patch_size=8,
        patch_stride=4,
        d_llm=768,
        d_model=256,
        d_ffn=128,
        n_heads=4,
        dropout=0.1,
        domain_prompt_content="PhysioNet ICU",
        ORT_weight=1,
        MIT_weight=1,
        batch_size=32,
        epochs=10,
        patience=3,
        optimizer=Adam(lr=1e-3),
        num_workers=0,
        device=device,
        saving_path=saving_path,
        model_saving_strategy="best",
    )

    # 3. Train the model
    timellm.fit(
        train_set=dataset_for_IMPU_training,
        val_set=dataset_for_IMPU_validating,
    )

    # 4. Test the model
    timellm_results = timellm.predict(dataset_for_IMPU_testing)
    timellm_imputation = timellm_results["imputation"]

    # 5. Evaluate the model
    testing_mae = calc_mae(
        timellm_imputation,
        test_X_ori,
        test_X_indicating_mask,
    )
    print(f"[TimeLLM] Testing MAE: {testing_mae:.4f}")