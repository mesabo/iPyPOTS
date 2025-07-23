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

import numpy as np
from pypots.nn.functional import calc_mae
from pypots.optim import Adam
from pypots.imputation.llm4imp import LLM4IMP


def train_and_evaluate_llm4imp(dataset, saving_path="output/imputation/llm4imp", device=None):
    # 1. Assemble train/val/test datasets
    dataset_for_IMPU_training = {
        "X": dataset["train_X"],
    }
    dataset_for_IMPU_validating = {
        "X": dataset["val_X"],
        "X_ori": dataset["val_X_ori"],
    }
    dataset_for_IMPU_testing = {
        "X": dataset["test_X"],
    }

    test_X_ori = np.nan_to_num(dataset["test_X_ori"])
    test_X_indicating_mask = (
        np.isnan(dataset["test_X_ori"]) ^ np.isnan(dataset["test_X"])
    )

    # 2. Initialize model
    model = LLM4IMP(
        n_steps=dataset["n_steps"],
        n_features=dataset["n_features"],
        patch_size=12,
        patch_stride=6,
        d_model=64,
        d_ffn=128,
        d_llm=768,
        n_heads=4,
        dropout=0.1,
        batch_size=32,
        epochs=10,
        patience=3,
        optimizer=Adam(lr=1e-3),
        device=device,
        saving_path=saving_path,
        model_saving_strategy="best",
    )

    # 3. Train
    model.fit(
        train_set=dataset_for_IMPU_training,
        val_set=dataset_for_IMPU_validating,
    )

    # 4. Test
    results = model.predict(dataset_for_IMPU_testing)
    imputation = results["imputation"]

    # 5. Evaluate
    mae = calc_mae(imputation, test_X_ori, test_X_indicating_mask)
    print(f"[LLM4IMP] Testing MAE: {mae:.4f}")