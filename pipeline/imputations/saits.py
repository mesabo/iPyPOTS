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

# pipeline/imputations/saits.py

import numpy as np
from pypots.nn.functional import calc_mae
from pypots.optim import Adam
from pypots.imputation import SAITS

def train_and_evaluate_saits(dataset, saving_path="output/imputation/saits", device=None):
    print("🚀 Starting SAITS training pipeline...")

    # 1. Assemble train/val/test datasets
    train_set = {"X": dataset["train_X"]}
    val_set = {
        "X": dataset["val_X"],
        "X_ori": dataset["val_X_ori"],
    }
    test_set = {"X": dataset["test_X"]}
    test_X_ori = np.nan_to_num(dataset["test_X_ori"])
    test_X_indicating_mask = np.isnan(dataset["test_X_ori"]) ^ np.isnan(dataset["test_X"])

    # 2. Initialize model
    saits = SAITS(
        n_steps=dataset["n_steps"],
        n_features=dataset["n_features"],
        n_layers=1,
        d_model=256,
        d_ffn=128,
        n_heads=4,
        d_k=64,
        d_v=64,
        dropout=0.1,
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

    # 3. Fit
    saits.fit(train_set=train_set, val_set=val_set)

    # 4. Predict
    results = saits.predict(test_set)
    imputations = results["imputation"]

    # 5. Evaluate
    testing_mae = calc_mae(imputations, test_X_ori, test_X_indicating_mask)
    print(f"[✅ SAITS] Testing MAE: {testing_mae:.4f}")