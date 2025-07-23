"""
The core wrapper assembles the submodules of LLM4IMP imputation model
and takes over the forward progress of the algorithm.
"""

# Created by Franck Junior Aboya <mesabo18@gmail.com / messouaboya17@gmail.com>
# License: BSD-3-Clause

import torch
from ...nn.modules import ModelCore
from ...nn.modules.loss import Criterion
from ...nn.modules.llm4imp.backbone import BackboneLLM4IMP


class _LLM4IMP(ModelCore):
    def __init__(
        self,
        n_steps: int,
        n_features: int,
        patch_size: int,
        patch_stride: int,
        d_model: int,
        d_ffn: int,
        d_llm: int,
        n_heads: int,
        dropout: float,
        prompt_template: str,
        device: str,
        training_loss: Criterion,
        validation_metric: Criterion,
    ):
        super().__init__()
        self.n_steps = n_steps
        self.n_features = n_features
        self.training_loss = training_loss
        if validation_metric.__class__.__name__ == "Criterion":
            self.validation_metric = self.training_loss
        else:
            self.validation_metric = validation_metric

        self.backbone = BackboneLLM4IMP(
            n_steps=n_steps,
            n_features=n_features,
            patch_size=patch_size,
            patch_stride=patch_stride,
            d_model=d_model,
            d_ffn=d_ffn,
            d_llm=d_llm,
            n_heads=n_heads,
            dropout=dropout,
            prompt_template=prompt_template,
        ).to(device)

    def forward(
        self,
        inputs: dict,
        calc_criterion: bool = False,
    ) -> dict:
        X = inputs["X"]  # [B, T, D]
        missing_mask = inputs["missing_mask"]  # [B, T, D]

        reconstruction = self.backbone(X, missing_mask)  # [B, T, D]
        imputed_data = missing_mask * X + (1 - missing_mask) * reconstruction

        results = {
            "imputation": imputed_data,
            "reconstruction": reconstruction,
        }

        if calc_criterion:
            if self.training:
                X_ori, indicating_mask = inputs["X_ori"], inputs["indicating_mask"]
                results["loss"] = self.training_loss(reconstruction, X_ori, indicating_mask)
            else:
                X_ori, indicating_mask = inputs["X_ori"], inputs["indicating_mask"]
                results["metric"] = self.validation_metric(reconstruction, X_ori, indicating_mask)

        return results
