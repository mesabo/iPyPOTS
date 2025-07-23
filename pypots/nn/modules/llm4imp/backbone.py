"""
The backbone of LLM4IMP: a prompted, reprogrammed large language model
for time-series imputation using frozen LLM (GPT2), FFT patching,
and instructional token prompts.
"""

# Created by Franck Junior Aboya <mesabo18@gmail.com / messouaboya17@gmail.com>
# License: BSD-3-Clause

import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Tokenizer

from ..patchtst.layers import PatchEmbedding, FlattenHead
from .layers import ReprogrammingLayer
from ..revin import RevIN


class BackboneLLM4IMP(nn.Module):
    def __init__(
        self,
        n_steps,
        n_features,
        patch_size,
        patch_stride,
        d_model,
        d_ffn,
        d_llm,
        n_heads,
        dropout,
        prompt_template: str = "Impute missing values at time steps where mask=0",
    ):
        super().__init__()

        self.n_steps = n_steps
        self.n_features = n_features
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.d_llm = d_llm
        self.n_heads = n_heads
        self.prompt_template = prompt_template

        self.tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token  # To fix padding issue
        self.llm = GPT2Model.from_pretrained(
            "openai-community/gpt2",
            output_hidden_states=True,
        )

        # Freeze LLM weights
        for param in self.llm.parameters():
            param.requires_grad = False

        self.patch_embedding = PatchEmbedding(
            d_model=d_model,
            patch_size=patch_size,
            patch_stride=patch_stride,
            padding=patch_stride,
            dropout=dropout,
            positional_embedding=True,
        )

        self.reprogramming = ReprogrammingLayer(
            d_model=d_model,
            n_heads=n_heads,
            d_k=d_ffn,
            d_llm=d_llm,
        )

        self.prompt_proj = nn.Linear(self.llm.config.n_embd, d_model)
        self.revin = RevIN(n_features=n_features, affine=False)

        # lazy initialization of output head
        self.output_head = None
        self.output_head_dropout = dropout

    def build_prompts(self, X, missing_mask):
        assert X.dim() == 3, f"Expected [B, T, D], got {X.shape}"
        B, T, D = X.shape
        prompts = []
        for b in range(B):
            for d in range(D):
                missing_indices = (missing_mask[b, :, d] == 0).nonzero(as_tuple=True)[0]
                idx_str = ",".join(map(str, missing_indices.tolist()))
                prompt = f"{self.prompt_template}. Missing indices: {idx_str}."
                prompts.append(prompt)
        tokenized = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        embeds = self.llm.get_input_embeddings()(tokenized.input_ids.to(X.device)).view(B, D, -1, self.llm.config.n_embd)
        return embeds

    def forward(self, X, missing_mask):
        X = self.revin(X, mode="norm")  # Normalize
        B, T, D = X.shape

        # Build prompt embeddings
        llm_embeds = self.build_prompts(X, missing_mask)         # [B, D, prompt_len, 768]
        prompts_proj = self.prompt_proj(llm_embeds)              # [B, D, prompt_len, d_model]

        X = X.permute(0, 2, 1).contiguous()     # [B, D, T]
        X = X.view(B * D, 1, T)                 # [B*D, 1, T]
        X = self.patch_embedding(X)             # [B*D, P, d_model]
        P = X.shape[1]                          # actual patch count
        X = X.view(B, D, P, self.d_model)       # [B, D, P, d_model]

        reprogrammed = self.reprogramming(
            X.flatten(0, 1),              # [B*D, P, d_model]
            llm_embeds.flatten(0, 1),     # [B*D, prompt_len, 768] as keys
            llm_embeds.flatten(0, 1)      # [B*D, prompt_len, 768] as values
        )

        input_repr = reprogrammed.view(B, D, self.d_llm, P).permute(0, 1, 3, 2).contiguous()  # [B, D, P, d_llm]

        # Initialize output head lazily with true patch count
        if self.output_head is None:
            self.output_head = FlattenHead(
                d_input=self.d_llm * P,
                d_output=self.n_steps,
                n_features=self.n_features,
                head_dropout=self.output_head_dropout,
            )

        output = self.output_head(input_repr)   # [B, D, T]
        output = output.permute(0, 2, 1).contiguous()  # [B, T, D]
        output = self.revin(output, mode="denorm")     # Denormalize
        return output
