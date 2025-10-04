"""
Reference code for LLaMA-3.1 training and inference.
Will save the model weights into files, to be read from C as initialization.

This code differs from GPT-2 very slightly, there are three main differences:
1) RoPE: LLaMA uses a different positional encoding scheme called Relative Positional Encoding (RoPE).
2) GQA: Grouped Query Attention (GQA) is used to reduce the number of attention heads.
3) SwiGLU: Swish-Gated Linear Unit (SwiGLU) is used as the activation function in the MLP.

References:
# 1) https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/api/tokenizer.py
# 2) https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/api/model.py
# 3) https://github.com/meta-llama/llama3/blob/11817d47e1ba7a4959b025eb1ca308572e0e3963/llama/generation.py

Example launches to only benchmark the speed of bfloat16 compiled GPU training:
python train_llama3.py # Train LLaMA3-tiny (125M) on TinyShakespeare
"""

import os
import sys
import math
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any, Union

# ---------------------------------------------------------------------------
# Configuration options and hyperparameters

@dataclass
class LlamaConfig:
    # Model architecture parameters
    dim: int = 768  # Embedding dimension
    n_layers: int = 12  # Number of transformer blocks
    n_heads: int = 12  # Number of attention heads
    n_kv_heads: Optional[int] = None  # Number of key/value heads (None = same as n_heads)
    vocab_size: int = 50257  # Size of the vocabulary
    hidden_dim: Optional[int] = None  # Size of the MLP hidden layer (None = 4*dim)
    multiple_of: int = 256  # Make hidden_dim a multiple of this value
    norm_eps: float = 1e-5  # LayerNorm epsilon
    max_seq_len: int = 2048  # Maximum sequence length
    dropout: float = 0.0  # Dropout rate
    
    # RoPE parameters
    rope_base: float = 10000.0  # Base for rotary position embeddings
    rope_scaling: Optional[Dict[str, Any]] = None  # Scaling factors for position embeddings
    
    # Training parameters
    seed: int = 42
    learning_rate: float = 6e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    
    def __post_init__(self):
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        if self.hidden_dim is None:
            self.hidden_dim = 4 * self.dim
        
        # Make hidden_dim a multiple of multiple_of
        self.hidden_dim = ((self.hidden_dim + self.multiple_of - 1) 
                          // self.multiple_of) * self.multiple_of
    
    @classmethod
    def from_name(cls, name: str) -> "LlamaConfig":
        """Create a configuration based on a model name."""
        if name == "llama3-tiny":  # 125M
            return cls(dim=768, n_layers=12, n_heads=12, n_kv_heads=12)
        elif name == "llama3-small":  # 1.3B
            return cls(dim=1536, n_layers=24, n_heads=12, n_kv_heads=12)
        elif name == "llama3-medium":  # 2.7B
            return cls(dim=2048, n_layers=32, n_heads=16, n_kv_heads=16)
        elif name == "llama3-base":  # 8B
            return cls(dim=4096, n_layers=32, n_heads=32, n_kv_heads=8)
        elif name == "llama3-large":  # 70B
            return cls(dim=8192, n_layers=80, n_heads=64, n_kv_heads=8)
        else:
            raise ValueError(f"Unknown model name: {name}")

# ---------------------------------------------------------------------------
# RoPE implementation

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.
    
    Args:
        xq: Query tensor of shape (batch_size, seq_len, n_heads, head_dim)
        xk: Key tensor of shape (batch_size, seq_len, n_kv_heads, head_dim)
        freqs_cis: Complex tensor of shape (seq_len, head_dim/2)
        
    Returns:
        Tuple of transformed query and key tensors.
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # Reshape to match the complex dimension
    freqs_cis = freqs_cis[:xq.shape[1]]
    
    # Apply rotary embeddings
    xq_out = torch.view_as_real(xq_ * freqs_cis.unsqueeze(0).unsqueeze(2)).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis.unsqueeze(0).unsqueeze(2)).flatten(3)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)

# ---------------------------------------------------------------------------
# Model implementation

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        x = x.float()  # Ensure computation in float32
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(x_dtype)

class Attention(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.dim = config.dim
        self.head_dim = config.dim // config.n_heads
        
        # For GQA: n_kv_heads can be less than n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        
        # Projection matrices
        self.wq = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=False)
        
        # Set output projection scale flag for residual scaling
        self.wo.LLMC_RESIDUAL_SCALE_FLAG = 1
        
        # Initialize flash attention if available
        self.flash = hasattr(F, "scaled_dot_product_attention")
    
    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Project to queries, keys, values
        xq = self.wq(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        xk = self.wk(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        
        # Apply rotary embeddings
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)
        
        # Repeat keys and values for GQA if necessary
        if self.n_rep > 1:
            xk = repeat_kv(xk, self.n_rep)
            xv = repeat_kv(xv, self.n_rep)
            
        # Reshape for attention computation
        xq = xq.transpose(1, 2)  # (batch_size, n_heads, seq_len, head_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        
        # Compute attention
        if self.flash:
            # Use flash attention if available
            output = F.scaled_dot_product_attention(xq, xk, xv, is_causal=True)
        else:
            # Manual implementation of attention
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1)
            scores = scores.masked_fill(causal_mask, -float('inf'))
            attention = F.softmax(scores, dim=-1)
            output = torch.matmul(attention, xv)
        
        # Reshape and project output
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.wo(output)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat keys/values for GQA."""
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    return (
        x[:, :, :, None, :]
        .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
        .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
    )

class FeedForward(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        dim = config.dim
        hidden_dim = config.hidden_dim
        
        # SwiGLU activation
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        
        # Set output projection scale flag for residual scaling
        self.w2.LLMC_RESIDUAL_SCALE_FLAG = 1
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU activation function
        swish = F.silu(self.w1(x))
        x_V = self.w3(x)
        x = swish * x_V
        return self.w2(x)

class LlamaBlock(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
    
    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        # Pre-normalization for attention
        h = x + self.attention(self.attention_norm(x), freqs_cis)
        # Pre-normalization for feed-forward
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class LlamaModel(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList([LlamaBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        
        # Pre-compute RoPE frequencies
        self.freqs_cis = precompute_freqs_cis(
            config.dim // config.n_heads,
            config.max_seq_len * 2,
            config.rope_base,
        )
    
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = tokens.shape
        h = self.tok_embeddings(tokens)
        
        # Retrieve the RoPE frequencies for the current sequence length
        freqs_cis = self.freqs_cis[:seq_len].to(h.device)
        
        # Apply transformer layers
        for layer in self.layers:
            h = layer(h, freqs_cis)
        
        h = self.norm(h)
        return h

class LlamaForCausalLM(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        
        # Tie weights between token embeddings and LM head
        self.lm_head.weight = self.model.tok_embeddings.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Special scaling for output projection layers
        for name, module in self.named_modules():
            if hasattr(module, "LLMC_RESIDUAL_SCALE_FLAG"):
                module.weight.data *= (1.0 / math.sqrt(2 * config.n_layers))
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            std = 0.02
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
    
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        h = self.model(tokens)
        logits = self.lm_head(h)
        return logits

# ---------------------------------------------------------------------------
# Data loading and tokenization functions similar to train_gpt2.py
# (These would need to be implemented based on the existing dataloader in llm.c)

# ---------------------------------------------------------------------------
# Training functions

def train_llama3(config_name: str = "llama3-tiny"):
    """
    Main training function for LLaMA3 models.
    """
    # Create configuration
    config = LlamaConfig.from_name(config_name)
    
    # Initialize model
    model = LlamaForCausalLM(config)
    
    # Training code would follow here - similar to train_gpt2.py
    # This would include data loading, optimizer setup, training loop, etc.
    
    # Save model weights in a format compatible with llm.c
    # This would involve exporting the weights similarly to how it's done in train_gpt2.py
    
    # Save the model in the appropriate format for C
    export_model_for_c(model, config)

def export_model_for_c(model, config):
    """
    Export model weights to binary files for C consumption.
    Similar to how it's done in train_gpt2.py
    """
    print("Exporting model weights for C...")
    # Implementation would extract weights and save them in the format
    # expected by the C code in llm.c
    
    # This would follow a similar pattern to what's in train_gpt2.py

if __name__ == "__main__":
    train_llama3()
            for name in ['feed_forward.w1', 'feed_forward.w2', 'feed_forward.w3']:
                old_key = f'layers.{i}.{name}.weight'
                new_key = f'transformer.h.{i}.mlp.{ffn_map[name.split(".")[-1]]}.weight'
                checkpoint[new_key] = checkpoint.pop(old_key)

        checkpoint['transformer.ln_f.weight'] = checkpoint.pop('norm.weight')
        checkpoint['lm_head.weight'] = checkpoint.pop('output.weight')

        return checkpoint

    @staticmethod
    def adapt_llama_state_dict_keys_hf(checkpoint, config: LlamaConfig):
        # Modify key names from HuggingFace's LLaMA to our LLaMA
        # our key names are derived from GPT-2's key names
        checkpoint['transformer.wte.weight'] = checkpoint.pop('model.embed_tokens.weight')

        # We need to unpermute K and V because HF script permuted the original Meta-LLaMA weights
        # see: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py
        def unpermute(w, n_heads, dim1, dim2):
            return w.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)

        for i in range(config.n_layer):
            for name in ['input_layernorm', 'post_attention_layernorm']:
                old_key = f'model.layers.{i}.{name}.weight'  # e.g. layers.x.attention_norm.weight -> transformer.h.x.ln_1.weight
                new_key = f'transformer.h.{i}.ln_{1 if name == "input_layernorm" else 2}.weight'
                checkpoint[new_key] = checkpoint.pop(old_key)

        for i in range(config.n_layer):
            for name in ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj']:
                old_key = f'model.layers.{i}.{name}.weight'
                new_key = f'transformer.h.{i}.attn.c_attn.weight'
                if name == 'self_attn.q_proj':
                    checkpoint[new_key] = unpermute(checkpoint.pop(old_key), config.n_head, config.n_embd, config.n_embd)
                else:  # merge 3 weights into transformer.h.x.attn.c_attn.weight
                    tensor = checkpoint.pop(old_key)
                    if name == 'self_attn.k_proj':
                        tensor = unpermute(tensor, config.n_kv_head, config.n_kv_head * (config.n_embd // config.n_head), config.n_embd)
                    checkpoint[new_key] = torch.cat((checkpoint[new_key], tensor), dim=0)
            old_key = f'model.layers.{i}.self_attn.o_proj.weight'
            new_key = f'transformer.h.{i}.attn.c_proj.weight'
            checkpoint[new_key] = checkpoint.pop(old_key)

        ffn_map = {'gate_proj': 'c_fc2', 'down_proj': 'c_proj', 'up_proj': 'c_fc'}
        for i in range(config.n_layer):
            for name in ['gate_proj', 'down_proj', 'up_proj']:
                old_key = f'model.layers.{i}.mlp.{name}.weight'
                new_key = f'transformer.h.{i}.mlp.{ffn_map[name]}.weight'
                checkpoint[new_key] = checkpoint.pop(old_key)

        checkpoint['transformer.ln_f.weight'] = checkpoint.pop('model.norm.weight')

        return checkpoint

    @classmethod
    def from_pretrained_llama3_hf(cls, model_id):
        """Loads pretrained LLaMA model weights from HuggingFace"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        assert model_id == "meta-llama/Meta-Llama-3.1-8B", "Only the 8B-base model is supported for now"
        model_args = LlamaConfig()

        model = AutoModelForCausalLM.from_pretrained(model_id)
        checkpoint = LLaMA.adapt_llama_state_dict_keys_hf(model.state_dict(), model_args)

        original_default_type = torch.get_default_dtype()  # save the default type
        torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)  # much faster loading
        model = LLaMA(model_args)
        model.load_state_dict(checkpoint, strict=False)
        torch.set_default_tensor_type(torch.tensor([], dtype=original_default_type, device="cpu").type())  # restore default type

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_id = 128004  # this is the pad token id for LLaMA 3.1 base, we need to set this explicitly as our generate func expects it
        tokenizer.stop_tokens = [tokenizer.eos_token_id]
        model.tokenizer = tokenizer
        return model

    @classmethod
    def from_pretrained_llama3_meta(cls, ckpt_dir, tokenizer_path):
        """Loads pretrained LLaMA model weights from a checkpoint directory"""
        model_args = LlamaConfig()

        ckpt_path = sorted(Path(ckpt_dir).glob("*.pth"))[0]
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        checkpoint = LLaMA.adapt_llama_state_dict_keys(checkpoint, model_args)

        original_default_type = torch.get_default_dtype()  # save the default type
        torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)  # much faster loading
        model = LLaMA(model_args)
        model.load_state_dict(checkpoint, strict=False)
        torch.set_default_tensor_type(torch.tensor([], dtype=original_default_type, device="cpu").type())  # restore default type

        tokenizer = Tokenizer(model_path=tokenizer_path)
        model.tokenizer = tokenizer
        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type, zero_stage):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print0(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print0(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        print0(f"using fused AdamW: {use_fused}")
        if zero_stage == 1:
            print0("using ZeroRedundancyOptimizer")
            optimizer = ZeroRedundancyOptimizer(**optim_groups[0], optimizer_class=torch.optim.AdamW,
                                                lr=learning_rate, betas=betas, fused=use_fused)
            optimizer.add_param_group(optim_groups[1])
        else:
            print0("using regular AdamW")
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, fused=use_fused)
        return optimizer

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: List[List[int]],
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
        echo: bool = False,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        """
        Generate text sequences based on provided prompts using the language generation model.

        Args:
            prompt_tokens (List[List[int]]): List of tokenized prompts, where each prompt is represented as a list of integers.
            max_gen_len (int): Maximum length of the generated text sequence.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.

        Returns:
            Tuple[List[List[int]], Optional[List[List[float]]]]: A tuple containing generated token sequences.

        Note:
            This method uses the provided prompts as a basis for generating text. It employs nucleus sampling to produce text with controlled randomness.

        """
        bsz = len(prompt_tokens)
        assert bsz <= self.config.max_gen_batch_size, f"Batch size {bsz} exceeds the maximum generation batch size {self.config.max_gen_batch_size}"
        device = next(self.parameters()).device

        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        assert max_prompt_len <= self.config.block_size, f"Prompt length {max_prompt_len} exceeds the maximum block size {self.config.block_size}"
        total_len = min(self.config.block_size, max_gen_len + max_prompt_len)

        pad_id = self.tokenizer.pad_id
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device=device)
        for idx, t in enumerate(prompt_tokens):
            tokens[idx, : len(t)] = torch.tensor(t, dtype=torch.long, device=device)

        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device=device)
        input_text_mask = tokens != pad_id

        if min_prompt_len == total_len:
            logits, _ = self.forward(tokens, start_pos=prev_pos)

        stop_tokens = torch.tensor(list(self.tokenizer.stop_tokens)).to(device)

        for cur_pos in range(min_prompt_len, total_len):
            logits, _ = self.forward(tokens[:, prev_pos:cur_pos], start_pos=prev_pos)
            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            tokens[:, cur_pos] = next_token
            eos_reached |= ~input_text_mask[:, cur_pos] & torch.isin(next_token, stop_tokens)
            prev_pos = cur_pos
            if all(eos_reached):
                break

        out_tokens = []
        for i, toks in enumerate(tokens.tolist()):
            # cut to max gen len
            start = 0 if echo else len(prompt_tokens[i])
            toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
            # cut to after eos tok if any
            for stop_token in self.tokenizer.stop_tokens:
                try:
                    eos_idx = toks.index(stop_token)
                    toks = toks[:eos_idx]
                except ValueError:
                    pass
            out_tokens.append(toks)
        return out_tokens

# -----------------------------------------------------------------------------
# sampling utils

def sample_top_p(probs, p):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.
    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

# -----------------------------------------------------------------------------
# Llama 3.1 Tokenizer

# The tiktoken tokenizer can handle <=400k chars without
# pyo3_runtime.PanicException.
TIKTOKEN_MAX_ENCODE_CHARS = 400_000

# https://github.com/openai/tiktoken/issues/195
# Here we iterate over subsequences and split if we exceed the limit
# of max consecutive non-whitespace or whitespace characters.
MAX_NO_WHITESPACES_CHARS = 25_000


class Tokenizer:
    """
    Tokenizing and encoding/decoding text using the Tiktoken tokenizer.
    """

    special_tokens: Dict[str, int]

    num_reserved_special_tokens = 256

    pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"  # noqa: E501

    def __init__(self, model_path: str):
        """
        Initializes the Tokenizer with a Tiktoken model.

        Args:
            model_path (str): The path to the Tiktoken model file.
        """
        assert os.path.isfile(model_path), model_path

        mergeable_ranks = load_tiktoken_bpe(model_path)
        num_base_tokens = len(mergeable_ranks)
        special_tokens = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|finetune_right_pad_id|>",
            "<|step_id|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|eom_id|>",  # end of message
            "<|eot_id|>",  # end of turn
            "<|python_tag|>",
        ]
        reserved_tokens = [
            f"<|reserved_special_token_{2 + i}|>"
            for i in range(self.num_reserved_special_tokens - len(special_tokens))
        ]
        special_tokens = special_tokens + reserved_tokens

        self.special_tokens = {
            token: num_base_tokens + i for i, token in enumerate(special_tokens)
        }
        self.model = tiktoken.Encoding(
            name=Path(model_path).name,
            pat_str=self.pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens,
        )

        self.n_words: int = num_base_tokens + len(special_tokens)
        # BOS / EOS token IDs
        self.bos_id: int = self.special_tokens["<|begin_of_text|>"]
        self.eos_id: int = self.special_tokens["<|end_of_text|>"]
        self.eot_id: int = self.special_tokens["<|eot_id|>"]
        self.eom_id: int = self.special_tokens["<|eom_id|>"]
        self.python_tag_id = self.special_tokens["<|python_tag|>"]
        self.pad_id: int = self.special_tokens["<|finetune_right_pad_id|>"]
        # hardcoded stop tokens for the base model
        self.stop_tokens = [
            self.special_tokens["<|begin_of_text|>"],
            self.special_tokens["<|end_of_text|>"],
        ]

    def encode(
        self,
        s: str,
        *,
        bos: bool,
        eos: bool,
        allowed_special: Optional[Union[Literal["all"], AbstractSet[str]]] = None,
        disallowed_special: Union[Literal["all"], Collection[str]] = (),
    ) -> List[int]:
        """
        Encodes a string into a list of token IDs.

        Args:
            s (str): The input string to be encoded.
            bos (bool): Whether to prepend the beginning-of-sequence token.
            eos (bool): Whether to append the end-of-sequence token.
            allowed_tokens ("all"|set[str]): allowed special tokens in string
            disallowed_tokens ("all"|set[str]): special tokens that raise an error when in string

        Returns:
            list[int]: A list of token IDs.

        By default, setting disallowed_special=() encodes a string by ignoring
        special tokens. Specifically:
        - Setting `disallowed_special` to () will cause all text corresponding
          to special tokens to be encoded as natural text (insteading of raising
          an error).
        - Setting `allowed_special` to "all" will treat all text corresponding
          to special tokens to be encoded as special tokens.
        """
        if allowed_special is None:
            allowed_special = set()
        assert type(s) is str

        substrs = (
            substr
            for i in range(0, len(s), TIKTOKEN_MAX_ENCODE_CHARS)
            for substr in self._split_whitespaces_or_nonwhitespaces(
                s[i : i + TIKTOKEN_MAX_ENCODE_CHARS], MAX_NO_WHITESPACES_CHARS
            )
        )
        t: List[int] = []
        for substr in substrs:
            t.extend(
                self.model.encode(
                    substr,
                    allowed_special=allowed_special,
                    disallowed_special=disallowed_special,
                )
            )
        if bos:
            t.insert(0, self.bos_id)
        if eos:
            t.append(self.eos_id)
        return t

    def decode(self, t: Sequence[int]) -> str:
        # Typecast is safe here. Tiktoken doesn't do anything list-related with the sequence.
        return self.model.decode(cast(List[int], t))

    @staticmethod
    def _split_whitespaces_or_nonwhitespaces(
        s: str, max_consecutive_slice_len: int
    ) -> Iterator[str]:
        """
        Splits the string `s` so that each substring contains no more than `max_consecutive_slice_len`
        consecutive whitespaces or consecutive non-whitespaces.
        """
        current_slice_len = 0
        current_slice_is_space = s[0].isspace() if len(s) > 0 else False
        slice_start = 0

        for i in range(len(s)):
            is_now_space = s[i].isspace()

            if current_slice_is_space ^ is_now_space:
                current_slice_len = 1
                current_slice_is_space = is_now_space
            else:
                current_slice_len += 1
                if current_slice_len > max_consecutive_slice_len:
                    yield s[slice_start:i]
                    slice_start = i
                    current_slice_len = 1
        yield s[slice_start:]

# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

def _peek_data_shard(filename):
    # only reads the header, returns header data
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
    if header[0] != 20240801:
        print("ERROR: magic number mismatch in the data .bin file!")
        exit(1)
    assert header[1] == 7, "unsupported version"
    ntok = header[2] # number of tokens (claimed)
    return ntok # for now just return the number of tokens

def _load_data_shard(filename):
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
        assert header[0] == 20240801, "magic number mismatch in the data .bin file"
        assert header[1] == 7, "unsupported version"
        ntok = header[2] # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        tokens = np.frombuffer(f.read(), dtype=np.uint32)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens

class DistributedShardedDataLoader:
    """
    This DataLoader is both:
    - distributed (works correctly in case of multiple processes in DDP)
    - sharded (supports datasets that are broken up into multiple data shards)
    It is not *permuted*, meaning that it itearates over the data in the order
    of the dataset on disk, so the user should make sure to shuffle their examples
    during the creation of their data shards for best performance.
    """
    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

        # load and validate all data shards, count number of tokens in total
        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * B * T + 1
            ntok_total += shard_ntok
        self.ntok_total = ntok_total
        print0(f"DataLoader: total number of tokens: {ntok_total:,} across {len(self.files)} files")

        # kick things off
        self.current_shard = None
        self.reset()

    def reset(self):
        # we're being a bit clever here: if we already had shard 0 loaded,
        # then don't do the work to reload it, just reset the pointer
        if self.current_shard != 0:
            self.current_shard = 0
            self.tokens = _load_data_shard(self.files[self.current_shard])
        self.current_position = self.process_rank * self.B * self.T

    def advance(self): # advance to next data shard
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        B = self.B
        T = self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        buf = torch.tensor(buf, dtype=torch.long)
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the start pointer in current shard
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds advance the shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()
        return x, y

# -----------------------------------------------------------------------------
# Python -> C bridge utilities for saving params/grads/activations to .bin files

def write_fp32(tensor, file):
    t = tensor.detach().cpu().to(torch.float32)
    b = t.numpy().tobytes()
    file.write(b)

def write_bf16(tensor, file):
    t = tensor.detach().cpu().to(torch.bfloat16)
    # numpy doesn't have bf16 datatype so we have to trick it
    t = t.view(torch.int16) # trick: reinterpret as int16
    b = t.numpy().tobytes()
    file.write(b)

def write_tensors(model_tensors, L, file, dtype):
    # writes LLaMA 3 model's weights to a binary file
    assert dtype in {"float32", "bfloat16"}
    write_fun = write_fp32 if dtype == "float32" else write_bf16
    write_fun(model_tensors["transformer.wte.weight"], file) # (V, C)
    for i in range(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.ln_1.weight"], file)
    for i in range(L): # (L, 3C, C)
        write_fun(model_tensors[f"transformer.h.{i}.attn.c_attn.weight"], file)
    for i in range(L): # (L, C, C)
        write_fun(model_tensors[f"transformer.h.{i}.attn.c_proj.weight"], file)
    for i in range(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.ln_2.weight"], file)
    for i in range(L): # (L, 4C, C)
        write_fun(model_tensors[f"transformer.h.{i}.mlp.c_fc.weight"], file)
    for i in range(L): # (L, 4C, C)
        write_fun(model_tensors[f"transformer.h.{i}.mlp.c_fc2.weight"], file)
    for i in range(L): # (L, C, 4C)
        write_fun(model_tensors[f"transformer.h.{i}.mlp.c_proj.weight"], file)
    write_fun(model_tensors["transformer.ln_f.weight"], file) # (C, )
    write_fun(model_tensors["lm_head.weight"], file) # (V, C)

def write_model(model, filename, dtype):
    # everything we need to instantiate the model
    # 1) header is: version int, LLaMAConfig ints, padding to 1024 bytes
    assert dtype in {"float32", "bfloat16"}
    version = {
        "float32": 3, # 3: all tensors are fp32
        "bfloat16": 5, # 5: all tensors are bf16
    }[dtype]
    header = torch.zeros(256, dtype=torch.int32)
    header[0] = 20240803 # magic
    header[1] = version # checkpoint version
    header[2] = model.config.block_size
    header[3] = model.config.vocab_size
    header[4] = model.config.n_layer
    header[5] = model.config.n_head
    header[6] = model.config.n_kv_head
    header[7] = model.config.n_embd
    header[8] = model.config.ffn_dim_multiplier
    header[9] = model.config.multiple_of
    header[10] = model.config.norm_eps
    header[11] = model.config.rope_theta
    header[12] = model.config.use_scaled_rope
    header[13] = model.config.max_gen_batch_size
    header[14] = int(model.config.version.split('.')[0]) # major version
    header[15] = int(model.config.version.split('.')[1]) # minor version
    # 2) the parameters follow the header
    params = {name: param.cpu() for name, param in model.named_parameters()}
    # now write to file
    with open(filename, "wb") as file:
        file.write(header.numpy().tobytes()) # header
        write_tensors(params, model.config.n_layer, file, dtype) # params
    print(f"wrote {filename}")

def write_state(model, x, y, logits, loss, filename):
    # the state is used for debugging.
    # it contains information about the input, logits, loss, and the parameter gradients
    # this can be used for checking the computation correctness in C
    header = torch.zeros(256, dtype=torch.int32)
    header[0] = 20240803 # magic
    header[1] = x.size(0) # batch size of the batch, B
    header[2] = x.size(1) # temporal extent of the batch, T
    grads = {name: param.grad.cpu() for name, param in model.named_parameters()}
    with open(filename, "wb") as file:
        # header
        file.write(header.numpy().tobytes())
        # input x
        file.write(x.cpu().numpy().astype("int32").tobytes()) # (B, T)
        # targets y
        file.write(y.cpu().numpy().astype("int32").tobytes()) # (B, T)
        # logits (result of the model forward pass)
        write_fp32(logits.cpu(), file)
        # loss (single float, result of the cross entropy loss)
        write_fp32(loss.cpu(), file)
        # gradients
        write_tensors(grads, model.config.n_layer, file, "float32")
    print(f"wrote {filename}")

# -----------------------------------------------------------------------------
# int main

def print0(*args, **kwargs):
    # modified print that only prints from the master process
    # if this is not a distributed run, it's just a print
    if int(os.environ.get("RANK", 0)) == 0:
        print(*args, **kwargs)

if __name__ == "__main__":
    print0(f"Running pytorch {torch.version.__version__}")

    # default settings will overfit a tiny batch of data
    # and save model weights and debug state to disk on the first iteration
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_hf", type=int, default=1, help="use HuggingFace (default) or use Meta's model")
    parser.add_argument("--ckpt_dir", type=str, default=None, help="path to llama3 model checkpoint (needed if use_hf=0)")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="path to llama3 tokenizer (needed if use_hf=0)")
    # file system input / output
    parser.add_argument("--input_bin", type=str, default="dev/data/tinyshakespeare/tiny_shakespeare_val.bin", help="input .bin to train on")
    parser.add_argument("--input_val_bin", type=str, default="", help="input .bin to eval validation loss on")
    parser.add_argument("--output_dir", type=str, default="", help="output directory to which to write logs and checkpoints")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-8B", help="chose the llama model")
    # token layout for each step of the optimization
    parser.add_argument("--batch_size", type=int, default=4, help="batch size, in units of #batch dimensions")
    parser.add_argument("--sequence_length", type=int, default=64, help="sequence length")
    parser.add_argument("--total_batch_size", type=int, default=256, help="total desired batch size, in units of #tokens")
    # workload (number of steps)
    parser.add_argument("--num_iterations", type=int, default=10, help="number of iterations to run")
    parser.add_argument("--inference_only", type=int, default=0, help="only run inference")
    # optimization
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="learning rate warmup iterations")
    parser.add_argument("--warmup_iters", type=int, default=0, help="learning rate warmup iterations")
    parser.add_argument("--learning_rate_decay_frac", type=float, default=1.0, help="learning rate warmup iterations")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="maximum gradient magnitude")
    # evaluation
    parser.add_argument("--val_loss_every", type=int, default=0, help="every how mant steps to evaluate val loss?")
    parser.add_argument("--val_max_steps", type=int, default=20, help="how many batches of val to average?")
    parser.add_argument("--sample_every", type=int, default=0, help="how often to sample from the model?")
    # debugging
    parser.add_argument("--overfit_single_batch", type=int, default=1, help="overfit just one batch of data")
    # numerics
    parser.add_argument("--tensorcores", type=int, default=0, help="use tensorcores")
    # memory management
    parser.add_argument("--device", type=str, default="", help="by default we autodetect, or set it here")
    parser.add_argument("--compile", type=int, default=0, help="torch.compile the model")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="float32|float16|bfloat16")
    parser.add_argument("--zero_stage", type=int, default=0, help="zero redundancy optimizer stage (0/1/2/3)")
    # python -> C bridge
    parser.add_argument("--write_tensors", type=int, default=0, help="write tensors to disk")
    args = parser.parse_args()

    # args error checking and convenience variables
    B, T = args.batch_size, args.sequence_length
    assert 1 <= T <= 8192, "sequence length must be between 1 and 8192"
    assert args.dtype in {"float32", "float16", "bfloat16"}
    assert args.model in {"meta-llama/Meta-Llama-3.1-8B"}  # only 8B base model supported for now

    # create the logging directory if it does not exist
    logfile = None
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        logfile = os.path.join(args.output_dir, "main.log")
        # create the log file "main.log" inside it, and wipe it clean
        with open(logfile, "w") as f:
            pass

    # set up DDP (distributed data parallel). torchrun sets this env variable
    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    if ddp:
        # use of DDP atm demands CUDA, we set the device appropriately according to rank
        assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
        seed_offset = 0 # each process gets the exact same seed
        zero_stage = args.zero_stage
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        zero_stage = 0
        ddp_world_size = 1
        master_process = True
        seed_offset = 0
        # select the device
        if args.device:
            # provided explicitly by the user
            device = args.device
        else:
            # attempt to autodetect the device
            device = "cpu"
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    assert device_type in {'cuda'}, "GPU required to run LLaMA 3"  # we need to load LLaMA as bf16 on CUDA
    print(f"using device: {device}")

    # calculate gradient accumulation from the desired total batch size and the current run configuration
    tokens_per_fwdbwd = B * T * ddp_world_size
    assert args.total_batch_size % tokens_per_fwdbwd == 0
    grad_accum_steps = args.total_batch_size // tokens_per_fwdbwd
    print0(f"total desired batch size: {args.total_batch_size}")
    print0(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    # set up a context manager following the desired dtype and device
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if (device_type == "cuda") else nullcontext()

    # rng / reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # set the torch precision mode to use TensorFloat32 (TF32) for matmuls
    # docs https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
    if args.tensorcores:
        torch.set_float32_matmul_precision('high')

    # init the model
    if args.use_hf:
        model = LLaMA.from_pretrained_llama3_hf(args.model)
    else:  # use Meta's checkpoint
        assert args.ckpt_dir is not None and os.path.exists(args.ckpt_dir), f"llama3 ckpt dir {args.ckpt_dir} does not exist"
        assert args.tokenizer_path is not None and os.path.exists(args.tokenizer_path), f"llama3 tokenizer path {args.tokenizer_path} does not exist"
        model = LLaMA.from_pretrained_llama3_meta(args.ckpt_dir, args.tokenizer_path)

    model.train()
    if args.compile:
        if hasattr(config, "coordinate_descent_tuning"):
            config.coordinate_descent_tuning = True # suggested by @Chillee
        print0("compiling the model...")
        model = torch.compile(model)

    # -------------------------------------------------------------------------
    # Our own version of a simple DistributedDataLoader

    # load tokens
    train_loader = DistributedShardedDataLoader(args.input_bin, B, T, ddp_rank, ddp_world_size)
    val_loader = None
    if args.input_val_bin:
        val_loader = DistributedShardedDataLoader(args.input_val_bin, B, T, ddp_rank, ddp_world_size)

    # -------------------------------------------------------------------------
    # PyTorch -> C bridge: save some weights and state for C to load later as reference

    # do one forward pass to generate ground truth for our C tests
    if master_process and args.write_tensors and (not args.inference_only):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        logits, loss = model(x, y)
        loss.backward()
        # save model params, in bfloat16
        model_to_size = {"meta-llama/Meta-Llama-3.1-8B": "8B"}
        model_size_str = model_to_size[args.model] # e.g. "8B"
        write_model(model, os.path.join(args.output_dir, f"llama3.1_{model_size_str}_bf16.bin"), dtype="bfloat16")
        # save x, y, logits, loss, and parameter gradients, for debugging C
        # always store these in fp32 to have an accurate reference (?)
        write_state(model, x, y, logits, loss, os.path.join(args.output_dir, f"llama3_{model_size_str}_debug_state.bin"))
        # reset the train_loader for the optimization below
        train_loader.reset()

    # -------------------------------------------------------------------------
    # main training loop

    # here we wrap model into DDP container
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

    # init the optimizer
    optimizer = raw_model.configure_optimizers(weight_decay=args.weight_decay,
                                               learning_rate=args.learning_rate, betas=(0.9, 0.95),
                                               device_type=device, zero_stage=zero_stage)

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(it):
        min_lr = args.learning_rate * args.learning_rate_decay_frac
        # 1) linear warmup for warmup_iters steps
        if it < args.warmup_iters:
            return args.learning_rate * (it+1) / args.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > args.num_iterations:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - args.warmup_iters) / (args.num_iterations - args.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
        return min_lr + coeff * (args.learning_rate - min_lr)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    timings = []
    norm = -1.0   # dummy value to print in inference-only mode
    for step in range(args.num_iterations + 1):
        t0 = time.time()
        last_step = (step == args.num_iterations)

        # once in a while evaluate the validation dataset
        if (args.val_loss_every > 0 \
            and (step % args.val_loss_every == 0 or last_step)) \
            and (val_loader is not None):
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss = 0.0
                for _ in range(args.val_max_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    _, loss = model(x, y, return_logits=False)
                    val_loss += loss.item()
                val_loss /= args.val_max_steps
            # log to console and to file
            print0(f"val loss {val_loss}")
            if master_process and logfile is not None:
                with open(logfile, "a") as f:
                    f.write("s:%d tel:%f\n" % (step, val_loss))

        # once in a while perform model inference on the master process
        if (args.sample_every > 0 \
            and (step % args.sample_every == 0 or last_step)) \
            and master_process:
            model.eval()
            prompts: List[str] = [
        "Clearly, the meaning of life is",
        "Simply put, the theory of relativity states that",
        """The repo llm.c on GitHub is""",
        """Translate English to French:

        sea otter => loutre de mer
        peppermint => menthe poivrée
        plush girafe => girafe peluche
        cheese =>""",
            ]
            if args.use_hf:
                prompt_tokens = [model.tokenizer(x).input_ids for x in prompts]
            else:  # Meta
                prompt_tokens = [model.tokenizer.encode(x, bos=True, eos=False) for x in prompts]

            generation_tokens = model.generate(prompt_tokens, max_gen_len=64, temperature=0.6, top_p=0.9, echo=False)
            results = [{"generation": model.tokenizer.decode(t)} for t in generation_tokens]
            for prompt, result in zip(prompts, results):
                print(prompt, end="")
                print(f"{result['generation']}")
                print("\n==================================\n")

        # bit confusing: we want to make sure to eval and sample on 0th iteration
        # but also after the very last iteration. so we loop for step <= num_iterations
        # instead of just < num_iterations (one extra due to <=), only to do
        # the validation/sampling one last time, and then we break right here as we're done.
        if last_step:
            break

        # --------------- TRAINING SECTION BEGIN -----------------
        model.train()
        optimizer.zero_grad(set_to_none=True)
        # if we are trying to overfit a single batch, we reset the loader here
        if args.overfit_single_batch:
            train_loader.reset()
        # micro-batch loop where we do gradient accumulation to reach desired total batch size
        lossf = 0.0 # for getting the mean loss (as simple float) over the accumulation steps
        for micro_step in range(grad_accum_steps):
            # fetch a batch
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            if ddp:
                # we want only the last micro-step to sync grads in a DDP model
                # the official way to do this is with model.no_sync(), but that is a
                # context manager that bloats the code, so we just toggle this variable
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            # forward pass
            with ctx:
                _, loss = model(x, y, return_logits=False)
                # we have to scale the loss to account for gradient accumulation,
                # because the gradients just add on each successive backward().
                # addition of gradients corresponds to a SUM in the objective, but
                # instead of a SUM we want MEAN, so we scale the loss here
                loss = loss / grad_accum_steps
                lossf += loss.detach() # keep track of the mean loss
            # backward pass
            if not args.inference_only:
                loss.backward()
        if ddp:
            dist.all_reduce(lossf, op=dist.ReduceOp.AVG)
        lossf = lossf.item()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        # determine and set the learning rate for this iteration
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        # step the optimizer
        optimizer.step()
        # --------------- TRAINING SECTION END -------------------
        # everything that follows now is just diagnostics, prints, logging, etc.

        # wait on the CPU for all device work to end so we get accurate per-iteration timings below
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()
        # time and print
        t1 = time.time()
        # the 0th iteration is often an outlier (much slower) => skip logging it
        tokens_per_second = grad_accum_steps * ddp_world_size * B * T / (t1-t0)
        print0(f"step {step+1:4d}/{args.num_iterations} | train loss {lossf:.6f} | norm {norm:.4f} | lr {lr:.2e} | ({(t1-t0)*1000:.2f} ms | {tokens_per_second:.0f} tok/s)")
        # log to logile
        if master_process and logfile is not None:
            with open(logfile, "a") as f:
                f.write("s:%d trl:%f\n" % (step, lossf))

        # keep track of smooth timings, last 20 iterations
        if step > 0 and step > args.num_iterations - 20:
            timings.append(t1-t0)

    # print the average of the last 20 timings, to get something smooth-ish
    timings = timings[-20:]
    print0(f"final {len(timings)} iters avg: {np.mean(timings)*1000:.3f}ms")
    print0(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

    # -------------------------------------------------------------------------
    # clean up nice
    if ddp:
        destroy_process_group()
