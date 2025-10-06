"""
Evaluates the SwiGLU-modified GPT-2 model on the HellaSwag dataset.
This script is adapted from dev/data/hellaswag.py but uses our SwiGLU model.
"""

import os
import json
import torch
import torch.nn.functional as F
import tiktoken
import numpy as np
from tqdm import tqdm
from train_gpt2 import GPT, GPTConfig, SwiGLU  # Import SwiGLU to make sure it's registered

# Directory where HellaSwag data is stored
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "dev/data/hellaswag")

# Initialize the tokenizer
enc = tiktoken.get_encoding("gpt2")

def render_example(example):
    """
    Given the example as a dictionary, render it for evaluation
    """
    ctx = example["ctx"]
    label = example["label"]
    endings = example["endings"]

    # Gather tokens for context and each ending
    ctx_tokens = enc.encode(ctx)
    tok_rows = []
    mask_rows = []
    for end in endings:
        end_tokens = enc.encode(" " + end)  # Prepend space for GPT-2 tokenizer
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0]*len(ctx_tokens) + [1]*len(end_tokens))

    # Collate the examples with padding
    max_len = max(len(row) for row in tok_rows)
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)
    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, :len(tok_row)] = torch.tensor(tok_row)
        mask[i, :len(mask_row)] = torch.tensor(mask_row)

    return tokens, mask, label

def iterate_examples():
    """
    Iterate through the validation examples
    """
    with open(os.path.join(DATA_CACHE_DIR, "hellaswag_val.jsonl"), "r") as f:
        for line in f:
            example = json.loads(line)
            yield example

def load_weights_from_binary(model, filename):
    """
    Load weights from .bin file into PyTorch model
    """
    print(f"Loading weights from {filename}...")
    
    with open(filename, "rb") as f:
        # Read header
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
        magic = header[0]
        version = header[1]
        max_seq_len = header[2]
        vocab_size = header[3]
        n_layer = header[4]
        n_head = header[5]
        n_embd = header[6]
        padded_vocab_size = header[7] if version >= 3 else vocab_size
        
        print(f"Model metadata: layers={n_layer}, heads={n_head}, dim={n_embd}, version={version}")
        
        # Check if it's a SwiGLU model (should be version 3)
        is_swiglu = (version == 3)
        if not is_swiglu:
            print(f"WARNING: This binary might not be a SwiGLU model (version = {version}, expected 3)")
        
        # Create a state dict to hold all model parameters
        state_dict = {}
        
        # Helper function to read a tensor of the specified shape
        def read_tensor(shape, dtype=np.float32):
            size = np.prod(shape)
            tensor = np.frombuffer(f.read(size * np.dtype(dtype).itemsize), dtype=dtype).copy()
            return torch.from_numpy(tensor.reshape(shape))
        
        # Load embeddings - these don't need transposition
        state_dict["transformer.wte.weight"] = read_tensor((padded_vocab_size, n_embd))[:vocab_size]  # Trim to actual vocab size
        state_dict["transformer.wpe.weight"] = read_tensor((max_seq_len, n_embd))
        
        # Load layer weights
        for i in range(n_layer):
            # Layer normalization weights - these don't need transposition
            state_dict[f"transformer.h.{i}.ln_1.weight"] = read_tensor((n_embd,))
            state_dict[f"transformer.h.{i}.ln_1.bias"] = read_tensor((n_embd,))
            
            # Attention weights
            c_attn_weight = read_tensor((3 * n_embd, n_embd))
            c_attn_bias = read_tensor((3 * n_embd,))
            state_dict[f"transformer.h.{i}.attn.c_attn.weight"] = c_attn_weight  # Removed transpose
            state_dict[f"transformer.h.{i}.attn.c_attn.bias"] = c_attn_bias
            
            # Add the attention bias buffer (causal mask)
            mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
            mask = mask.view(1, 1, max_seq_len, max_seq_len)
            state_dict[f"transformer.h.{i}.attn.bias"] = mask
            
            proj_weight = read_tensor((n_embd, n_embd))
            proj_bias = read_tensor((n_embd,))
            state_dict[f"transformer.h.{i}.attn.c_proj.weight"] = proj_weight  # Removed transpose
            state_dict[f"transformer.h.{i}.attn.c_proj.bias"] = proj_bias
            
            # Layer normalization weights
            state_dict[f"transformer.h.{i}.ln_2.weight"] = read_tensor((n_embd,))
            state_dict[f"transformer.h.{i}.ln_2.bias"] = read_tensor((n_embd,))
            
            # MLP weights - different for SwiGLU
            hidden_dim = 4 * n_embd // 3  # SwiGLU hidden dimension
            
            # SwiGLU weights - removed transpositions
            gate_proj_weight = read_tensor((hidden_dim, n_embd))
            state_dict[f"transformer.h.{i}.mlp.gate_proj.weight"] = gate_proj_weight  # Removed transpose
            
            up_proj_weight = read_tensor((hidden_dim, n_embd))
            state_dict[f"transformer.h.{i}.mlp.up_proj.weight"] = up_proj_weight  # Removed transpose
            
            down_proj_weight = read_tensor((n_embd, hidden_dim))
            state_dict[f"transformer.h.{i}.mlp.down_proj.weight"] = down_proj_weight  # Removed transpose
        
        # Final layer norm
        state_dict["transformer.ln_f.weight"] = read_tensor((n_embd,))
        state_dict["transformer.ln_f.bias"] = read_tensor((n_embd,))
        
        # Tie weights between embedding and output
        state_dict["lm_head.weight"] = state_dict["transformer.wte.weight"]

        print("Loading weights into model...")
        
        # Load the state dict directly
        model.load_state_dict(state_dict, strict=False)
        
        print("Successfully loaded weights from binary file!")
    
    return model

@torch.no_grad()
def evaluate_model(model_path, device="cuda"):
    """
    Evaluate the model on HellaSwag
    """
    print(f"Loading model from {model_path}...")
    
    # Read the model config from the .bin file header
    with open(model_path, "rb") as f:
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
    
    # Extract model configuration from header
    vocab_size = header[3].item()  # Also read vocab_size from header
    n_layer = header[4].item()
    n_head = header[5].item() 
    n_embd = header[6].item()
    
    # Create model config and initialize model with the correct vocab size
    config = GPTConfig(n_layer=n_layer, n_head=n_head, n_embd=n_embd, vocab_size=vocab_size)
    model = GPT(config)
    
    # Load weights from binary file
    model = load_weights_from_binary(model, model_path)
    
    # Move model to device
    model.to(device)
    model.eval()
    
    # Start evaluation
    print("Starting evaluation...")
    num_correct_norm = 0
    num_correct = 0
    num_total = 0
    
    for example in tqdm(iterate_examples(), desc="Evaluating"):
        tokens, mask, label = render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)

        # Get logits from the model
        logits, _ = model(tokens)
        
        # Debug shapes
        print(f"Tokens shape: {tokens.shape}, Logits shape: {logits.shape}")
        
        # Ensure logits are the right shape
        if logits.dim() < 3:
            print(f"Warning: Unexpected logits dimension {logits.dim()}, expected at least 3")
            continue
            
        # Evaluate autoregressive loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_tokens = tokens[..., 1:].contiguous()
        shift_mask = mask[..., 1:].contiguous()
        
        # Debug shapes after shifting
        print(f"Shifted logits shape: {shift_logits.shape}, Shifted tokens shape: {shift_tokens.shape}")
        
        # Ensure we have valid data to compute loss
        if shift_logits.numel() == 0 or shift_tokens.numel() == 0:
            print("Warning: Empty tensors encountered, skipping example")
            continue
            
        try:
            # Calculate cross-entropy loss
            loss = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_tokens.reshape(-1),
                reduction='none'
            )
            
            # Reshape loss back to match the token dimensions
            loss = loss.view(shift_tokens.size())
            
            # Apply mask to focus on ending tokens only
            masked_loss = loss * shift_mask
            
            # Two scoring methods: sum loss and normalized loss
            sum_loss = masked_loss.sum(dim=1)
            avg_loss = sum_loss / shift_mask.sum(dim=1).clamp(min=1)  # Avoid division by zero
            
            # Get predictions from both methods
            pred = sum_loss.argmin().item()
            pred_norm = avg_loss.argmin().item()
            
            # Accumulate stats
            num_total += 1
            num_correct += int(pred == label)
            num_correct_norm += int(pred_norm == label)
            
            # After first successful example, remove debug prints
            if num_total == 1:
                print("First example processed successfully, disabling debug prints")
        
        except Exception as e:
            print(f"Error processing example: {e}")
            # Try to continue with the next example
            continue
        
        # Print progress every 100 examples
        if num_total % 100 == 0:
            print(f"{num_total} examples: acc={num_correct/num_total:.4f}, acc_norm={num_correct_norm/num_total:.4f}")
            
    # Final results
    print(f"Final results on {num_total} examples:")
    print(f"Accuracy (sum loss): {num_correct/num_total:.4f}")
    print(f"Accuracy (normalized loss): {num_correct_norm/num_total:.4f}")
    return num_correct/num_total, num_correct_norm/num_total

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="gpt2_d12.bin", help="Path to the model .bin file")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda, cpu)")
    args = parser.parse_args()
    
    # Check if HellaSwag dataset exists
    if not os.path.exists(os.path.join(DATA_CACHE_DIR, "hellaswag_val.jsonl")):
        print("HellaSwag dataset not found. Please run 'python dev/data/hellaswag.py' first to download it.")
        exit(1)
        
    # Check if model file exists
    if not os.path.exists(args.model_path):
        print(f"Model file {args.model_path} not found. Please train a model first.")
        exit(1)
        
    acc, acc_norm = evaluate_model(args.model_path, args.device)
    print(f"Done. Final Accuracies: acc={acc:.4f}, acc_norm={acc_norm:.4f}")

