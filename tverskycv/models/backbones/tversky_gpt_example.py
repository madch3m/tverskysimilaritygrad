"""
Example usage of TverskyGPT model with parameter sharing via GlobalFeature bank.

This example demonstrates how to create a TverskyGPT model from GPT2Config
and how the shared feature bank reduces parameter count.
"""

from transformers import GPT2Config
from tverskycv.models.backbones.tversky_gpt import (
    create_tversky_gpt_from_config,
    count_parameters,
    get_shared_parameter_info
)


def example_basic_usage():
    """Basic example of creating a TverskyGPT model."""
    # Create a GPT2Config (small model for demonstration)
    config = GPT2Config(
        vocab_size=50257,
        n_embed=256,      # Embedding dimension
        n_layer=6,         # Number of transformer layers
        n_head=8,          # Number of attention heads
        n_positions=1024,  # Maximum sequence length
        n_inner=1024,      # FFN inner dimension
        resid_pdrop=0.1,   # Residual dropout
        embd_pdrop=0.1,    # Embedding dropout
    )
    
    # Create model with shared parameters (maximizes parameter reduction)
    model = create_tversky_gpt_from_config(
        config=config,
        feature_key='shared',
        alpha=0.5,
        beta=0.5,
        gamma=1.0,
        share_across_layers=True,  # All layers share the same feature_key
    )
    
    print(f"Model created with {count_parameters(model):,} parameters")
    
    # Get information about shared parameters
    info = get_shared_parameter_info(model)
    print(f"\nShared Parameter Info:")
    print(f"  Total shared parameters: {info['total_shared_params']:,}")
    print(f"  Total model parameters: {info['total_model_params']:,}")
    print(f"  Shared percentage: {info['shared_percentage']:.2f}%")
    
    return model


def example_comparison():
    """Compare shared vs non-shared parameter models."""
    config = GPT2Config(
        vocab_size=50257,
        n_embed=128,
        n_layer=4,
        n_head=4,
        n_positions=512,
    )
    
    # Model with shared parameters (maximizes reduction)
    model_shared = create_tversky_gpt_from_config(
        config=config,
        share_across_layers=True,
    )
    
    # Model without shared parameters (each layer has own features)
    model_not_shared = create_tversky_gpt_from_config(
        config=config,
        share_across_layers=False,
    )
    
    params_shared = count_parameters(model_shared)
    params_not_shared = count_parameters(model_not_shared)
    
    reduction = ((params_not_shared - params_shared) / params_not_shared) * 100
    
    print(f"Model with shared features: {params_shared:,} parameters")
    print(f"Model without shared features: {params_not_shared:,} parameters")
    print(f"Parameter reduction: {reduction:.2f}%")
    
    return model_shared, model_not_shared


def example_forward_pass():
    """Example of forward pass with the model."""
    import torch
    
    config = GPT2Config(
        vocab_size=50257,
        n_embed=128,
        n_layer=2,
        n_head=4,
        n_positions=256,
    )
    
    model = create_tversky_gpt_from_config(config)
    model.eval()
    
    # Create dummy input
    batch_size = 2
    seq_length = 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        hidden_states = outputs.last_hidden_state
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {hidden_states.shape}")
    print(f"Expected: ({batch_size}, {seq_length}, {config.n_embed})")
    
    return outputs


if __name__ == "__main__":
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)
    model1 = example_basic_usage()
    
    print("\n" + "=" * 60)
    print("Example 2: Comparison (Shared vs Non-Shared)")
    print("=" * 60)
    model2, model3 = example_comparison()
    
    print("\n" + "=" * 60)
    print("Example 3: Forward Pass")
    print("=" * 60)
    outputs = example_forward_pass()

