import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from transformers import GPT2Config
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from .shared_tversky import TverskyTransformerBlock, GlobalFeature


class TverskyGPTModel(nn.Module):
    """
    GPT-like model using Tversky Attention with shared feature bank for parameter reduction.
    
    This model uses the GlobalFeature bank to share feature matrices and Tversky parameters
    across layers, significantly reducing the total parameter count compared to standard GPT.
    """
    
    def __init__(
        self,
        config: GPT2Config,
        feature_key: str = 'shared',
        alpha: float = 0.5,
        beta: float = 0.5,
        gamma: float = 1.0,
        share_across_layers: bool = True,
    ):
        """
        Args:
            config: GPT2Config with model hyperparameters
            feature_key: Base key for feature sharing. If share_across_layers=True, 
                        all layers use the same key to maximize parameter sharing.
            alpha: Tversky alpha parameter (controls false positives)
            beta: Tversky beta parameter (controls false negatives)
            gamma: Tversky gamma parameter (controls common features weight)
            share_across_layers: If True, all layers share the same feature_key,
                                maximizing parameter reduction via GlobalFeature bank.
        """
        super().__init__()
        self.config = config
        self.embed_dim = config.n_embed
        self.num_layers = config.n_layer
        self.vocab_size = config.vocab_size
        self.max_position_embeddings = config.n_positions
        
        # Token embeddings
        self.wte = nn.Embedding(config.vocab_size, config.n_embed)
        
        # Position embeddings
        self.wpe = nn.Embedding(config.n_positions, config.n_embed)
        
        # Dropout
        self.drop = nn.Dropout(config.embd_pdrop)
        
        # Tversky Transformer blocks
        # Use shared feature_key across all layers to maximize parameter sharing
        if share_across_layers:
            # All layers share the same feature_key, maximizing parameter reduction
            layer_feature_keys = [feature_key] * config.n_layer
        else:
            # Each layer gets its own feature_key (less parameter sharing)
            layer_feature_keys = [f"{feature_key}_layer_{i}" for i in range(config.n_layer)]
        
        self.h = nn.ModuleList([
            TverskyTransformerBlock(
                config,
                feature_key=layer_feature_keys[i],
                alpha=alpha,
                beta=beta,
                gamma=gamma
            )
            for i in range(config.n_layer)
        ])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(config.n_embed, eps=config.layer_norm_epsilon)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights following GPT-2 initialization scheme."""
        # Token embeddings
        nn.init.normal_(self.wte.weight, std=0.02)
        # Position embeddings
        nn.init.normal_(self.wpe.weight, std=0.02)
    
    def get_input_embeddings(self):
        return self.wte
    
    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        Forward pass of TverskyGPT model.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_length)
            past_key_values: Cached key-value pairs for autoregressive generation
            attention_mask: Attention mask
            position_ids: Position IDs
            inputs_embeds: Optional pre-computed embeddings
            use_cache: Whether to return past_key_values
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return all hidden states
            return_dict: Whether to return a dict or tuple
        
        Returns:
            CausalLMOutputWithCrossAttentions or tuple
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        
        batch_size = input_shape[0]
        
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])
        
        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
        
        # Prepare attention mask
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=self.wte.weight.dtype)
            attention_mask = (1.0 - attention_mask) * -10000.0
        
        # Get embeddings
        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        
        hidden_states = self.drop(hidden_states)
        
        output_shape = input_shape + (hidden_states.size(-1),)
        
        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        
        # Apply transformer blocks
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            outputs = block(
                hidden_states,
                attention_mask=attention_mask,
                layer_past=layer_past,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            
            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)
            
            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
        
        hidden_states = self.ln_f(hidden_states)
        
        hidden_states = hidden_states.view(*output_shape)
        
        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)
        
        return CausalLMOutputWithCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


def create_tversky_gpt_from_config(
    config: GPT2Config,
    feature_key: str = 'shared',
    alpha: float = 0.5,
    beta: float = 0.5,
    gamma: float = 1.0,
    share_across_layers: bool = True,
) -> TverskyGPTModel:
    """
    Configuration helper to create a TverskyGPT model from GPT2Config.
    
    This function creates a TverskyGPT model that uses the GlobalFeature bank
    to share parameters across layers, significantly reducing the total parameter count.
    
    Args:
        config: GPT2Config with model hyperparameters (vocab_size, n_embed, n_layer, etc.)
        feature_key: Base key for feature sharing. When share_across_layers=True,
                    all layers use this same key to maximize parameter sharing.
        alpha: Tversky alpha parameter (default: 0.5)
        beta: Tversky beta parameter (default: 0.5)
        gamma: Tversky gamma parameter (default: 1.0)
        share_across_layers: If True (default), all layers share the same feature_key,
                           maximizing parameter reduction. Set to False for layer-specific features.
    
    Returns:
        TverskyGPTModel instance
    
    Example:
        >>> from transformers import GPT2Config
        >>> config = GPT2Config(
        ...     vocab_size=50257,
        ...     n_embed=768,
        ...     n_layer=12,
        ...     n_head=12,
        ...     n_positions=1024
        ... )
        >>> model = create_tversky_gpt_from_config(config, share_across_layers=True)
        >>> # Model now uses shared feature bank, reducing parameters significantly
    """
    return TverskyGPTModel(
        config=config,
        feature_key=feature_key,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        share_across_layers=share_across_layers,
    )


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Helper function to count parameters in a model.
    
    Args:
        model: PyTorch model
        trainable_only: If True, only count trainable parameters
    
    Returns:
        Total number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def get_shared_parameter_info(model: TverskyGPTModel) -> dict:
    """
    Get information about shared parameters via GlobalFeature bank.
    
    Args:
        model: TverskyGPTModel instance
    
    Returns:
        Dictionary with information about shared parameters
    """
    global_feature = GlobalFeature()
    shared_features = {}
    
    # Count shared feature matrices
    for key in global_feature._feature_matrices.keys():
        if isinstance(global_feature._feature_matrices[key], nn.Parameter):
            shared_features[key] = global_feature._feature_matrices[key].numel()
        elif isinstance(global_feature._feature_matrices[key], dict):
            # Tversky parameters (alpha, beta, gamma)
            total = sum(p.numel() for p in global_feature._feature_matrices[key].values() if isinstance(p, nn.Parameter))
            shared_features[key] = total
    
    total_shared = sum(shared_features.values())
    total_model = count_parameters(model, trainable_only=True)
    
    return {
        'shared_parameters': shared_features,
        'total_shared_params': total_shared,
        'total_model_params': total_model,
        'shared_percentage': (total_shared / total_model * 100) if total_model > 0 else 0.0,
    }

