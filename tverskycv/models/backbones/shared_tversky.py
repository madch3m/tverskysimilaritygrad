import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from transformers import GPT2Model, GPT2Config, GPT2Tokenizer
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
class TverskyAttentionShared(nn.Module):
    def __init__(self, embed_dim, num_heads, feature_key='main', dropout=0.1, bias=True, alpha=0.5,beta=0.5, gamma=1.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        '''
        Combination of the query key and value vectors used for the attention in the original GPT model replaced with differentiable Tversky Linear activation function
        '''
        
        self.q_proj = SharedTverskyLinear(embed_dim, embed_dim, feature_key=feature_key,alpha=alpha, beta=beta, gamma=gamma, bias=bias)
        self.k_proj = SharedTverskyLinear(embed_dim,embed_dim, feature_key=feature_key,alpha=alpha,beta=beta,gamma=gamma,bias=bias)
        self.v_proj = SharedTverskyLinear(embed_dim,embed_dim, feature_key=feature_key,alpha=alpha,beta=beta, gamma=gamma, bias=bias)
        self.out_proj = SharedTverskyLinear(embed_dim, embed_dim, feature_key=feature_key, alpha=alpha, beta=beta, gamma=gamma, bias=bias)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)


    def __split_heads(self, tensor, batch_size):
        return tensor.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,2)
    
    def __merge_heads(self, tensor, batch_size):
        return tensor.transpose(1,2).contiguous().view(batch_size,-1, self.embed_dim)
    
    def forward(self, hidden_states, attention_mask=None, layer_past=None, use_cache=False, output_attentions=False):
        batch_size, seq_length = hidden_states.size()[:2]
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        # Split heads
        query = self.__split_heads(query, batch_size)
        key = self.__split_heads(key, batch_size)
        value = self.__split_heads(value, batch_size)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat([past_key, key],dim=-2)
            value = torch.cat([past_value, value],dim=-2)
        present = (key, value) if use_cache else None

        attn_weights = torch.matmul(query, key.transpose(-1,-2))
        attn_weights = attn_weights / (self.head_dim**0.5)  # Fixed: was self.attn_weights and self.head_dimm

        if attention_mask is not None:
            attn_weights = attn_weights.masked_fill(attention_mask == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, value)
        attn_output = self.__merge_heads(attn_output, batch_size)  # Fixed: was self._merge_heads
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs = outputs + (attn_weights,)  # Fixed: was output +=
        return outputs

class SharedTverskyLinear(nn.Module):
    def __init__(self, in_features, out_features, feature_key="main", alpha=0.5,beta=0.5,gamma=1.0, bias=True, share_features=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.feature_key = feature_key

        self.prototypes = nn.Parameter(torch.randn(out_features,in_features))

        self.register_feature = GlobalFeature()
        feature_matrix_key = f"{feature_key}_{in_features}"

        if not self.register_feature.has_key(feature_matrix_key):
            features = nn.Parameter(torch.randn(in_features, in_features))
            self.register_feature.register_feature(feature_matrix_key,features)
        
        self._feature_matrix_key = feature_matrix_key

        if share_features:
            param_key = f"tversky_params_{feature_key}"
            if not self.register_feature.has_key(param_key):
                params = {
                    'alpha': nn.Parameter(torch.tensor(alpha)),
                    'beta': nn.Parameter(torch.tensor(beta)),
                    'gamma': nn.Parameter(torch.tensor(gamma))
                }
                self.register_feature.register_feature(param_key, params)  # Fixed: was self.register_feature(param_key, params)
            
            self._param_key = param_key
            self._alpha = None
            self._beta = None
            self._gamma = None
        else:
            self._alpha = nn.Parameter(torch.tensor(alpha))
            self._beta = nn.Parameter(torch.tensor(beta))
            self._gamma = nn.Parameter(torch.tensor(gamma))
            self._param_key = None
        
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)
        
        self._reset_parameters()

    @property
    def features(self):
        return self.register_feature.get_feature(self._feature_matrix_key)  # Fixed: was self.registry
    
    @property
    def alpha(self):
        if self._param_key:
            return self.register_feature.get_feature(self._param_key)['alpha']
        return self._alpha
    
    @alpha.setter
    def alpha(self, value):
        if not self._param_key:
            self._alpha = value
    
    @property
    def beta(self):
        if self._param_key:
            return self.register_feature.get_feature(self._param_key)['beta']
        return self._beta
    
    @beta.setter
    def beta(self, value):
        if not self._param_key:
            self._beta = value

    @property
    def gamma(self):
        if self._param_key:
            return self.register_feature.get_feature(self._param_key)['gamma']
        return self._gamma
    
    @gamma.setter
    def gamma(self, value):
        if not self._param_key:
            self._gamma = value

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.prototypes)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def tversky_similarity_batch(self, x, prototypes):
        batch_size = x.size(0)
        features = self.features
        x_activations = torch.matmul(x, features)
        x_features = F.relu(x_activations)
        prototypes_activations = torch.matmul(prototypes,features)
        prototype_features = F.relu(prototypes_activations)

        x_features_expanded = x_features.unsqueeze(1)
        prototype_features_expanded = prototype_features.unsqueeze(0)

        common_features = torch.sum(torch.min(x_features_expanded, prototype_features_expanded), dim=-1)
        distinctive_x = torch.sum(F.relu(x_features_expanded - prototype_features_expanded),dim=-1)
        distinctive_prototypes = torch.sum(F.relu(prototype_features_expanded - x_features_expanded),dim=-1)

        numerator = self.gamma * common_features
        denominator = (self.gamma * common_features + torch.abs(self.alpha) * distinctive_x + torch.abs(self.beta) * distinctive_prototypes + 1e-8)
        similarity = numerator / denominator
        return similarity
    
    def forward(self, x):
        original_shape = x.shape[:-1]
        x_flattened = x.view(-1, self.in_features)
        output = self.tversky_similarity_batch(x_flattened, self.prototypes)

        if self.bias is not None:
            output += self.bias
        output = output.view(*original_shape, self.out_features)
        return output
    


class SharedTverskyNetwork(nn.Module):
    def __init__(self, embed_dim, intermediate_dim, feature_key='main', dropout=0.1, alpha=0.5,beta=0.5, gamma=1.0):
        super().__init__()
        self.layer1 = SharedTverskyLinear(embed_dim,intermediate_dim, feature_key=feature_key,alpha=alpha,beta=beta,gamma=gamma)
        self.layer2 = SharedTverskyLinear(intermediate_dim,embed_dim,feature_key=f"{feature_key}_intermediate", alpha=alpha, beta=beta, gamma=gamma)
        self.act = F.relu  # Fixed: was F.relu()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, hidden_states):
        hidden_state = self.layer1(hidden_states)
        hidden_state = self.act(hidden_state)
        hidden_state = self.layer2(hidden_state)
        hidden_state = self.dropout(hidden_state)
        return hidden_state


class TverskyTransformerBlock(nn.Module):
    def __init__(self, config, feature_key='main', alpha=0.5, beta=0.5, gamma=1.0):
        super().__init__()
        hidden_size = config.n_embed
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = TverskyAttentionShared(embed_dim=hidden_size, num_heads=config.n_head, feature_key=feature_key, dropout=config.resid_pdrop, alpha=alpha,beta=beta, gamma=gamma)

        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.network = SharedTverskyNetwork(embed_dim=hidden_size, intermediate_dim=inner_dim, feature_key=feature_key, dropout=config.resid_pdrop, alpha=alpha,beta=beta,gamma=gamma)
    def forward(self, hidden_states, attention_mask=None, layer_past=None, use_cache=False, output_attentions=False):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(hidden_states, attention_mask=attention_mask, layer_past=layer_past, use_cache=use_cache)

        attn_output = attn_outputs[0]
        outputs = attn_outputs[1:]

        hidden_states = residual + attn_output

        residual = hidden_states

        hidden_states = self.ln_2(hidden_states)

        ffn_states = self.network(hidden_states)

        hidden_states = residual + ffn_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        
        else:
            outputs = (hidden_states,) + outputs[1:]
        return outputs
    

class GlobalFeature:
    _instance = None
    _feature_matrices = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GlobalFeature, cls).__new__(cls)
            cls._instance._feature_matrices = {}
        return cls._instance
    def register_feature(self, key, feature_matrix):
        self._feature_matrices[key] = feature_matrix

    def get_feature(self, key):
        return self._feature_matrices.get(key)
    
    def clear(self):
        self._feature_matrices.clear()
    def has_key(self, key):
        return key in self._feature_matrices 


