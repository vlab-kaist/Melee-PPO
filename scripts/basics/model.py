import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.distributions.categorical import Categorical
from skrl.models.torch import Model, CategoricalMixin, DeterministicMixin
from enum import Enum

class AgentType(Enum):
    CPU = 1
    MLP = 2
    STACK = 3
    GRU = 4

class Policy(CategoricalMixin, Model):
    def __init__(self, observation_space, action_space, device, unnormalized_log_prob=True):
        Model.__init__(self, observation_space, action_space, device)
        CategoricalMixin.__init__(self, unnormalized_log_prob)

        self.linear_layer_1 = nn.Linear(self.num_observations, 256)
        self.linear_layer_2 = nn.Linear(256, 256)
        self.output_layer = nn.Linear(256, self.num_actions)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def compute(self, inputs, role):
        x = F.relu(self.linear_layer_1(inputs["states"]))
        x = F.relu(self.linear_layer_2(x))
        return self.output_layer(x), {}

class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        self.net.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}
    

class SwiGLU(nn.Module):
    def __init__(self, d_model, ffn_size=2048):
        super(SwiGLU, self).__init__()
        self.linear1 = nn.Linear(d_model, ffn_size)
        self.linear2 = nn.Linear(d_model, ffn_size)
        self.linear3 = nn.Linear(ffn_size, d_model)

    def forward(self, x):
        return self.linear3(F.silu(self.linear1(x)) * self.linear2(x))

class GRUAttentionBlock(nn.Module):
    def __init__(self, d_model):
        super(GRUAttentionBlock, self).__init__()
        self.gru = nn.GRU(d_model, d_model, batch_first=True)
        
    def forward(self, x, hidden_state):
        residual = x
        if hidden_state is not None and hidden_state.ndim + 1 == x.ndim:
            hidden_state = hidden_state[None]
        x, hidden_state = self.gru(x, hidden_state)
        return residual, hidden_state

class TransformerGRU(nn.Module):
    def __init__(self, input_size, d_model, hidden_size, num_layers, num_actions, sequence_length):
        super(TransformerGRU, self).__init__()
        self.d_model = d_model
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length

        self.pre_embed = nn.Linear(input_size, d_model)
        
        self.layers = nn.ModuleList([
            nn.ModuleList([GRUAttentionBlock(d_model), SwiGLU(d_model, hidden_size), nn.LayerNorm(d_model), nn.LayerNorm(d_model)]) 
            for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(d_model)
        self.output_layer = nn.Linear(d_model, num_actions)
        
        self._initialize_weights()
        self.output_layer.weight.data.fill_(0)
        self.output_layer.bias.data.fill_(0)

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.normal_(module.weight, mean=0.0, std=0.02)
                module.weight.contiguous()
                if module.bias is not None:
                    init.constant_(module.bias, 0)
            elif isinstance(module, nn.GRU):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)
                    param.contiguous()

    def forward(self, x, hidden_states=None):
        with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            if hidden_states is None:
                hidden_states = [None] * len(self.layers)
            
            x = self.pre_embed(x)
            for i, (gru_block, ffn_block, norm1, norm2) in enumerate(self.layers):
                y, hidden_states[i] = gru_block(norm1(x), hidden_states[i])
                x = x + y
                x = x + ffn_block(norm2(x))
            x = self.final_norm(x)
            return self.output_layer(x).float(), hidden_states
    
class GRUPolicy(CategoricalMixin, Model):
    def __init__(self, observation_space, action_space, device, unnormalized_log_prob=True,
                 num_envs=1, num_layers=4, hidden_size=1024, ffn_size=2048, sequence_length=64):
        Model.__init__(self, observation_space, action_space, device)
        CategoricalMixin.__init__(self, unnormalized_log_prob)

        self.num_envs = num_envs
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.ffn_size = ffn_size
        self.sequence_length = sequence_length

        self.transformer_gru = TransformerGRU(input_size=self.num_observations,
                                              d_model=self.hidden_size,
                                              hidden_size=self.ffn_size,
                                              num_layers=self.num_layers,
                                              num_actions=self.num_actions,
                                              sequence_length=self.sequence_length)

    def get_specification(self):
        # batch size (N) is the number of envs
        return {"rnn": {"sequence_length": self.sequence_length,
                        "sizes": [(self.num_layers, self.num_envs, self.hidden_size)]}}  # hidden states (D * num_layers, N, Hout)

    def compute(self, inputs, role):
        states = inputs["states"]
        terminated = inputs.get("terminated", None)
        hidden_states = inputs["rnn"][0]

        if hidden_states is None:
            hidden_states = [None] * self.num_layers

        # Training
        if self.training:
            rnn_input = states.reshape(-1, self.sequence_length, states.shape[-1])  # (N, L, Hin): N=batch_size, L=sequence_length
            hidden_states = [h.reshape(-1, self.sequence_length, self.hidden_size)[:, 0, :].contiguous() for h in hidden_states]  # Reshape hidden states
            
            # Reset the RNN state in the middle of a sequence
            if terminated is not None and torch.any(terminated):
                rnn_outputs = []
                terminated = terminated.view(-1, self.sequence_length)
                indexes = [0] + (terminated[:,:-1].any(dim=0).nonzero(as_tuple=True)[0] + 1).tolist() + [self.sequence_length]

                for i in range(len(indexes) - 1):
                    i0, i1 = indexes[i], indexes[i + 1]
                    rnn_output, hidden_states = self.transformer_gru(rnn_input[:, i0:i1, :], hidden_states)
                    for j in range(self.num_layers):
                        hidden_states[j][:, (terminated[:, i1-1]), :] = 0
                    rnn_outputs.append(rnn_output)

                rnn_output = torch.cat(rnn_outputs, dim=1)
            # No need to reset the RNN state in the sequence
            else:
                rnn_output, hidden_states = self.transformer_gru(rnn_input, hidden_states)
        # Rollout
        else:
            rnn_input = states.view(-1, 1, states.shape[-1])  # (N, L, Hin): N=num_envs, L=1
            rnn_output, hidden_states = self.transformer_gru(rnn_input, hidden_states)

        # Flatten the RNN output
        rnn_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)  # (N, L, D * Hout) -> (N * L, D * Hout)

        # Pendulum-v1 action_space is -2 to 2
        return rnn_output, {"rnn": [hidden_states]}

class GRUValue(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 num_envs=1, num_layers=4, hidden_size=1024, ffn_size=2048, sequence_length=64):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.num_envs = num_envs
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.ffn_size = ffn_size
        self.sequence_length = sequence_length

        self.transformer_gru = TransformerGRU(input_size=self.num_observations,
                                              d_model=self.hidden_size,
                                              hidden_size=self.ffn_size,
                                              num_layers=self.num_layers,
                                              num_actions=1,
                                              sequence_length=self.sequence_length)

    def get_specification(self):
        # batch size (N) is the number of envs
        return {"rnn": {"sequence_length": self.sequence_length,
                        "sizes": [(self.num_layers, self.num_envs, self.hidden_size)]}}  # hidden states (D * num_layers, N, Hout)

    def compute(self, inputs, role):
        states = inputs["states"]
        terminated = inputs.get("terminated", None)
        hidden_states = inputs["rnn"][0]

        if hidden_states is None:
            hidden_states = [None] * self.num_layers

        # Training
        if self.training:
            rnn_input = states.reshape(-1, self.sequence_length, states.shape[-1])  # (N, L, Hin): N=batch_size, L=sequence_length
            hidden_states = [h.reshape(-1, self.sequence_length, self.hidden_size)[:, 0, :].contiguous() for h in hidden_states]  # Reshape hidden states
            
            # Reset the RNN state in the middle of a sequence
            if terminated is not None and torch.any(terminated):
                rnn_outputs = []
                terminated = terminated.view(-1, self.sequence_length)
                indexes = [0] + (terminated[:,:-1].any(dim=0).nonzero(as_tuple=True)[0] + 1).tolist() + [self.sequence_length]

                for i in range(len(indexes) - 1):
                    i0, i1 = indexes[i], indexes[i + 1]
                    rnn_output, hidden_states = self.transformer_gru(rnn_input[:, i0:i1, :], hidden_states)
                    for j in range(self.num_layers):
                        hidden_states[j][:, (terminated[:, i1-1]), :] = 0
                    rnn_outputs.append(rnn_output)

                rnn_output = torch.cat(rnn_outputs, dim=1)
            # No need to reset the RNN state in the sequence
            else:
                rnn_output, hidden_states = self.transformer_gru(rnn_input, hidden_states)
        # Rollout
        else:
            rnn_input = states.view(-1, 1, states.shape[-1])  # (N, L, Hin): N=num_envs, L=1
            rnn_output, hidden_states = self.transformer_gru(rnn_input, hidden_states)

        # Flatten the RNN output
        rnn_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)  # (N, L, D * Hout) -> (N * L, D * Hout)

        # Pendulum-v1 action_space is -2 to 2
        return rnn_output, {"rnn": [hidden_states]}
