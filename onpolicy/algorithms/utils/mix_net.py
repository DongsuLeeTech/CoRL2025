import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class VDNMixer(nn.Module):
    """
    Value Decomposition Network (VDN) for CACOM algorithm.
    Implements a simple additive value decomposition.
    """
    def __init__(self):
        super(VDNMixer, self).__init__()

    def forward(self, agent_q_values):
        """
        Mixes the agent's Q-values by simple addition.
        
        Args:
            agent_q_values (torch.Tensor): Q-values from individual agents [batch_size, n_agents, 1]
            
        Returns:
            torch.Tensor: Mixed Q-value for the team [batch_size, 1]
        """
        return torch.sum(agent_q_values, dim=1, keepdim=True)


class QMixer(nn.Module):
    """
    QMixer Network for CACOM algorithm.
    Implements a state-dependent mixing network that combines agent Q-values.
    """
    def __init__(self, args):
        """
        Initialize QMixer network.
        
        Args:
            args: Arguments containing network configuration
                - num_agents: Number of agents
                - state_shape: Shape of state input (optional, defaults to obs_shape)
                - mixing_embed_dim: Dimension of mixing network (optional, default 32)
                - hypernet_layers: Number of hypernetwork layers (optional, default 1)
                - hypernet_embed: Dimension of hypernetwork embedding (optional)
        """
        super(QMixer, self).__init__()
        
        self.num_agents = getattr(args, 'num_agents', 2)  # Default to 2 agents if not specified
        
        # Get state shape from args
        if hasattr(args, 'state_shape'):
            self.state_dim = int(np.prod(args.state_shape))
        elif hasattr(args, 'obs_shape'):
            self.state_dim = int(np.prod(args.obs_shape))
        else:
            self.state_dim = self.num_agents * 10  # Default state dimension if not specified
            
        # Get mixing network parameters
        self.embed_dim = getattr(args, 'mixing_embed_dim', 32)  # Default to 32 if not specified
        self.hypernet_layers = getattr(args, 'hypernet_layers', 1)  # Default to 1 if not specified
        
        if getattr(args, 'hypernet_embed', None) is None:
            self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.num_agents)
            self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
        else:
            hypernet_embed = args.hypernet_embed
            self.hyper_w_1 = nn.Sequential(
                nn.Linear(self.state_dim, hypernet_embed),
                nn.ReLU(),
                nn.Linear(hypernet_embed, self.embed_dim * self.num_agents)
            )
            self.hyper_w_final = nn.Sequential(
                nn.Linear(self.state_dim, hypernet_embed),
                nn.ReLU(),
                nn.Linear(hypernet_embed, self.embed_dim)
            )
            
        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(
            nn.Linear(self.state_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1)
        )

    def forward(self, agent_qs, states):
        """
        Forward pass of mixing network.
        
        Args:
            agent_qs: Q-values from individual agents [batch_size, n_agents, 1]
            states: State input for the mixing network [batch_size, state_dim]
            
        Returns:
            torch.Tensor: Mixed Q-value for the team [batch_size, 1]
        """
        batch_size = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, 1, self.num_agents)
        
        # First layer
        w1 = torch.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.num_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)
        
        # Second layer
        w_final = torch.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)
        
        # Compute final output
        y = torch.bmm(hidden, w_final) + v
        
        return y.view(batch_size, -1, 1)