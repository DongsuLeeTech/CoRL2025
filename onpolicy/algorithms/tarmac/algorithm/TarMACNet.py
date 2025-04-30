import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical

from onpolicy.utils.util import get_shape_from_obs_space
from onpolicy.algorithms.utils.util import check
from onpolicy.algorithms.utils.rnn import RNNLayer


class TarCommNetMLP(nn.Module):
    def __init__(self, args, num_inputs, action_space, device=torch.device("cpu")):
        super(TarCommNetMLP, self).__init__()
        self.args = args
        self.nagents = args.num_agents
        self.hid_size = args.hidden_size
        self.comm_passes = 3
        self.batch_size = args.n_rollout_threads
        self.tpdv = dict(dtype=torch.float32, device=device)

        if action_space.__class__.__name__ == "Discrete":
            action_dim = action_space.n
        elif action_space.__class__.__name__ == "Box":
            action_dim = action_space.shape[0]
        elif action_space.__class__.__name__ == "MultiBinary":
            action_dim = action_space.shape[0]
        elif action_space.__class__.__name__ == "MultiDiscrete":
            self.multi_discrete = True
            action_dims = action_space.high - action_space.low + 1
        else:  # discrete + continous
            self.mixed_action = True
            continous_dim = action_space[0].shape[0]
            discrete_dim = action_space[1].n

        self.action_mean = nn.Linear(self.hid_size, action_dim)
        self.action_log_std = nn.Parameter(torch.zeros(1, action_dim))

        self.init_std = args.init_std if hasattr(args, 'comm_init_std') else 0.2

        # Mask for communication
        # if self.args.comm_mask_zero:
        #     self.comm_mask = torch.zeros(self.nagents, self.nagents)
        # else:
        self.comm_mask = torch.ones(self.nagents, self.nagents) - torch.eye(self.nagents, self.nagents)

        obs_shape = get_shape_from_obs_space(num_inputs)
        obs_dim = obs_shape[0]
        self.encoder = nn.Linear(obs_dim, self.hid_size)
        self.hidd_encoder = nn.Linear(self.hid_size, self.hid_size)

        self.init_hidden(self.batch_size)
        self.f_modules = nn.ModuleList([nn.Linear(self.hid_size, self.hid_size)
                                        for _ in range(self.comm_passes)])
        # self.f_module = RNNLayer(self.hid_size, self.hid_size, args.recurrent_N, args.use_orthogonal)

        # Our main function for converting current hidden state to next state
        self.C_module = nn.Linear(self.hid_size, self.hid_size)
        self.C_modules = nn.ModuleList([self.C_module
                                        for _ in range(self.comm_passes)])

        self.tanh = nn.Tanh()

        self.value_head = nn.Linear(self.hid_size, 1)

        self.state2query = nn.Linear(self.hid_size, 16)
        self.state2key = nn.Linear(self.hid_size, 16)
        self.state2value = nn.Linear(self.hid_size, self.hid_size)

        self.to(device)

    def forward_state_encoder(self, x):
        x = self.encoder(x)
        return x

    def forward(self, x, masks, info={}):
        x = check(x).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        x = self.forward_state_encoder(x)
        hidden_state = self.tanh(x)

        batch_size = int(x.shape[0] / self.nagents)
        if batch_size == 0:
            batch_size = 1
        x = x.view(batch_size, self.nagents, self.hid_size)
        hidden_state = hidden_state.view(batch_size, self.nagents, self.hid_size)
        masks = masks.view(batch_size, self.nagents, 1)

        # print(hidden_state.shape, batch_size, x.shape)
        n = self.nagents

        for i in range(self.comm_passes):
            # Choose current or prev depending on recurrent
            comm = hidden_state.view(batch_size, n, self.hid_size)

            # if self.args.comm_mask_zero:
            #     comm_mask = torch.zeros_like(comm)
            #     comm = comm * comm_mask

            query = self.state2query(comm)
            key = self.state2key(comm)
            value = self.state2value(comm)

            # scores
            scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.hid_size)
            scores = scores.masked_fill(masks.expand(-1, -1, scores.size(-1)) == 0, -1e9)

            # softmax + weighted sum
            attn = F.softmax(scores, dim=-1)
            attn = attn * masks.expand(-1, -1, scores.size(-1))  # cannot use inplace operation *=
            comm = torch.matmul(attn, value)
            comm *= masks.expand(-1, -1, scores.size(-1))[:, 0].unsqueeze(-1).expand(batch_size, n, self.hid_size)
            c = self.C_modules[i](comm)

            # skip connection - combine comm. matrix and encoded input for all agents
            hidden_state = sum([x, self.f_modules[i](hidden_state), c])
            hidden_state = self.tanh(hidden_state)
            # inp = x + c
            # inp = inp.view(batch_size * n, self.hid_size)
            #
            # output, hidden_state = self.f_module(inp, hidden_state, masks.view(batch_size * n, 1))

        value_head = self.value_head(hidden_state)
        h = hidden_state.view(batch_size, n, self.hid_size)

        action_mean = self.action_mean(h)
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        return action_mean, action_log_std, action_std, value_head

    def init_weights(self, m):
        if type(m) == nn.Linear:
            m.weight.data.normal_(0, self.init_std)

    def init_hidden(self, batch_size):
        # dim 0 = num of layers * num of direction
        return tuple((torch.zeros(batch_size * self.nagents, self.hid_size, requires_grad=True),
                      torch.zeros(batch_size * self.nagents, self.hid_size, requires_grad=True)))

    def evaluate_actions(self, obs, action, masks, available_actions=None):
        """
        Compute action log probabilities and entropy for given actions
        """
        action_mean, action_log_std, action_std, values = self.forward(
            obs, masks, available_actions
        )
        action = check(action).to(**self.tpdv)

        # Create normal distribution
        dist = torch.distributions.Normal(action_mean, action_std)
        
        # Compute log probabilities and entropy
        action_log_probs = dist.log_prob(action.view(16000, self.nagents, 2))
        dist_entropy = dist.entropy().mean()

        return values, action_log_probs.view(16000, self.nagents, 2), dist_entropy

    def act(self, obs, masks, available_actions=None, deterministic=False):
        """
        Compute actions for given inputs
        """
        action_mean, action_log_std, action_std, value_head = self.forward(
            obs, masks, available_actions
        )

        if deterministic:
            actions = action_mean
        else:
            # Sample from normal distribution
            dist = torch.distributions.Normal(action_mean, action_std)
            actions = dist.sample()

        # Compute log probabilities
        dist = torch.distributions.Normal(action_mean, action_std)
        action_log_probs = dist.log_prob(actions)

        return actions, action_log_probs

    def get_probs(self, obs, masks, available_actions=None):
        """
        Compute action probabilities for given inputs
        """
        action_mean, action_log_std, action_std, value_head = self.forward(
            obs, masks, available_actions
        )

        # For continuous actions, return the parameters of the distribution
        return action_mean, action_std

