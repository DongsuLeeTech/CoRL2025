import torch
import torch.nn as nn
from onpolicy.algorithms.r_mappo.algorithm.gat_comm import GAT_Comm
from onpolicy.algorithms.r_mappo.algorithm.transformer_comm import Transformer_Comm
from onpolicy.algorithms.r_mappo.algorithm.scheduler import Scheduler
from onpolicy.algorithms.utils.util import init, check
from onpolicy.algorithms.utils.cnn import CNNBase
from onpolicy.algorithms.utils.mlp import MLPBase
from onpolicy.algorithms.utils.rnn import RNNLayer
from onpolicy.algorithms.utils.act import ACTLayer
from onpolicy.algorithms.utils.popart import PopArt
from onpolicy.utils.util import get_shape_from_obs_space
from onpolicy.algorithms.utils.quantizer import VQVAEQuantizer, NearestEmbed
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
import scipy.linalg as linalg


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, latent_dim)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc2(h1)


class R_Actor_Comm(nn.Module):
    """
    Actor network class for MAPPO. Outputs actions given observations.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param obs_space: (gym.Space) observation space.
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(R_Actor_Comm, self).__init__()
        self.hidden_size = args.hidden_size

        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.num_agents = args.num_agents
        self.use_scheduler = args.use_scheduler
        self.sch_slope = args.negative_slope
        self.sch_head = args.scheduler_head
        self.comm_type = args.comm_type
        self.use_pos_embed = args.pos_embed
        self.obs_pos_embed_end = args.obs_pos_embed_end
        self.obs_pos_embed_start = args.obs_pos_embed_start
        self.obs_info_scheduler = args.obs_info_scheduler
        self.skip_connect_final = args.skip_connect_final
        self.mask_threshold = args.mask_threshold
        self.obs_enc_type = args.obs_enc_type
        self.use_vq_vae = args.use_vq_vae
        self.code_size = args.code_size

        obs_shape = get_shape_from_obs_space(obs_space)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        if self.obs_enc_type == 'attention':
            self.att_obs = SelfAttentionModule(obs_shape[0], num_heads=args.n_obs_head)
            # att_sample = self.att_obs(torch.zeros(1, obs_shape[0], obs_shape[0]))
            sch_shape = obs_shape
            # print(sch_shape)
        else:
            base = CNNBase if len(obs_shape) == 3 else MLPBase
            self.base = base(args, obs_shape)
            if self.obs_info_scheduler == 'obs_enc':
                enc_sample = self.base(torch.zeros(obs_shape)).squeeze(0)
                sch_shape = enc_sample.shape
            elif self.obs_info_scheduler == 'rnn_enc':
                sch_shape = self.hidden_size
            elif self.obs_info_scheduler == 'obs':
                sch_shape = obs_shape

        if self.use_scheduler:
            self.scheduler = Scheduler(args, sch_shape, self.hidden_size, self.sch_head, self.sch_slope)

        if args.comm_type == 0:
            self.comm = GAT_Comm(self.hidden_size, self.hidden_size, args.gat_hidden_size, args.gat_num_heads)
        elif args.comm_type == 1:
            self.comm = Transformer_Comm(self.hidden_size, self.hidden_size, args.comm_hidden_size,
                                         args.num_comm_hops, args.comm_num_heads, self.num_agents,
                                         args.causal_masked, args.fixed_masked, self.mask_threshold, device)
        else:
            self.encode_comm = Transformer_Comm(self.hidden_size, self.hidden_size, args.comm_hidden_size,
                                                args.num_comm_hops, args.comm_num_heads, self.num_agents, device)
            self.comm = Transformer_Comm(self.hidden_size, self.hidden_size, args.comm_hidden_size,
                                         args.num_comm_hops, args.comm_num_heads, self.num_agents, device)

        if self.use_vq_vae:
            self.encoder = Encoder(self.hidden_size, self.hidden_size, self.code_size)
            self.vqvae_quantizer = VQVAEQuantizer(num_embeddings=512, embedding_dim=self.code_size)

        if self.use_pos_embed:
            self.pos_encoder = nn.Linear(args.obs_pos_embed_end - args.obs_pos_embed_start, self.hidden_size)

        if not self.skip_connect_final:
            self.act = ACTLayer(action_space, self.code_size, self._use_orthogonal, self._gain)
        else:
            if type(sch_shape) == int:
                self.act = ACTLayer(action_space, self.code_size + sch_shape, self._use_orthogonal, self._gain)
            else:
                self.act = ACTLayer(action_space, self.code_size + sch_shape[0], self._use_orthogonal, self._gain)

        self.to(device)

    def forward(self, obs, rnn_states, masks, comm_graphs, available_actions=None, deterministic=False):
        """
        Compute actions from the given inputs.
        :param obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        # graph_mask = self.construct_graph_mask(masks)
        masks = check(masks).to(**self.tpdv)
        comm_graphs = check(comm_graphs).to(**self.tpdv)
        # graph_mask: [batch_size, num_agents, num_agents]
        # graph_mask = check(graph_mask).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if self.use_pos_embed:
            pos_embed = self.pos_encoder(obs[:, self.obs_pos_embed_start - 1:self.obs_pos_embed_end - 1])
            # actor_features = actor_features + pos_embed
        else:
            pos_embed = None

        if self.obs_enc_type == 'attention':
            B, O = obs.shape
            actor_features = self.att_obs(obs.unsqueeze(-1).expand(B, O, O)).mean(dim=-1)
            sch_input = actor_features.clone()
        else:
            actor_features = self.base(obs)
            if self.obs_info_scheduler == 'obs_enc':
                sch_input = actor_features.clone()
                actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)
            elif self.obs_info_scheduler == 'rnn_enc':
                actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)
                sch_input = actor_features.clone()
            elif self.obs_info_scheduler == 'obs':
                sch_input = obs.clone()
                actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        if self.use_scheduler:
            graphs = self.scheduler(sch_input) * comm_graphs
        else:
            graphs = comm_graphs

        # if self.comm_type == 2:
        #     actor_features = self.encode_comm(actor_features.view(-1, self.num_agents, self.hidden_size), graphs, pos_embed.view(-1, self.num_agents, self.hidden_size)).view(-1, self.hidden_size)
        #
        # if self._use_naive_recurrent_policy or self._use_recurrent_policy:
        #     actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        if self.comm_type == 1:
            actor_features, att, att_mask = self.comm(actor_features.view(-1, self.num_agents, self.hidden_size),
                                                      graphs, pos_embed.view(-1, self.num_agents, self.hidden_size))
            actor_features = actor_features.view(-1, self.hidden_size)
            comm_info = actor_features.clone()
        elif self.comm_type == 0:
            actor_features = self.comm(actor_features.view(-1, self.num_agents, self.hidden_size), graphs).view(-1,
                                                                                                                self.hidden_size)

        if self.use_vq_vae:
            actor_features = self.encoder(actor_features)
            actor_features, vq_loss, _ = self.vqvae_quantizer(actor_features)

        if self.skip_connect_final:
            actor_features = torch.cat((actor_features, sch_input), 1)

        actions, action_log_probs = self.act(actor_features, available_actions, deterministic)

        return actions, action_log_probs, rnn_states, comm_info, att, att_mask, graphs, vq_loss

    def evaluate_actions(self, obs, rnn_states, action, masks, comm_graphs, available_actions=None, active_masks=None):
        """
        Compute log probability and entropy of given actions.
        :param obs: (torch.Tensor) observation inputs into network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param rnn_states: (torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)

        # graph_mask = self.construct_graph_mask(masks)
        masks = check(masks).to(**self.tpdv)
        comm_graphs = check(comm_graphs).to(**self.tpdv)
        # graph: [batch_size, num_agents, num_agents]
        # graph_mask = check(graph_mask).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        if self.use_pos_embed:
            pos_embed = self.pos_encoder(obs[:, self.obs_pos_embed_start - 1:self.obs_pos_embed_end - 1])
            # actor_features = actor_features + pos_embed
        else:
            pos_embed = None

        if self.obs_enc_type == 'attention':
            B, O = obs.shape
            actor_features = self.att_obs(obs.unsqueeze(-1).expand(B, O, O)).mean(dim=-1)
            sch_input = actor_features.clone()
        else:
            actor_features = self.base(obs)
            if self.obs_info_scheduler == 'obs_enc':
                sch_input = actor_features.clone()
                actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)
            elif self.obs_info_scheduler == 'rnn_enc':
                actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)
                sch_input = actor_features.clone()
            elif self.obs_info_scheduler == 'obs':
                sch_input = obs.clone()
                actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        if self.use_scheduler:
            graphs = self.scheduler(sch_input) * comm_graphs
        else:
            graphs = comm_graphs

        if self.comm_type == 1:
            actor_features, att, att_mask = self.comm(actor_features.view(-1, self.num_agents, self.hidden_size),
                                                      graphs,
                                                      pos_embed.view(-1, self.num_agents, self.hidden_size))
            actor_features = actor_features.view(-1, self.hidden_size)
            comm = actor_features.clone()

            if self.use_vq_vae:
                actor_features = self.encoder(actor_features)
                actor_features, vq_loss, _ = self.vqvae_quantizer(actor_features)

            if self.skip_connect_final:
                actor_features = torch.cat((actor_features, sch_input), 1)

            action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features,
                                                                       action, available_actions,
                                                                       active_masks=
                                                                       active_masks if self._use_policy_active_masks
                                                                       else None)

            return action_log_probs, dist_entropy, comm, att, att_mask, graphs, vq_loss

        elif self.comm_type == 0:
            actor_features = self.comm(actor_features.view(-1, self.num_agents, self.hidden_size), graphs).view(-1,
                                                                                                                self.hidden_size)

            if self.skip_connect_final:
                actor_features = torch.cat((actor_features, sch_input), 1)

            action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features,
                                                                       action, available_actions,
                                                                       active_masks=
                                                                       active_masks if self._use_policy_active_masks
                                                                       else None)

            return action_log_probs, dist_entropy


    def construct_graph_mask(self, masks):
        # masks: [batch_size * num_agents, 1]
        mask0 = masks.reshape(-1, self.num_agents, 1)
        mask1 = np.transpose(mask0, (0, 2, 1))
        mask = np.ones((mask0.shape[0], self.num_agents, self.num_agents))
        mask = mask * mask0 * mask1
        # subgraph: [num_agents, num_agents]
        subgraphs = [mask[i] for i in range(mask.shape[0])]
        # graph: [batch_size, num_agents, num_agents]
        graph_mask = np.stack(subgraphs)

        return graph_mask


class R_Critic_Comm(nn.Module):
    """
    Critic network class for MAPPO. Outputs value function predictions given centralized input (MAPPO) or
                            local observations (IPPO).
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param cent_obs_space: (gym.Space) (centralized) observation space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        super(R_Critic_Comm, self).__init__()
        self.hidden_size = args.hidden_size
        self.use_scheduler = args.use_scheduler
        self._use_orthogonal = args.use_orthogonal
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self._use_popart = args.use_popart
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.num_agents = args.num_agents
        self.comm_type = args.comm_type
        self.sch_slope = args.negative_slope
        self.sch_head = args.scheduler_head
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]
        self.use_pos_embed = args.pos_embed
        self.obs_pos_embed_end = args.obs_pos_embed_end
        self.obs_pos_embed_start = args.obs_pos_embed_start
        self.obs_info_scheduler = args.obs_info_scheduler
        self.skip_connect_final = args.skip_connect_final
        self.mask_threshold = args.mask_threshold
        self.obs_enc_type = args.obs_enc_type
        self.use_vq_vae = args.use_vq_vae
        self.code_size = args.code_size

        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        base = CNNBase if len(cent_obs_shape) == 3 else MLPBase
        self.base = base(args, cent_obs_shape)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        if self.obs_enc_type == 'attention':
            self.att_obs = SelfAttentionModule(cent_obs_shape[0], num_heads=args.n_obs_head)
            # att_sample = self.att_obs(torch.zeros(1, cent_obs_shape[0], cent_obs_shape[0]))
        else:
            if self.obs_info_scheduler == 'obs_enc':
                enc_sample = self.base(torch.zeros(cent_obs_shape))
                cent_obs_shape = enc_sample.shape
            elif self.obs_info_scheduler == 'rnn_enc':
                cent_obs_shape = self.hidden_size

        if self.use_scheduler:
            self.scheduler = Scheduler(args, cent_obs_shape, self.hidden_size, self.sch_head, self.sch_slope)

        if args.comm_type == 0:
            self.comm = GAT_Comm(self.hidden_size, self.hidden_size, args.gat_hidden_size, args.gat_num_heads)
        else:
            self.comm = Transformer_Comm(self.hidden_size, self.hidden_size, args.comm_hidden_size,
                                         args.num_comm_hops, args.comm_num_heads, self.num_agents,
                                         args.causal_masked, args.fixed_masked, self.mask_threshold, device)

        if self.use_vq_vae:
            self.encoder = Encoder(self.hidden_size, self.hidden_size, self.code_size)
            self.vqvae_quantizer = VQVAEQuantizer(num_embeddings=512, embedding_dim=self.code_size)

        if self.use_pos_embed:
            self.pos_encoder = nn.Linear(args.obs_pos_embed_end - args.obs_pos_embed_start, self.hidden_size)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        if not self.skip_connect_final:
            v_input = self.code_size
        else:
            if type(cent_obs_shape) == int:
                v_input = self.code_size + cent_obs_shape
            else:
                v_input = self.code_size + cent_obs_shape[0]

        if self._use_popart:
            self.v_out = init_(PopArt(v_input, 1, device=device))
        else:
            self.v_out = init_(nn.Linear(v_input, 1))

        self.to(device)

    def forward(self, cent_obs, rnn_states, masks, comm_graphs):
        """
        Compute actions from the given inputs.
        :param cent_obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if RNN states should be reinitialized to zeros.

        :return values: (torch.Tensor) value function predictions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        # graph_mask = self.construct_graph_mask(masks)
        masks = check(masks).to(**self.tpdv)
        comm_graphs = check(comm_graphs).to(**self.tpdv)
        # graph: [batch_size, num_agents, num_agents]
        # graph_mask = check(graph_mask).to(**self.tpdv)

        if self.use_pos_embed:
            pos_embed = self.pos_encoder(cent_obs[:, self.obs_pos_embed_start - 1:self.obs_pos_embed_end - 1])
            # critic_features = critic_features + pos_embed
        else:
            pos_embed = None

        if self.obs_enc_type == 'attention':
            B, O = cent_obs.shape
            critic_features = self.att_obs(cent_obs.unsqueeze(-1).expand(B, O, O)).mean(dim=-1)
            sch_input = critic_features.clone()
        else:
            critic_features = self.base(cent_obs)
            if self.obs_info_scheduler == 'obs_enc':
                sch_input = critic_features.clone()
                critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)
            elif self.obs_info_scheduler == 'rnn_enc':
                critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)
                sch_input = critic_features.clone()
            elif self.obs_info_scheduler == 'obs':
                sch_input = cent_obs.clone()

        if self.use_scheduler:
            graphs = self.scheduler(sch_input) * comm_graphs
        else:
            graphs = comm_graphs

        if self.comm_type == 1:
            critic_features, _, _ = self.comm(critic_features.view(-1, self.num_agents, self.hidden_size), graphs,
                                              pos_embed.view(-1, self.num_agents, self.hidden_size))
            critic_features = critic_features.view(-1, self.hidden_size)

            if self.use_vq_vae:
                critic_features = self.encoder(critic_features)
                critic_features, vq_loss, _ = self.vqvae_quantizer(critic_features)

            if not self.skip_connect_final:
                v_input = critic_features.clone()
            else:
                v_input = torch.cat((critic_features, sch_input), 1)

            values = self.v_out(v_input)

            return values, rnn_states, vq_loss

        if self.comm_type == 0:
            critic_features = self.comm(critic_features.view(-1, self.num_agents, self.hidden_size), graphs).view(-1,
                                                                                                                  self.hidden_size)

            if not self.skip_connect_final:
                v_input = critic_features.clone()
            else:
                v_input = torch.cat((critic_features, sch_input), 1)

            values = self.v_out(v_input)

            return values, rnn_states


    def construct_graph_mask(self, masks):
        # masks: [batch_size * num_agents, 1]
        mask0 = masks.reshape(-1, self.num_agents, 1)
        mask1 = np.transpose(mask0, (0, 2, 1))
        mask = np.ones((mask0.shape[0], self.num_agents, self.num_agents))
        mask = mask * mask0 * mask1
        # subgraph: [num_agents, num_agents]
        subgraphs = [mask[i] for i in range(mask.shape[0])]
        # graph: [batch_size, num_agents, num_agents]
        graph_mask = np.stack(subgraphs)

        return graph_mask
