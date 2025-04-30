import torch
import torch.nn as nn
from onpolicy.algorithms.r_mappo.algorithm.gat_comm import GAT_Comm
from onpolicy.algorithms.r_mappo.algorithm.scheduler import Scheduler
from onpolicy.algorithms.utils.util import init, check
from onpolicy.algorithms.utils.cnn import CNNBase
from onpolicy.algorithms.utils.mlp import MLPBase
from onpolicy.algorithms.utils.rnn import RNNLayer
from onpolicy.algorithms.utils.act import ACTLayer
from onpolicy.algorithms.utils.popart import PopArt
from onpolicy.utils.util import get_shape_from_obs_space
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
import scipy.linalg as linalg


class MAGIC_Actor(nn.Module):
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(MAGIC_Actor, self).__init__()
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
        self.obs_info_scheduler = args.obs_info_scheduler
        self.mask_threshold = args.mask_threshold

        obs_shape = get_shape_from_obs_space(obs_space)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        base = CNNBase if len(obs_shape) == 3 else MLPBase
        self.base = base(args, obs_shape)

        self.scheduler = Scheduler(args, self.hidden_size, self.hidden_size, self.sch_head, self.sch_slope)
        self.comm = GAT_Comm(self.hidden_size, self.hidden_size, args.gat_hidden_size, args.gat_num_heads)
        self.act = ACTLayer(action_space, self.hidden_size + self.hidden_size, self._use_orthogonal, self._gain)

        self.to(device)

    def forward(self, obs, rnn_states, masks, comm_graphs, available_actions=None, deterministic=False):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features = self.base(obs)
        actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)
        sch_input = actor_features.clone()
        
        graphs = self.scheduler(actor_features)
        actor_features = self.comm(actor_features.view(-1, self.num_agents, self.hidden_size), graphs).view(-1, self.hidden_size)
        comm_info = actor_features.clone()

        actor_features = torch.cat((actor_features, sch_input), 1)

        actions, action_log_probs = self.act(actor_features, available_actions, deterministic)

        return actions, action_log_probs, rnn_states, comm_info, graphs

    def evaluate_actions(self, obs, rnn_states, action, masks, comm_graphs, available_actions=None, active_masks=None):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        actor_features = self.base(obs)
        actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)
        sch_input = actor_features.clone()

        graphs = self.scheduler(sch_input)
        actor_features = self.comm(actor_features.view(-1, self.num_agents, self.hidden_size), graphs).view(-1, self.hidden_size)
        comm_info = actor_features.clone()

        actor_features = torch.cat((actor_features, sch_input), 1)
        action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features,
                                                                    action, available_actions,
                                                                    active_masks=
                                                                    active_masks if self._use_policy_active_masks
                                                                    else None)

        return action_log_probs, dist_entropy, comm_info, graphs    


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


class MAGIC_Critic(nn.Module):
    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        super(MAGIC_Critic, self).__init__()
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

        self.scheduler = Scheduler(args, self.hidden_size, self.hidden_size, self.sch_head, self.sch_slope)
        self.comm = GAT_Comm(self.hidden_size, self.hidden_size, args.gat_hidden_size, args.gat_num_heads)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        if self._use_popart:
            self.v_out = init_(PopArt(self.hidden_size + self.hidden_size, 1, device=device))
        else:
            self.v_out = init_(nn.Linear(self.hidden_size + self.hidden_size, 1))

        self.to(device)

    def forward(self, cent_obs, rnn_states, masks, comm_graphs):
        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        critic_features = self.base(cent_obs)  
        critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)
        sch_input = critic_features.clone()

        graphs = self.scheduler(sch_input)

        critic_features = self.comm(critic_features.view(-1, self.num_agents, self.hidden_size), graphs).view(-1, self.hidden_size)

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
