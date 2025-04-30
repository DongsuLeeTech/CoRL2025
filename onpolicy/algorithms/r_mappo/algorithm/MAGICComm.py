import torch
from onpolicy.algorithms.r_mappo.algorithm.MAGICNet import MAGIC_Actor, MAGIC_Critic
from onpolicy.utils.util import update_linear_schedule


class MAGICComm:
    def __init__(self, args, obs_space, cent_obs_space, act_space, device=torch.device("cpu")):
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay
        self.num_agents = args.num_agents
        self.max_comm_graph_batch_size = args.max_comm_graph_batch_size
        self.r_length = args.data_chunk_length

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space

        self.actor = MAGIC_Actor(args, self.obs_space, self.act_space, self.device)
        self.critic = MAGIC_Critic(args, self.share_obs_space, self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.critic_lr,
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)

    def lr_decay(self, episode, episodes):
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, available_actions=None, comm_graphs=None,
                    deterministic=False):
        actions, action_log_probs, rnn_states_actor, actor_comm, graphs = self.actor(obs,
                                                                    rnn_states_actor,
                                                                    masks,
                                                                    comm_graphs,
                                                                    available_actions,
                                                                    deterministic)

        values, rnn_states_critic = self.critic(cent_obs, rnn_states_critic, masks, comm_graphs)
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic, actor_comm, graphs

    def get_values(self, cent_obs, rnn_states_critic, masks, comm_graphs=None):
        values, _ = self.critic(cent_obs, rnn_states_critic, masks, comm_graphs)
        return values

    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, action, masks,
                         available_actions=None, active_masks=None, comm_graphs=None):
        action_log_probs, dist_entropy, actor_comm, graphs = self.actor.evaluate_actions(obs,
                                                                        rnn_states_actor,
                                                                        action,
                                                                        masks,
                                                                        comm_graphs,
                                                                        available_actions,
                                                                        active_masks)

        values, _ = self.critic(cent_obs, rnn_states_critic, masks, comm_graphs)
        return values, action_log_probs, dist_entropy, actor_comm, graphs

    def act(self, obs, rnn_states_actor, masks, available_actions=None, comm_graphs=None, deterministic=False):
        actions, _, rnn_states_actor, comm_info, graphs = \
            self.actor(obs, rnn_states_actor, masks, comm_graphs, available_actions, deterministic)
        return actions, rnn_states_actor, comm_info, graphs

    def train(self):
        self.actor.train()
        self.critic.train()

    def eval(self):
        self.actor.eval()
        self.critic.eval()
