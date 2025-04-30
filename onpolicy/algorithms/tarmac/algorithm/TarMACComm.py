import torch
from onpolicy.algorithms.tarmac.algorithm.TarMACNet import TarCommNetMLP
from onpolicy.utils.util import update_linear_schedule


class TarMACComm:
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

        self.agent = TarCommNetMLP(args, self.obs_space, act_space, device)

        self.optimizer = torch.optim.RMSprop(self.agent.parameters(),
                                       lr=args.lr, alpha=0.97, eps=1e-6)

    def lr_decay(self, episode, episodes):
        update_linear_schedule(self.optimizer, episode, episodes, self.lr)

    def get_actions(self, obs, masks):
        action_mean, action_log_std, action_std, values = self.agent(obs,
                                                                     masks)

        dist = torch.distributions.Normal(action_mean, action_std)
        actions = dist.sample()
        action_log_probs = dist.log_prob(actions)

        return values, actions, action_log_probs

    def get_values(self, obs, masks):
        action_mean, action_log_std, action_std, values = self.agent(obs, masks)
        return values

    def evaluate_actions(self, obs, action, masks):
        values, action_log_probs, dist_entropy = self.agent.evaluate_actions(obs,
                                                                        action,
                                                                        masks)

        return values, action_log_probs, dist_entropy

    def act(self, obs, masks, deterministic=False):
        actions, action_log_probs = self.agent.act(obs, masks, deterministic)
        return actions

    def train(self):
        self.agent.train()

    def eval(self):
        self.agent.eval()

