import time
import numpy as np
import torch
from onpolicy.runner.shared.base_runner import Runner
from onpolicy.utils.video_util import *
from collections import defaultdict
import wandb
import imageio
import os
import pygame
import pickle
import uuid
import psutil
os.environ["SDL_VIDEODRIVER"] = "dummy"

def _t2n(x):
    return x.detach().cpu().numpy()

class MetaDriveRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for the MetaDrive. See parent class for details."""
    def __init__(self, config):
        super(MetaDriveRunner, self).__init__(config)
        self.env_infos = defaultdict(list)
        self.my_uuid = uuid.uuid4()
        self.best_reward = 0

    def run(self):
        self.restore()
        self.eval()

    def warmup(self):
        # reset env
        obs = self.envs.reset()
        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()

    @torch.no_grad()
    def eval(self):
        all_frames = []
        all_3d_frames = []
        graphs = []

        # eval_obs, image = self.eval_envs.reset()
        eval_obs = self.eval_envs.reset()
        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
        eval_comm_graphs = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.num_agents), dtype=np.float32)
        eval_episode_steps = []

        for eval_step in range(self.episode_length):
            self.trainer.prep_rollout()
            # if isinstance(image, dict):
            #     img = image['agent1'][..., -2]*255
            #
            # else:
            #     img = image[0]['agent1'][..., -2]*255

            image = self.eval_envs.render(mode="rgb_array")
            image = pygame.surfarray.array3d(image[0]).astype(np.uint8)
            image = np.transpose(image, (1, 0, 2))
            all_3d_frames.append(image)

            # img = np.expand_dims(img, axis=0)
            # all_3d_frames.append(img)
            #
            # if len(eval_obs.shape) == 3:
            #     eval_obs = eval_obs.squeeze(0)

            eval_obs = np.concatenate(eval_obs)
            if self.eval_bit:
                eval_action, eval_rnn_states, comm_info, att, att_mask, graph = self.trainer.policy.act(eval_obs,
                                                np.concatenate(eval_rnn_states),
                                                np.concatenate(eval_masks),
                                                comm_graphs = eval_comm_graphs,
                                                eval_bit=self.eval_bit,
                                                deterministic=True)
            else:
                eval_action, eval_rnn_states, comm_info, att, att_mask, graph = self.trainer.policy.act(eval_obs,
                                                np.concatenate(eval_rnn_states),
                                                np.concatenate(eval_masks),
                                                comm_graphs = eval_comm_graphs,
                                                deterministic=True)

            graphs.append(graph.detach().cpu().numpy())
            eval_actions = np.array(np.split(_t2n(eval_action), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))
            
            if self.eval_envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
                for i in range(self.eval_envs.action_space[0].shape):
                    eval_uc_actions_env = np.eye(self.eval_envs.action_space[0].high[i]+1)[eval_actions[:, :, i]]
                    if i == 0:
                        eval_actions_env = eval_uc_actions_env
                    else:
                        eval_actions_env = np.concatenate((eval_actions_env, eval_uc_actions_env), axis=2)
            elif self.eval_envs.action_space[0].__class__.__name__ == 'Discrete':
                eval_actions_env = np.squeeze(np.eye(self.eval_envs.action_space[0].n)[eval_actions], 2)
            elif self.envs.action_space[0].__class__.__name__ == 'Box':
                eval_actions_env = eval_actions
            else:
                raise NotImplementedError

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            # eval_obs, eval_rewards, eval_dones, eval_infos, image = self.eval_envs.step(eval_actions_env)
            eval_dones_env = np.all(eval_dones, axis=-1)

            if not self.all_args.meta_allow_respawn:
                for i in range(eval_dones_env.shape[0]):
                    if eval_dones_env[i]:
                        for k, v in eval_infos[i].items():
                            if 'agent' in k:
                                eval_episode_steps.append(eval_infos[i][k]['episode_length'])
                                break
            else:
                if eval_step == self.episode_length - 1:
                    for i in range(eval_dones_env.shape[0]):
                        eval_episode_steps.append(self.episode_length)

            eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)
            
            eval_comm_graphs = [eval_infos[i]["comm_graph"] for i in range(self.n_eval_rollout_threads)]
            eval_comm_graphs = np.stack(eval_comm_graphs)

        graphs = np.array(graphs)

        graph_viz = save_gif_with_graph(graphs, slicin_idx=1, filename=f'./evaluate/graph_{self.my_uuid}.gif')
        video = record_video('Video', renders=all_3d_frames, filename=f'./evaluate/video_{self.my_uuid}.gif')
        # renders = np.concatenate(all_3d_frames, axis=0)  # (T, H, W, C)로 변환
        #
        # # RGB 채널이 맞지 않으면 변환
        # if renders.shape[-1] != 3:
        #     renders = renders[..., :3]  # RGB 첫 3채널만 사용
        # #
        # record_video(label, renders=[renders], filename=filename)
        # video_3d = record_video('Video', renders=[renders], skip_frames=1, filename=f'./evaluate/video_{self.my_uuid}.gif')
        # % filename=f'./evaluate/3d_{self.my_uuid}.gif'


    @torch.no_grad()
    def render(self):
        """Visualize the env."""
        envs = self.envs
        
        all_frames = []
        all_rewards = []
        all_3d_frames = []
        for episode in range(self.all_args.render_episodes):
            obs = envs.reset()
            if self.all_args.save_gifs:
                image = self.envs.envs[0].env.render(mode="top_down", track_target_vehicle=False)
                image = pygame.surfarray.array3d(image).astype(np.uint8)
                all_frames.append(image)

                image_3d = self.envs.envs[0].env.render()
                image_3d = pygame.surfarray.array3d(image).astype(np.uint8)
                all_3d_frames.append(image_3d)
            else:
                envs.render('human')

            rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
            comm_graphs = np.zeros((self.n_rollout_threads, self.num_agents, self.num_agents), dtype=np.float32)
            
            episode_rewards = []
            success_rates = []
            
            for step in range(self.episode_length):
                calc_start = time.time()

                self.trainer.prep_rollout()
                action, rnn_states = self.trainer.policy.act(np.concatenate(obs),
                                                    np.concatenate(rnn_states),
                                                    np.concatenate(masks),
                                                    comm_graphs = comm_graphs,
                                                    deterministic=True)
           
                actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
                rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
                actions_env = actions

                # Obser reward and next obs
                obs, rewards, dones, infos = envs.step(actions_env)
                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
                
                comm_graphs = [infos[i]["comm_graph"] for i in range(self.n_rollout_threads)]
                comm_graphs = np.stack(comm_graphs)

                if self.all_args.save_gifs:
                    image = self.envs.envs[0].env.render(mode="top_down", track_target_vehicle=False)
                    image = pygame.surfarray.array3d(image).astype(np.uint8)
                    all_frames.append(image)
                else:
                    envs.render('human')
                
                if not self.all_args.meta_allow_respawn:
                    if np.all(dones[0]):
                        success_rates.append(infos[0]['success_rate'])
                        break
                    elif step == self.episode_length - 1:
                        success_rates.append(infos[0]['success_rate'])
                else:
                    if step == self.episode_length - 1:
                        success_rates.append(infos[0]['success_rate'])
            
            all_rewards.append(np.array(episode_rewards))
            
        if self.all_args.save_gifs:
            self.gif_dir = './'
            imageio.mimsave(str(self.gif_dir) + '/render.gif', all_frames, duration=self.all_args.ifi)

        return all_frames


    def restore(self):
        """Restore policy's networks from a saved model."""
        self.policy.actor = torch.load('/home/dongsu/CommMARL/onpolicy/scripts/eval_metadrive_scripts/actor_cau.pt')
