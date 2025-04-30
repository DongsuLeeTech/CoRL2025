#!/bin/sh
env="MetaDrive"
scenario="myparkinglot"
num_agents=5
algo="TarMAC"
seed_start=1
seed_max=3
episode_length=1000

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in $(seq ${seed_start} ${seed_max});  # Modified to start from seed_start;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=1 python ../train/train_TarMAC.py --env_name ${env} --algorithm_name ${algo} \
    --scenario_name ${scenario} --num_agents ${num_agents} --seed ${seed} --episode_length ${episode_length} \
    --n_training_threads 1 --n_rollout_threads 16 --num_mini_batch 1  --num_env_steps 10000000 \
    --ppo_epoch 15 --log_interval 10 --use_eval --eval_interval 10 --n_eval_rollout_threads 1 \
    --lr 3e-4 --critic_lr 3e-4 --hidden_size 128 \
    --meta_lidar_num_lasers 2 --meta_lidar_dist 10 --meta_lidar_num_others 4 \
    --meta_reward_coeff 1. --meta_global_pos --meta_navi_pos \
    --meta_comm_range 25 --meta_comm_max_num 4 \
    --wandb_project_name unknown --user_name unknown
done