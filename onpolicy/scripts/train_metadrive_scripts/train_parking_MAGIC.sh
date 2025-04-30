#!/bin/sh
env="MetaDrive"
scenario="myparkinglot"
num_agents=5
algo="MAGIC"
seed_start=0
seed_max=3
episode_length=1000

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in $(seq ${seed_start} ${seed_max});  # Modified to start from seed_start;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python ../train/train_MAGIC.py --env_name ${env} --algorithm_name ${algo} \
    --scenario_name ${scenario} --num_agents ${num_agents} --seed ${seed} --episode_length ${episode_length} \
    --n_training_threads 1 --n_rollout_threads 16 --num_mini_batch 1  --num_env_steps 10000000 \
    --ppo_epoch 15 --log_interval 10 --use_eval --eval_interval 10 --n_eval_rollout_threads 1 \
    --lr 5e-4 --critic_lr 5e-4 --hidden_size 128 \
    --meta_lidar_num_lasers 2 --meta_lidar_dist 10 --meta_lidar_num_others 2 \
    --meta_reward_coeff 1. --meta_global_pos --meta_navi_pos \
    --meta_comm_range 25 --meta_comm_max_num 4 \
    --num_comm_hops 4 --comm_hidden_size 128 --comm_num_heads 4 \
    --pos_embed --obs_pos_embed_start 20 --obs_pos_embed_end 22 \
    --obs_info_scheduler obs --skip_connect_final --fixed_masked --mask_threshold 0.5 --use_scheduler \
    --wandb_project_name unknown --user_name unknown
done
