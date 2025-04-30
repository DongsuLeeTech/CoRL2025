#!/bin/sh
env="robotarium"
scenario="MaterialTransport"
exp="robo_simple"
algo="CommFormer"
seed_start=0
seed_max=0
episode_length=70

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in $(seq ${seed_start} ${seed_max});  # Modified to start from seed_start;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=1 python ../train/train_robotarium_CommFormer.py --env_name ${env} --algorithm_name ${algo} \
    --scenario_name ${scenario} --seed ${seed} --episode_length ${episode_length} \
    --n_training_threads 1 --n_rollout_threads 32 --num_mini_batch 16  --num_env_steps 10000000 \
    --ppo_epoch 5 --log_interval 10 --use_eval --eval_interval 10 --n_eval_rollout_threads 1 \
    --lr 5e-4 --critic_lr 5e-4 --hidden_size 128 \
    --comm_type 1 --num_comm_hops 4 --comm_hidden_size 128 --comm_num_heads 4 \
    --n_block 1 --n_head 4 --share_actor --pos_embed --obs_pos_embed_start 1 --obs_pos_embed_end 2 \
    --use_bilevel \
    --wandb_project_name unknown --user_name unknown
done