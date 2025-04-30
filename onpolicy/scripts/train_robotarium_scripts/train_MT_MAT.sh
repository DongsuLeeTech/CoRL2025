#!/bin/sh
env="robotarium"
scenario="MaterialTransport"
algo="mat" #"mappo" "ippo" "rmappo_comm"
exp="robo_simple"
# exp="test"
seed_max=3
episode_length=70

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=1 python ../train/train_robotarium_MAT.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --scenario_name ${scenario} --seed ${seed} --episode_length ${episode_length} \
    --n_training_threads 1 --n_rollout_threads 32 --num_mini_batch 16  --num_env_steps 1000000 \
    --ppo_epoch 5 --log_interval 1 --use_eval --eval_interval 10 --n_eval_rollout_threads 1 \
    --lr 5e-4 --critic_lr 5e-4 --hidden_size 128 \
    --n_block 4 --n_head 4 --share_actor --pos_embed --obs_pos_embed_start 0 --obs_pos_embed_end 3 \
    --wandb_project_name unknown --user_name unknown
done

