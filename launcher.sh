

python experiments/eval.py \
    run.output_dir=/home/ricca/projects/sumo-human-feedback-rl/sumo-rl-ego/outputs \
    source.policy_id=FastPolicy-v0 \
    env.kwargs.ego=discrete \
    run.seed=0 \
    wandb.enabled=true \
    run.n_episodes=100 \
    run.name=eval_FastPolicy-v0_n100
    

