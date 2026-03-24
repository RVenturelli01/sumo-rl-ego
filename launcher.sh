
taskset -c 12-23 python experiments/finetune.py \
    source.model_path=/work/fis1/sumo-human-feedback-rl/sumo-rl-ego/outputs/2026-03-24_11-41-45_dqn_v1_ft/model.zip \
    source.replay_buffer_path=/work/fis1/sumo-human-feedback-rl/sumo-rl-ego/outputs/2026-03-24_11-41-45_dqn_v1_ft/replay_buffer.pkl \

# taskset -c 12-23 python experiments/train.py

