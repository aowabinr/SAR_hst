#!/bin/bash
#SBATCH -A optimas
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH -p dl
#source /etc/profile.d/modules.sh
module unload python
module load python/miniconda3.9
source /share/apps/python/miniconda3.9/etc/profile.d/conda.sh
conda activate multi-explore_tut


#python timer_inf.py inf_e4b_352a --map_ind 352 --num_agents 3 --train_time 1600000 --rogue_agents 3 --output_dir inf_e4b_352a --rogue_reward_factor 1.0 --random_target 0 --model_path ./saved_models/model_timer_e4b3/model.pt --load_model 1 --inference 1 --max_episode_length 400000 --img_interval 3000
#python timer_inf.py inf_e4b_352a --map_ind 352 --num_agents 3 --train_time 1600000 --rogue_agents 3 --output_dir inf_e4b_352a --rogue_reward_factor 1.0 --random_target 0 --model_path ./saved_models/model_timer_e4b5/model.pt --load_model 1 --inference 1 --max_episode_length 400000 --img_interval 3000
python timer_inf.py inf_E0B_356 --map_ind 356 --num_agents 3 --train_time 1600000 --rogue_agents 3 --output_dir /qfs/projects/optimas/search_and_rescue/results/inf_E0B_356b --rogue_reward_factor 1.0 --random_target 0 --model_path /qfs/projects/optimas/search_and_rescue/saved_models/model_e0/model.pt --adv_path /qfs/projects/optimas/search_and_rescue/saved_models/model_timer_e4b5/model.pt --load_model 1 --inference 1 --max_episode_length 400000 --img_interval 300
