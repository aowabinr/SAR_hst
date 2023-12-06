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

#python timer.py model_E0 --map_ind 350 --num_agents 3 --num_objects 2 --train_time 600000 --output_dir E0 --rogue_reward_factor 1.0 --random_target 0  --max_episode_length 200 --explore_mode 1 --img_interval 300 
python timer.py model_E0 --map_ind 350 --num_agents 3 --num_objects 2 --train_time 600000 --output_dir ./results/E0_load --rogue_reward_factor 1.0 --random_target 0  --model_path /qfs/projects/optimas/search_and_rescue/saved_models/model_e0/model.pt --load_model 1 --max_episode_length 200 --explore_mode 1 --img_interval 300 

