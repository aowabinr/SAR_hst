# Adversarial Search and Rescue via Multi-AgentReinforcement Learning
<table align="center">
<tr>
<th>Proposed Algorithm</th>
<th>Naive Algorithm</th>
</tr>
<tr>
<td><img src="./gifs/case_iv.gif" width="240px"></td>
<td><img src="./gifs/case_iii.gif" width="240px"></td>
</tr>
</table>

## Abstract
Search and Rescue (SAR) missions in remote environments often employ autonomous multi-robot systems that learn, plan, and execute a combination of local single-robot control action collaboratively. Often, SAR coordination strategies are manually designed by human experts who can remotely control the multi-robot system and enable semi-autonomous operations in challenging scenarios. However, in remote environments where connectivity is limited and human intervention is not possible, decentralized collaboration strategies that utilize communication between agents and information sharing are needed for fully-autonomous operations. In this context, the objective of the multi-robot team is to discover targets strategically in an obstacle-strewn environment by minimizing the average time needed to find  the targets. Even deccentralized coordination strategies may encounter adversarial situations that are subject to sensor corruption or manipulation of communication data without being detected by the agents themselves. This research project, we develop an approach for modeling adversarial scenarios where one or more agents have been compromised. We also develop an algorithmic approach such that the cooperative team can mitigate the adversarial presence and find the locate the targets.

## Team Members

Aowabin Rahman,  Arnab Bhattacharya, Thiagarajan Ramachandran, Sayak Mukherjee,  Himanshu Sharma, Ted Fujimoto, Samrat Chatterjee

## Folder Structure
~~~
.
├── envs   # contains maps and python scripts pertaining to the environment
  ├──magw  # maps and scripts related to the gridworld environment
    ├──maps  # .txt files for the map environment (nomenclature: map{$map_id}_{$num_object}_multi.txt)
    ├──multiagent_env.py  # contains the environment and agent classes (from Iqbal and Sha)
    ├──load_env.py  # wrapper for environment, visualizations, new reward based on novel states
    ├──comms.py  # communication between the agents (under development)
├── utils         # network architectures, wrappers for environment parallelization
  ├── agents.py # Agent class and initialize agent policies
  ├── buffer.py # Methods for configuring replay buffer
  ├── critics.py # Methods for centralized critic
  ├── env.wrapper # Functions for running environment in parallel envs
  ├── misc.py # Functions for gradient descent, activation functions etc.
  ├── networks.py # Classes containing different ML architectures
  ├── policies.py # Classes containing the individual policies
├── models # model checkpoints saved here
├── timer.py # Main code for running the experiments
~~~

## Requirements
Conda environment specification is located in `environment.yml`. Use this file to manually install dependencies if desired.
Otherwise, follow instructions in the next section.

## Installation
Install conda environment with all dependencies
```shell
conda env create -f environment.yml
```

Activate environment
```shell
conda activate multi-explore
```

A few additional notes:
- You may have issues installing initially some of the libraries under "pip" using "conda env create -f environment.yml". If this happens, I recommend activating the conda environment and then manually using pip install ...
- To run GridWorld environment only in Baselines, you probably don't need MuJoCo and other libraries that the authors used for VizDoom environments. For me, I had issues installing these libraries in Marianas. To get around this problem,  git clone baselines only (i.e. "git clone https://github.com/openai/baselines.git") and install the baselines package only. 

## Methods
### Environment
We used a 2D Gridworld environment for the search and rescue problem. The black cells correspond to obstacles that the agents can not occupy.  
You can customize your own environment by editing any of the current environments, or adding a new environemnt in ./envs/magw/maps. Follow the nomenclature for creating the .txt file: map{$map_id}_{num_target}_multi.txt. If you'd like to customize the environment further, please do so in the following directory: ./envs/magw/maps. To edit the environment,  you can change ./envs/magw/maps/multiagent_env.py. 
To change the novelty-based reward function, edit the "compute_final_reward" function in load_env.py

<img src="./gifs/init.png" width="360px">

### State Space and Action Space
We examine two state spaces to deal with possible disaster areas and scenarios. Action space consists of four possible directions, A={up,down,right,left}.

<!-- All training code is contained within `main.py`. To view options simply run:

```shell
python main.py --help
```

## Citing our work

If you use this repo in your work, please consider citing the corresponding paper:

```bibtex
@article{iqbal2019coordinated,
  title={Coordinated Exploration via Intrinsic Rewards for Multi-Agent Reinforcement Learning},
  author={Iqbal, Shariq and Sha, Fei},
  journal={arXiv preprint arXiv:1905.12127},
  year={2019}
}
```


## Usage

- Check any of the expt_*.sh files. Based on the experimental specs, you can change the parameters and then do "sbatch expt_x.sh" (x is the expt id)
