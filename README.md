# Adversarial Search and Rescue via Multi-AgentReinforcement Learning
<table align="center">
<tr>
<th>Proposed Algorithm</th>
<th>Naive Algorithm</th>
</tr>
<tr>
<td><img src="./figs/case_iv.gif" width="240px"></td>
<td><img src="./figs/case_iii.gif" width="240px"></td>
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


<!-- Nevertheless, decentralized coordination may be ineffective in adversarial environments due to sensor noise, actuation faults, or even manipulation of the communication data without agents' knowledge. In this paper, we propose a new algorithmic approach based on adversarial multi-agent reinforcement learning (MARL) that allows robots to efficiently coordinate their strategies in the presence of adversarial communications. The objective of the multi-robot team is to discover targets strategically in an obstacle-strewn environment by minimizing the average time needed to find  the targets. In our setup, the robots have no prior knowledge of the target locations, and they can interact with only a subset of neighboring robots at any time due to communication constraints. Based on the centralized training with decentralized execution (CTDE) paradigm in MARL, we utilize a hierarchical meta-learning framework to learn dynamic team-coordination modalities and discover emergent team behavior under complex cooperative-competitive scenarios. The effectiveness of our approach is compared to other state-of-the-art adversarial MARL algorithms is demonstrated on a collection of Grid-world environments with different specifications  of  benign and adversarial agents, target locations and agent rewards. -->

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
<img src="./gifs/init.png" width="360px">

### State Space and Action Space
We examine two state spaces to deal with possible disaster areas and scenarios. Action space consists of four possible directions, A={up,down,right,left}.

<!-- All training code is contained within `main.py`. To view options simply run:

```shell
python main.py --help
```

All hyperparameters can be found in the Appendix of the paper. Default hyperparameters are for Task 1 in the GridWorld environment using 2 agents.
For Flip-Task include the flags `--task_config 4 --map_ind -1`. -->

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
