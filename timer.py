import argparse
import torch
import os
import multiprocessing
import numpy as np
#from vizdoom import ViZDoomErrorException, ViZDoomIsNotRunningException, ViZDoomUnexpectedExitException
from gym.spaces import Box, Discrete
from pathlib import Path
from collections import deque
from tensorboardX import SummaryWriter
from utils.buffer import ReplayBuffer
from utils.env_wrappers import SubprocVecEnv
from utils.misc import apply_to_all_elements, timeout, RunningMeanStd
#from algorithms.sac import SAC

from algorithms.sac_adv import SAC
#from envs.ma_vizdoom.ma_vizdoom import VizdoomMultiAgentEnv
from envs.magw.multiagent_env import GridWorld, VectObsEnv
from envs.magw.load_env import search_and_rescue

from envs.magw.comms import get_count_based_novelties_comm, get_intrinsic_rewards_with_comm

np.random.seed(2)

AGENT_CMAPS = ['Reds', 'Blues', 'Greens', 'Wistia']


def get_count_based_novelties(env, state_inds, device='cpu'):
    """
    Method to compute novelties based on state visits
    :param env:

    """
    env_visit_counts = env.get_visit_counts()

    # samp_visit_counts[i,j,k] is # of times agent j has visited the state that agent k occupies at time i
    samp_visit_counts = np.concatenate(
        [np.concatenate(
            [env_visit_counts[j][tuple(zip(*state_inds[k]))].reshape(-1, 1, 1)
             for j in range(config.num_agents)], axis=1)
         for k in range(config.num_agents)], axis=2)

    # how novel each agent considers all agents observations at every step
    novelties = np.power(np.maximum(samp_visit_counts, 1), -config.decay)
    return torch.tensor(novelties, device=device, dtype=torch.float32)

def get_intrinsic_rewards(novelties, config, intr_rew_rms,
                          update_irrms=False, active_envs=None, device='cpu'):

    """
    Method to compute the intrinsic rewards based on novelties
    :param novelties: (np.array) -  array of size (num_threads, i, j) describing novelty by agent i as perceived by agent j.
    :param config: (dict) - inputs parsed from command line
    :param intr_rew_rms:
    """
    if update_irrms:
        assert active_envs is not None
    intr_rews = []

    for i, exp_type in enumerate(config.explr_types):
        if exp_type == 0:  # independent
            intr_rews.append([novelties[:, ai, ai] for ai in range(config.num_agents)])
        elif exp_type == 1:  # min
            intr_rews.append([novelties[:, :, ai].min(axis=1)[0] for ai in range(config.num_agents)])
        elif exp_type == 2:  # covering
            type_rews = []
            for ai in range(config.num_agents):
                rew = novelties[:, ai, ai] - novelties[:, :, ai].mean(axis=1)
                rew[rew > 0.0] += novelties[rew > 0.0, :, ai].mean(axis=1)
                rew[rew < 0.0] = 0.0
                type_rews.append(rew)
            intr_rews.append(type_rews)
        elif exp_type == 3:  # burrowing
            type_rews = []
            for ai in range(config.num_agents):
                rew = novelties[:, ai, ai] - novelties[:, :, ai].mean(axis=1)
                rew[rew > 0.0] = 0.0
                rew[rew < 0.0] += novelties[rew < 0.0, :, ai].mean(axis=1)
                type_rews.append(rew)
            intr_rews.append(type_rews)
        elif exp_type == 4:  # leader-follow
            type_rews = []
            for ai in range(config.num_agents):
                rew = novelties[:, ai, ai] - novelties[:, :, ai].mean(axis=1)
                if ai == 0:
                    rew[rew > 0.0] = 0.0
                    rew[rew < 0.0] += novelties[rew < 0.0, :, ai].mean(axis=1)
                else:
                    rew[rew > 0.0] += novelties[rew > 0.0, :, ai].mean(axis=1)
                    rew[rew < 0.0] = 0.0
                type_rews.append(rew)
            intr_rews.append(type_rews)

    for i in range(len(config.explr_types)):
        for j in range(config.num_agents):
            if update_irrms:
                intr_rew_rms[i][j].update(intr_rews[i][j].cpu().numpy(), active_envs=active_envs)
            intr_rews[i][j] = intr_rews[i][j].to(device)
            norm_fac = torch.tensor(np.sqrt(intr_rew_rms[i][j].var),
                                    device=device, dtype=torch.float32)
            intr_rews[i][j] /= norm_fac

    return intr_rews

def make_parallel_env(config, seed):
    """Method to create parallel environemnts

    """
    lock = multiprocessing.Lock()
    print(config)
    def get_env_fn(rank):
        def init_env():
            if config.env_type == 'gridworld':
                env = VectObsEnv(GridWorld(config.map_ind,
                                           seed=(seed * 1000),
                                           task_config=config.task_config,
                                           num_agents=config.num_agents,
                                           num_objects=config.num_objects,
                                           need_get=False,
                                           size=(config.length, config.width),
                                           rogue_agents=config.rogue_agents,
                                           treasure_locs=config.treasure_cords,
                                           comm_radius=config.comm_radius,
                                           rogue_reward_factor=config.rogue_reward_factor,
                                           stay_act=False), l=3)
            else:  # vizdoom
                env = VizdoomMultiAgentEnv(task_id=config.task_config,
                                           env_id=(seed - 1) * 64 + rank,  # assumes no more than 64 environments per run
                                           seed=seed * 640 + rank * 10,  # assumes no more than 10 agents per run
                                           lock=lock,
                                           skip_frames=config.frame_skip)
            return env
        return init_env
    return SubprocVecEnv([get_env_fn(i) for i in
                          range(config.n_rollout_threads)])
    #return get_env_fn(100)


def run(config, dir_idx=0):
    """
    Main function to run the
    """
    #set torch params, and relevant directories
    torch.set_num_threads(1)
    env_descr = 'map%i_%iagents_task%i' % (config.map_ind, config.num_agents,
                                           config.task_config)
    model_dir = Path('./models') / config.env_type / env_descr / config.model_name
    
    if not model_dir.exists():
        run_num = 1
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            run_num = 1
        else:
            run_num = max(exst_run_nums) + 1
    curr_run = 'run%i' % run_num
    run_dir = model_dir / curr_run
    log_dir = run_dir / 'logs' #creating the log files
    os.makedirs(log_dir)
    logger = SummaryWriter(str(log_dir))


    torch.manual_seed(run_num)
    np.random.seed(run_num)

    #Load the environment
    #Note that env loads multiple parallel threads (default = 12)
    env, SRObj = search_and_rescue(
        seed=10,
        num_objects=config.num_objects,
        rogue_agents=config.rogue_agents,
        env_type=config.env_type,
        map_ind=config.map_ind,
        comm_radius=config.comm_radius,
        rogue_reward_factor=config.rogue_reward_factor,
        task_config=config.task_config,
        size_L=config.length,
        size_W=config.width,
        need_get=False,
        stay_act=False,
        parallel_threads=True,
        n_rollout_threads=config.n_rollout_threads,
        data_dir=config.output_dir,
        random_target=config.random_target,
        random_epsilon=config.random_epsilon,
        number_of_random_targets=config.number_of_random_targets,
        explore_mode=bool(config.explore_mode)
    )

    #add the nonlinear units for the NN
    if config.nonlinearity == 'relu':
        nonlin = torch.nn.functional.relu
    elif config.nonlinearity == 'leaky_relu':
        nonlin = torch.nn.functional.leaky_relu

    #initialize rewwad heads
    if config.intrinsic_reward == 0:
        n_intr_rew_types = 0
        sep_extr_head = True
    else:
        n_intr_rew_types = len(config.explr_types)
        sep_extr_head = False
    n_rew_heads = n_intr_rew_types + int(sep_extr_head)

    #this block of code initializes the model and parameters associated with the training
    #Load model from file if provided, else initialize e a SAC model
    #Initailize the model
    print(f"config load model: {config.load_model}")
    print(f"config model path: {config.model_path}")
    if config.load_model and config.model_path is not None:
        print(f"-----------------------Loading model-------------------------------")
        model = SAC.init_from_save(filename=config.model_path, load_critic=True)
    else:
        model = SAC.init_from_env(env,
                                  nagents=config.num_agents,
                                  tau=config.tau,
                                  hard_update_interval=config.hard_update,
                                  pi_lr=config.pi_lr,
                                  q_lr=config.q_lr,
                                  phi_lr=config.phi_lr,
                                  adam_eps=config.adam_eps,
                                  q_decay=config.q_decay,
                                  phi_decay=config.phi_decay,
                                  gamma_e=config.gamma_e,
                                  gamma_i=config.gamma_i,
                                  pol_hidden_dim=config.pol_hidden_dim,
                                  critic_hidden_dim=config.critic_hidden_dim,
                                  nonlin=nonlin,
                                  reward_scale=config.reward_scale,
                                  head_reward_scale=config.head_reward_scale,
                                  beta=config.beta,
                                  n_intr_rew_types=n_intr_rew_types,
                                  sep_extr_head=sep_extr_head)
    #initialize
    #Note: Need to include the size of the message space
    replay_buffer = ReplayBuffer(config.buffer_length, model.nagents,
                                 env.state_space,
                                 env.observation_space,
                                 env.action_space)
    if config.rogue_agents is not None:
        #set up adversarial buffer
        adv_buffer = ReplayBuffer(config.buffer_length, model.nagents,
                                 env.state_space,
                                 env.observation_space,

                                 env.action_space)

    #Initialize intrinsic rewards
    intr_rew_rms = [[RunningMeanStd()
                     for i in range(config.num_agents)]
                    for j in range(n_intr_rew_types)]
    eps_this_turn = 0  # episodes so far this turn
    active_envs = np.ones(config.n_rollout_threads)  # binary indicator of whether env is active
    env_times = np.zeros(config.n_rollout_threads, dtype=int)
    env_ep_extr_rews = np.zeros(config.n_rollout_threads)
    env_extr_rets = np.zeros(config.n_rollout_threads)
    env_ep_intr_rews = [[np.zeros(config.n_rollout_threads) for i in range(config.num_agents)]
                        for j in range(n_intr_rew_types)]
    recent_ep_extr_rews = deque(maxlen=100)
    recent_ep_intr_rews = [[deque(maxlen=100) for i in range(config.num_agents)]
                           for j in range(n_intr_rew_types)]
    recent_ep_lens = deque(maxlen=100)
    recent_found_treasures = [deque(maxlen=100) for i in range(config.num_agents)]
    recent_tiers_completed = deque(maxlen=100)
    meta_turn_rets = []
    extr_ret_rms = [RunningMeanStd() for i in range(n_rew_heads)]
    t = 0
    steps_since_update = 0
    rollout_id = 0

    #Reset state
    state, obs, target_pos = env.reset()
    #initialize processing time for steps
    t_rollout = np.zeros(config.n_rollout_threads)
    rollout_ids = np.zeros_like(t_rollout)
    bool_rollout = np.zeros(config.n_rollout_threads, dtype=bool)
    need_plot = bool_rollout.copy()

    #initialize global rewards
    global_rewards = np.zeros_like(rollout_ids)
    global_adv_rewards = np.zeros_like(global_rewards)

    #initialize the vectors
    rewards_with_timer = np.zeros(config.n_rollout_threads)
    old_rewards_with_timer = np.zeros_like(rewards_with_timer)
    old_adv_rewards_with_timer = np.zeros_like(old_rewards_with_timer)

    ####Calling ob the SRObj to (a) initialize; (b) updating the global state and (c)
    SRObj.initialize_global_state(state=state)
    SRObj.update_global_state(
        state=state,
        active_envs=active_envs,
        step=0,
        rollout_ids=rollout_ids
    )
    SRObj.visualize(
        step=0,
        init_flag=True,
        target_pos=target_pos,
        rollout_ids=rollout_ids
    )

    #This block of code defines the cooperative and adv agents and user inputss
    #get the agent ids
    if config.rogue_agents is not None:
        rogue_agents = config.rogue_agents #rogue agents
        c_agents = [a+1 for a in range(config.num_agents) if a+1 not in rogue_agents] #cooperative agents
        print(f"rogue_agents: {rogue_agents}")
        print(f"c_agents: {c_agents}")
    else:
        c_agents = [a+1 for a in range(config.num_agents)]

    #training parameters
    global_time = []

    #initialize aggregations for saving into a csv
    agg_coop_rewards_time = None
    agg_adv_rewards_time = None

    #compute parameters for beta
    param_k = -np.log(config.beta_low/config.beta) / (1 - config.threshold_t)*config.max_episode_length

    #training starts here
    while t < config.train_time: #and not np.all(bool_rollout)
        model.prep_rollouts(device='cuda' if config.gpu_rollout else 'cpu')
        # convert to torch tensor
        torch_obs = apply_to_all_elements(obs, lambda x: torch.tensor(x, dtype=torch.float32, device='cuda' if config.gpu_rollout else 'cpu'))
        # get actions as torch tensors
        torch_agent_actions = model.step(torch_obs, explore=True)
        #this block of code provides the
        # convert actions to numpy arrays
        agent_actions = apply_to_all_elements(torch_agent_actions, lambda x: x.cpu().data.numpy())
        # rearrange actions to be per environment
        actions = [[ac[i] for ac in agent_actions] for i in range(int(active_envs.sum()))]

        #this block of code tries to take a step. If not possible, it restarts the environment
        try:
            with timeout(seconds=1):
                next_state, next_obs, rewards, dones, infos = env.step(actions, env_mask=active_envs)
                if config.rogue_agents is not None:
                    adv_rewards = np.array([i['adv_rewards'] for i in infos],
                                          dtype=float)
                    #print(f"Testing adversarial rewars: {adv_rewards}")
        # either environment got stuck or vizdoom crashed (vizdoom is unstable w/ multi-agent scenarios)
        except (TimeoutError, ViZDoomErrorException, ViZDoomIsNotRunningException, ViZDoomUnexpectedExitException) as e:
            print("Environments are broken...")
            env.close(force=True)
            print("Closed environments, starting new...")
            env = make_parallel_env(config, run_num)
            state, obs, target_pos = env.reset()
            env_ep_extr_rews[active_envs.astype(bool)] = 0.0
            env_extr_rets[active_envs.astype(bool)] = 0.0
            for i in range(n_intr_rew_types):
                for j in range(config.num_agents):
                    env_ep_intr_rews[i][j][active_envs.astype(bool)] = 0.0
            env_times = np.zeros(config.n_rollout_threads, dtype=int)
            state = apply_to_all_elements(state, lambda x: x[active_envs.astype(bool)])
            obs = apply_to_all_elements(obs, lambda x: x[active_envs.astype(bool)])
            continue

        steps_since_update += int(active_envs.sum())

        #This block of code ccomputes the intrinsic rewards.
        if config.intrinsic_reward == 1:
            # if using state-visit counts, store state indices
            # shape = (n_envs, n_agents, n_inds)
            state_inds = np.array([i['visit_count_lookup'] for i in infos],
                                  dtype=int)
            state_inds_t = state_inds.transpose(1, 0, 2)
            novelties = get_count_based_novelties(env, state_inds_t, device='cpu')
            intr_rews = get_intrinsic_rewards(novelties, config, intr_rew_rms,
                                              update_irrms=True, active_envs=active_envs,
                                              device='cpu')
            intr_rews = apply_to_all_elements(intr_rews, lambda x: x.numpy().flatten())
            #print(f"intr_rews: {intr}")
        else:
            intr_rews = None
            state_inds = None
            state_inds_t = None

        #block of code to compute extrinsic rewards based on unique state visits
        if config.explore_mode == 1:
            #print(f"max episode: {config.max_episode_length}")
            rewards_with_timer, adv_rewards_with_timer = SRObj.compute_final_reward()
            rewards = rewards_with_timer - old_rewards_with_timer
            adv_rewards = adv_rewards_with_timer - old_adv_rewards_with_timer
            old_rewards_with_timer = rewards_with_timer.copy()
            old_adv_rewards_with_timer = adv_rewards_with_timer.copy()
            #print(f"rewards: {rewards_with_timer}")
            #print(f"adv rewards: {adv_rewards_with_timer}")

        #Note: Need to include Message space as part of the push method
        replay_buffer.push(state, obs, agent_actions, rewards, next_state, next_obs, dones,
                           state_inds=state_inds)
        #print(f"adv rewards: {adv_rewards}")
        if config.rogue_agents is not None:
            adv_buffer.push(state, obs, agent_actions, adv_rewards, next_state, next_obs, dones,
                               state_inds=state_inds)

        #print(f"rewards: {rewards}")
        env_ep_extr_rews[active_envs.astype(bool)] += np.array(rewards)
        env_extr_rets[active_envs.astype(bool)] += np.array(rewards) * config.gamma_e**(env_times[active_envs.astype(bool)])
        env_times += active_envs.astype(int)
        #print(f"env_times: {env_times}")
        if intr_rews is not None:
            for i in range(n_intr_rew_types):
                for j in range(config.num_agents):
                    env_ep_intr_rews[i][j][active_envs.astype(bool)] += intr_rews[i][j]
        over_time = env_times >= config.max_episode_length
        full_dones = np.zeros(config.n_rollout_threads)

        #dones is for active environment, full_dones contains the global array
        for i, env_i in enumerate(np.where(active_envs)[0]):
            full_dones[env_i] = dones[i]

        #call the visualize and the update function here
        if np.any(full_dones) or np.any(over_time):
            SRObj.update_global_state(
                state=next_state,
                active_envs=active_envs,
                rewards=rewards,
                step=t,
                bool_rollout=bool_rollout,
                rollout_ids=rollout_ids
            )
            SRObj.visualize(
                step=t,
                init_flag=False,
                full_dones=full_dones,
                bool_rollout=bool_rollout,
                rollout_ids=rollout_ids,
                target_pos=target_pos
            )

        need_reset = np.logical_or(full_dones, over_time)
        active_over_time = env_times[active_envs.astype(bool)] >= config.max_episode_length
        active_need_reset = np.logical_or(dones, active_over_time)

        if any(need_reset):
            #reset environemnt
            try:
                with timeout(seconds=100):
                    SRObj.update_global_state(
                        state=next_state,
                        active_envs=active_envs,
                        need_reset=need_reset,
                        rewards=rewards,
                        rollout_ids=rollout_ids,
                        bool_rollout=bool_rollout,
                        step=t
                    )
                    print(f"Episode complete for environments: {np.where(need_reset)[0]}")
                    #print(f"Bool rollout: {bool_rollout}")

                    state, obs, target_pos = env.reset(need_reset=need_reset)

            # either environment got stuck or vizdoom crashed (vizdoom is unstable w/ multi-agent scenarios)
            except (TimeoutError, ViZDoomErrorException, ViZDoomIsNotRunningException, ViZDoomUnexpectedExitException) as e:
                print("Environments are broken...")
                env.close(force=True)
                print("Closed environments, starting new...")
                env = make_parallel_env(config, run_num)
                state, obs = env.reset()
                # other envs that were force reset (rest taken care of in subsequent code)
                other_reset = np.logical_not(need_reset)
                env_ep_extr_rews[other_reset.astype(bool)] = 0.0
                env_extr_rets[other_reset.astype(bool)] = 0.0
                for i in range(n_intr_rew_types):
                    for j in range(config.num_agents):
                        env_ep_intr_rews[i][j][other_reset.astype(bool)] = 0.0
                env_times = np.zeros(config.n_rollout_threads, dtype=int)
        else:
            state, obs = next_state, next_obs
            #print(f"bool rollout 4: {bool_rollout}")
            SRObj.update_global_state(
                state=state,
                active_envs=active_envs,
                need_reset=need_reset,
                rewards=rewards,
                rollout_ids=rollout_ids,
                bool_rollout=bool_rollout,
                step=t
            )

            #saving
            step_range = [i for i in range(int(t), int(t + config.n_rollout_threads))]
            bool_range = np.asarray(step_range) % config.img_interval == 0
            if np.any(bool_range):
                SRObj.visualize(
                    step=t,
                    init_flag=False,
                    rollout_ids=rollout_ids,
                    full_dones=full_dones,
                    bool_rollout=bool_rollout,
                    target_pos=target_pos
                )

        #add adv_rewars to cases where over_time
        if config.explore_mode == 0 and config.rogue_agents is not None:
            for i, env_i in enumerate(np.where(over_time)[0]):
                adv_rewards[i] += 10

        #This block of code keeps track of which environments are completed
        #dones is for active environment, full_dones contains the global array
        for i, env_i in enumerate(np.where(active_envs)[0]):
            global_rewards[env_i] = rewards[i]
            if config.rogue_agents is not None:
                global_adv_rewards[env_i] = adv_rewards[i]
            if full_dones[env_i] > 0 or over_time[env_i] > 0:
                rollout_ids[env_i] += 1
            if not bool_rollout[env_i] and (full_dones[env_i]  or over_time[env_i])> 0:
                t_rollout[env_i] = t #if false, then store the numbers
                bool_rollout[env_i] = True

        #add adv_rewards for over_time
        #This block of code resets environment when all the targets have been located in all environments and writes to global_time.txt
        #reinitialize after all the enviroments have been collected

        #Export time
        if config.explore_mode == 0 or config.explore_mode == 1:
            # print(f'global rewards: {global_rewards}')
            # print(f"global adv rewards: {global_adv_rewards}")
            agg_coop_rewards_time = global_rewards if agg_coop_rewards_time is None else np.vstack((agg_coop_rewards_time, global_rewards))
            agg_adv_rewards_time = global_adv_rewards if agg_adv_rewards_time is None else np.vstack((agg_adv_rewards_time, global_adv_rewards))
            np.savetxt(os.path.join(config.output_dir, "rewards_with_time.csv"), agg_coop_rewards_time, delimiter=',')
            np.savetxt(os.path.join(config.output_dir, "adv_rewards_with_time.csv"), agg_adv_rewards_time, delimiter=',')
        if np.all(bool_rollout):
            global_time.append(np.mean(t_rollout))
            t_rollout = np.zeros(config.n_rollout_threads)
            bool_rollout = np.zeros(config.n_rollout_threads, dtype=bool)
            print(f"global time: {global_time}")
            with open(os.path.join(config.output_dir, "global_time.txt"), "a") as f:
                f.write(f"Mean Global Time for rollout {rollout_id} is {global_time}")
            #if config.explore_mode == 1:
                # np.savetxt("rewards_with_time.csv", agg_coop_rewards_time, delimiter=',')
                # np.savetxt("adv_rewards_with_time.csv", agg_adv_rewards_time, delimiter=',')
        ###-------end of block--------------------------------------------------------

        #This block keeps track of both intrinsic and extrinsic rewards

        for env_i in np.where(need_reset)[0]:
            recent_ep_extr_rews.append(env_ep_extr_rews[env_i])
            meta_turn_rets.append(env_extr_rets[env_i])
            if intr_rews is not None:
                for j in range(n_intr_rew_types):
                    for k in range(config.num_agents):
                        # record intrinsic rewards per step (so we don't confuse shorter episodes with less intrinsic rewards)
                        recent_ep_intr_rews[j][k].append(env_ep_intr_rews[j][k][env_i] / env_times[env_i])
                        env_ep_intr_rews[j][k][env_i] = 0

            recent_ep_lens.append(env_times[env_i])
            env_times[env_i] = 0
            env_ep_extr_rews[env_i] = 0
            env_extr_rets[env_i] = 0
            if config.explore_mode == 1:
                rewards_with_timer[env_i] = 0
                old_rewards_with_timer[env_i] = 0
                old_adv_rewards_with_timer[env_i] = 0
                #r_ext[env_i] = 0
            eps_this_turn += 1

            if eps_this_turn + active_envs.sum() - 1 >= config.metapol_episodes:
                active_envs[env_i] = 0

        for i in np.where(active_need_reset)[0]:
            for j in range(config.num_agents):
                # len(infos) = number of active envs
                recent_found_treasures[j].append(infos[i]['n_found_treasures'][j])
            if config.env_type == 'gridworld':
                recent_tiers_completed.append(infos[i]['tiers_completed'])

        #This block of code is where the RL model is updated
        if eps_this_turn >= config.metapol_episodes:
            if not config.uniform_heads and n_rew_heads > 1:
                meta_turn_rets = np.array(meta_turn_rets)
                if all(errms.count < 1 for errms in extr_ret_rms):
                    for errms in extr_ret_rms:
                        errms.mean = meta_turn_rets.mean()
                extr_ret_rms[model.curr_pol_heads[0]].update(meta_turn_rets)
                #Updates the meta policy
                for i in range(config.metapol_updates):
                    model.update_heads_onpol(meta_turn_rets, extr_ret_rms, logger=logger)
            pol_heads = model.sample_pol_heads(uniform=config.uniform_heads)
            #Troubleshoot policy heads:
            with open(os.path.join(config.output_dir, "metapolicy_info.txt"), "a") as f:
                f.write(f"rollout_id: {rollout_id}, config uniform heads: {config.uniform_heads}, episodes this turn: {eps_this_turn}, pol_heads: {pol_heads}")
            print(f"config uniform heads: {config.uniform_heads}")
            print(f"episodes this turn: {eps_this_turn}")
            print(f"roll_out_id is {rollout_id}")
            print(f"pol_heads: {pol_heads}")
            model.set_pol_heads(pol_heads)
            eps_this_turn = 0
            meta_turn_rets = []
            active_envs = np.ones(config.n_rollout_threads)

        #saving intrinsic rewards to file
        # print(f"length of intr rews: {len(intr_rews)}")
        # print(f"length of arrays: {(len(intr_rews[0]))}")

        if any(need_reset):  # reset returns state and obs for all envs, so make sure we're only looking at active envs
            state = apply_to_all_elements(state, lambda x: x[active_envs.astype(bool)])
            obs = apply_to_all_elements(obs, lambda x: x[active_envs.astype(bool)])

        if (len(replay_buffer) >= max(config.batch_size,
                                      config.steps_before_update) and
                (steps_since_update >= config.steps_per_update)):
            steps_since_update = 0
            print('Updating at time step %i' % t)
            #print(f"External rewards at time {t}: {r_ext}")
            model.prep_training(device='cuda' if config.use_gpu else 'cpu')

            for u_i in range(config.num_updates):
                sample = replay_buffer.sample(config.batch_size,
                                              to_gpu=config.use_gpu,
                                              state_inds=(config.intrinsic_reward == 1))

                ##additional block of coded for adversarial rewards
                if config.rogue_agents is not None:
                    adv_sample = adv_buffer.sample(config.batch_size,
                                            to_gpu=config.use_gpu,
                                            state_inds=(config.intrinsic_reward == 1))

                if config.intrinsic_reward == 0:  # no intrinsic reward
                    intr_rews = None
                    state_inds = None
                else:
                    sample, state_inds = sample
                    if config.rogue_agents is not None:
                        adv_sample, _ = adv_sample
                    novelties = get_count_based_novelties(
                        env, state_inds,
                        device='cuda' if config.use_gpu else 'cpu')
                    intr_rews = get_intrinsic_rewards(novelties, config, intr_rew_rms,
                                                      update_irrms=False,
                                                      device='cuda' if config.use_gpu else 'cpu')
                    #print(f"intr_rews: {intr_rews}")
                if config.rogue_agents is None and config.inference == 0:
                    model.update_critic(sample, logger=logger, intr_rews=intr_rews, selected_agent_ids=c_agents)
                    model.update_policies(sample, logger=logger)
                    model.update_all_targets()
                else:
                    if config.inference == 0:
                        #get the sample for adversarial agents
                        #for critic_iter in range(1):
                        #compute beta for cooperative agents
                        beta_running = config.beta if env_times[0] <= config.threshold_t*config.max_episode_length else config.beta*np.exp(-param_k*env_times[0])
                        #print(f"The value of beta is: {beta_running}")
                        model.update_critic(sample, logger=logger, intr_rews=intr_rews, selected_agent_ids=c_agents)
                        model.update_policies(sample, logger=logger, selected_agent_ids=c_agents, beta=beta_running)
                        model.update_all_targets()

                        #adversarial agents
                        #for critic_iter in range(1):
                        model.update_critic(adv_sample, logger=logger, intr_rews=intr_rews, selected_agent_ids=rogue_agents)
                        model.update_policies(adv_sample, logger=logger, selected_agent_ids=rogue_agents, beta=0.0)
                        model.update_all_targets()

            if len(recent_ep_extr_rews) > 10:
                logger.add_scalar('episode_rewards/extrinsic/mean',
                                  np.mean(recent_ep_extr_rews), t)
                logger.add_scalar('episode_lengths/mean',
                                  np.mean(recent_ep_lens), t)
                if config.intrinsic_reward == 1:
                    for i in range(n_intr_rew_types):
                        for j in range(config.num_agents):
                            logger.add_scalar('episode_rewards/intrinsic%i_agent%i/mean' % (i, j),
                                              np.mean(recent_ep_intr_rews[i][j]), t)
                for i in range(config.num_agents):
                    logger.add_scalar('agent%i/n_found_treasures' % i, np.mean(recent_found_treasures[i]), t)
                logger.add_scalar('total_n_found_treasures', sum(np.array(recent_found_treasures[i]) for i in range(config.num_agents)).mean(), t)
                if config.env_type == 'gridworld':
                    logger.add_scalar('tiers_completed', np.mean(recent_tiers_completed), t)

        if t % config.save_interval < config.n_rollout_threads:
            model.prep_training(device='cpu')
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            model.save(run_dir / 'incremental' / ('model_%isteps.pt' % (t + 1)))
            model.save(run_dir / 'model.pt')

        t += active_envs.sum()
    model.prep_training(device='cpu')
    model.save(run_dir / 'model.pt')
    logger.close()
    env.close(force=(config.env_type == 'vizdoom'))

    return model, t_rollout, global_time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name",
                        help="Name of directory to store " +
                             "model/training contents")
    parser.add_argument("--env_type", type=str, default='gridworld', choices=['gridworld', 'vizdoom'])
    parser.add_argument("--map_ind", help="Index of map to use (only for gridworld)", type=int,
                        default=1)
    parser.add_argument("--num_agents", help="Number of agents (vizdoom only supports 2 at the moment)", type=int,
                        default=2)
    parser.add_argument("--num_objects", help="Number of treasures", type=int,
                        default=2)
    parser.add_argument("--task_config", help="Index of task configuration",
                        type=int, default=1) #probably not relevant
    parser.add_argument("--frame_skip", help="How many frames to skip per step (only for vizdoom)", type=int,
                        default=2)
    parser.add_argument("--intrinsic_reward", type=int, default=1,
                        help="Use intrinsic reward for exploration\n" +
                             "0: No intrinsic reward\n" +
                             "1 (default): Intrinsic reward using state visit counts")
    parser.add_argument("--explr_types", type=int, nargs='*', default=[1, 2, 3], #change to [2, 4]
                        help="Type of exploration, can provide multiple\n" +
                             "0: Independent exploration\n" +
                             "1: Minimum exploration\n" +
                             "2: Covering exploration\n" +
                             "3: Burrowing exploration\n" +
                             "4: Leader-Follower exploration\n")
    parser.add_argument("--uniform_heads", action="store_true",
                        help="Meta-policy samples all heads uniformly")
    parser.add_argument("--beta", type=float, default=0.1,
                        help="Weighting for intrinsic reward")
    parser.add_argument("--decay", type=float, default=0.7,
                        help="Decay rate for state-visit counts in intrinsic reward")
    parser.add_argument("--n_rollout_threads", default=12, type=int) #number of parallel environments
    parser.add_argument("--buffer_length", default=int(1e6), type=int,
                        help="Set to 5e5 for ViZDoom (if memory limited)")
    #parser.add_argument("--train_time", default=int(1e6), type=int)
    parser.add_argument("--train_time", default=int(1e4), type=int)
    parser.add_argument("--max_episode_length", default=50000, type=int)
    parser.add_argument("--steps_per_update", default=100, type=int)
    parser.add_argument("--metapol_episodes", default=12, type=int,
                        help="Number of episodes to rollout before updating the meta-policy " +
                             "(policy selector). Better if a multiple of n_rollout_threads")
    parser.add_argument("--steps_before_update", default=0, type=int)
    parser.add_argument("--num_updates", default=50, type=int,
                        help="Number of SAC updates per cycle")
    parser.add_argument("--metapol_updates", default=100, type=int,
                        help="Number of updates for meta-policy per turn")
    parser.add_argument("--batch_size",
                        default=1024, type=int,
                        help="Batch size for training. \n"
                             "Set to 128 for ViZDoom scenarios")
    #parser.add_argument("--save_interval", default=100000, type=int)
    parser.add_argument("--save_interval", default=200, type=int)
    parser.add_argument("--pol_hidden_dim", default=32, type=int)
    parser.add_argument("--critic_hidden_dim", default=128, type=int,
                        help="Set to 256 for ViZDoom scenarios")
    parser.add_argument("--nonlinearity", default="relu", type=str,
                        choices=["relu", "leaky_relu"])
    parser.add_argument("--pi_lr", default=0.001, type=float,
                        help="Set to 0.0005 for ViZDoom scenarios")
    parser.add_argument("--q_lr", default=0.001, type=float,
                        help="Set to 0.0005 for ViZDoom scenarios")
    parser.add_argument("--phi_lr", default=0.04, type=float)
    parser.add_argument("--adam_eps", default=1e-8, type=float)
    parser.add_argument("--q_decay", default=1e-3, type=float)
    parser.add_argument("--phi_decay", default=1e-3, type=float)
    parser.add_argument("--tau", default=0.005, type=float)
    parser.add_argument("--hard_update", default=None, type=int,
                        help="If specified, use hard update for target critic" +
                             "every _ steps instead of soft update w/ tau")
    parser.add_argument("--gamma_e", default=0.99, type=float)
    parser.add_argument("--gamma_i", default=0.99, type=float)
    parser.add_argument("--reward_scale", default=100., type=float)
    parser.add_argument("--head_reward_scale", default=5., type=float)
    parser.add_argument("--use_gpu", action='store_true',
                        help='Use GPU for training')
    parser.add_argument("--gpu_rollout", action='store_true',
                        help='Use GPU for rollouts (more useful for lots of '
                        'parallel envs or image-based observations')

    parser.add_argument("--length", default=20, type=int)
    parser.add_argument("--width", default=20, type=int)
    parser.add_argument("--rogue_agents", default=None, type=list)
    parser.add_argument("--rogue_reward_factor", default=1.0, type=float)
    #output directory
    parser.add_argument("--output_dir", default="out_baseline", type=str)
    #adding communication radius to the list of arguments
    parser.add_argument("--comm_radius", default=20, type=int)

    #add parameters relevant to imaage and save interval
    parser.add_argument("--img_interval", default=100, type=int,
                        help="Frequency with which the images are saved")
    parser.add_argument("--csv_interval", default=1000, type=int,
                        help="Frequency with which the csvs are saved")

    #parameters associated with load_model
    parser.add_argument("--load_model", default=False, type=bool,
                        help="whether to load a model from past state")
    parser.add_argument("--model_path", default=None, type=str,
                        help="Full path to location of model to be loaded")
    parser.add_argument("--inference", default=0, type=int,
                        help="Bool to select if we are interested in inference only")

    #parameter associated with stochastic policy
    # parameters associated with load_model
    parser.add_argument("--random_target", default=1, type=int,
                        help="Whether we have a stochastic target or not")
    parser.add_argument("--list_of_random_targets", nargs="+",  default=None,
                        help="Whether we have a stochastic target or not")
    parser.add_argument("--random_epsilon",  default=None, type=int,
                        help="Bounding the randomness by epsilon")
    parser.add_argument("--number_of_random_targets", default=None, type=int,
                        help="Whether we have a stochastic target or not")

    #parameters pertaining to extrinsic reward structure
    parser.add_argument("--explore_mode", default=0, type=int,
                        help="Whether we should assign")

    #parameters for weighting extrinsic vs. intrinsic reward
    parser.add_argument("--threshold_t", default=0.4, type=float,
                        help="Fraction of time for which intrinsic rewards are weighed at their max")
    parser.add_argument("--beta_low", default=0.001, type=float,
                        help="Value of beta when t = max_episode ")

    config = parser.parse_args()
    if config.rogue_agents is not None:
        config.rogue_agents = [int(rg) for rg in config.rogue_agents]
    config.random_target = bool(config.random_target)
    config.load_model = bool(config.load_model)
    print(f"list_of_random_targets: {config.list_of_random_targets}")
    if config.list_of_random_targets is not None:
        config.list_of_random_targets = [int(rt) for rt in config.list_of_random_targets]
    print(f"rnadom target: {config.random_target}")
    print(f"list_of_random_targets: {config.list_of_random_targets}")
    print(f"config.train_time: {config.train_time}")
    print(f"config.train_time: {config.train_time}")
    print(f"config.load _model {config.load_model}")
    print(f"config.model_dict {config.model_path}")
    #Add iinputs either from the command line or from here
    #config.rogue_agents = [3]

    #print(env)
    model, t_roll, global_time = run(config)

