import argparse
import torch
import os
import multiprocessing
import numpy as np
import pandas as pd

from utils.env_wrappers import SubprocVecEnv
from envs.magw.multiagent_env import GridWorld, VectObsEnv

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

#wrapper for
class SearchRescue:

    def __init__(
            self,
            seed,
            num_objects,
            rogue_agents,
            env_type="gridworld",
            map_ind=None,
            comm_radius=20,
            rogue_reward_factor=1.0,
            task_config=1,
            size_L=20,
            size_W=20,
            need_get=False,
            stay_act=False,
            random_target=False,
            random_epsilon=None,
            number_of_random_targets=None,
            parallel_threads=True,
            n_rollout_threads=12,
            data_dir="magw_data",
            explore_mode=False
    ):
        """
        :param seed: (int) - random seed for simulation
        :param num_agents: (int) - total number of agents (including rogue and adversarial)
        :param state_shape: (tuple) - shape of the the global state space
        :param map_file: (str) - full path to txt file containing the map, needed for gridworld
        :param comm_radius: (int) - communication radius, needed for gridworld
        :param rogue_reward_factor: (float) - factor by which the adversary's reward is scaled
        """

        self.seed = seed
        self.num_objects = num_objects
        self.map_ind = map_ind
        self.rogue_agents = rogue_agents
        self.comm_radius = comm_radius
        self.rogue_reward_factor = rogue_reward_factor
        self.env_type = env_type
        self.task_config = task_config
        self.size = (int(size_L), int(size_W))
        self.need_get = need_get
        self.stay_act = stay_act
        self.random_target = random_target
        self.random_epsilon = random_epsilon
        self.parallel_thread = parallel_threads
        self.data_dir = data_dir
        self.explore_mode = explore_mode
        self.n_rollout_threads = n_rollout_threads
        self.number_of_random_targets = number_of_random_targets

        #generate data directory
        self.create_dirs()

        if self.env_type == "gridworld":
            self.map_file = os.path.join(
                "envs", "magw", "maps", "map{}_{}_multi.txt".format(self.map_ind, self.num_objects)
            )
            self.target_locs, self.num_objects, self.agent_locs, self.num_agents, self.agent_ids = self.get_pos()
            self.wall_mat = self.get_wall_cords()

        #TO DO
        # - AssertionError mapfie needes to be provide

    def initialize_global_state(self, state):
        """
        method to intiialize global state matrix need ed
        """
        self.global_state =  np.zeros((self.n_rollout_threads, state.shape[-1]))
        self.global_state_mat = np.zeros((self.n_rollout_threads, self.num_agents, self.size[0], self.size[1]))
        self.global_rewards = np.zeros((self.n_rollout_threads, ))

        self.list_of_dicts = []

        #initialize the ddtaframe
        for n in range(self.n_rollout_threads):
            data = dict()
            data["step"] = [0]
            data["reward"] = [0]

            for a, agent_id in enumerate(self.agent_ids):
                data["Agent_{}".format(agent_id)] = [self.agent_locs[a]]
            self.list_of_dicts.append(data)

        return None

    def create_dirs(self):

        """
        method to create relevant directories to store data
        """
        curr_dir = os.getcwd()
        dir_to_gen = os.path.join(curr_dir, self.data_dir)
        if not os.path.exists(dir_to_gen):
            os.makedirs(dir_to_gen)
        return None

    def make_parallel_env(self):
        """
        method to create parallel environment
        """
        map_ind = self.map_ind
        seed = self.seed
        task_config = self.task_config
        num_agents = self.num_agents
        num_objects = self.num_objects
        size = self.size
        rogue_agents = self.rogue_agents
        treasure_cords = self.target_locs
        comm_radius = self.comm_radius
        rogue_reward_factor = self.rogue_reward_factor
        stay_act = self.stay_act
        random_target = self.random_target
        random_epsilon = self.random_epsilon
        number_of_random_targets = self.number_of_random_targets
        explore_mode = self.explore_mode

        lock = multiprocessing.Lock()
        def get_env_fn(rank):
            def init_env():
                if self.env_type == 'gridworld':
                    env = VectObsEnv(GridWorld(map_ind,
                                               seed=(seed * 1000),
                                               task_config=task_config,
                                               num_agents=num_agents,
                                               num_objects=num_objects,
                                               need_get=self.need_get,
                                               size=size,
                                               rogue_agents=rogue_agents,
                                               treasure_locs=treasure_cords,
                                               comm_radius=comm_radius,
                                               rogue_reward_factor=rogue_reward_factor,
                                               random_target=random_target,
                                               random_epsilon=random_epsilon,
                                               number_of_random_targets=number_of_random_targets,
                                               explore_mode=explore_mode,
                                               stay_act=stay_act), l=3)
                else:  # vizdoom
                    env = VizdoomMultiAgentEnv(task_id=config.task_config,
                                               env_id=(seed - 1) * 64 + rank,
                                               # assumes no more than 64 environments per run
                                               seed=seed * 640 + rank * 10,
                                               # assumes no more than 10 agents per run
                                               lock=lock,
                                               skip_frames=config.frame_skip)
                return env

            return init_env

        return SubprocVecEnv([get_env_fn(i) for i in
                              range(self.n_rollout_threads)])


    def get_pos(self):

        """
        method to get target coordinates
        """

        with open(self.map_file) as f:
            lines = f.readlines()

        # initialize wall matrix
        list_of_locs = []
        list_of_agent_locs = []
        list_of_agent_ids = []

        for row, line in enumerate(lines):
            cells = list(line.rstrip("\n"))
            for c, cell in enumerate(cells):
                if cell.isalpha():
                    list_of_locs.append((row, c))
                if cell.isnumeric():
                    list_of_agent_locs.append((row, c))
                    list_of_agent_ids.append(int(cell))

        return list_of_locs, len(list_of_locs), list_of_agent_locs, len(list_of_agent_locs), list_of_agent_ids

    def update_global_state(
            self,
            state,
            step=0,
            need_reset=None,
            active_envs=None,
            rewards=None,
            save_interval=100,
            bool_rollout=None,
            rollout_ids=None
    ):

        """
        method for managing all the parallel environments and storing data
        :param state: (np.array) - 2D matrix (n_rollout_thread, number of features) containing the global state
        :param step: (int) - the timestep at a given timestep
        :param need_reset: (np.array) - array of bool of shape (n_rollout_thread,) indiciating whether any of the environment needs reset
        :param active_envs: (np.array) - array of ints indicating whether a thread is active
        :param rewards: (np.array) - array of shape (n_rollout_thread, ) indicating
        """

        #create directory if already not created
        # generate the rollout path
        if not os.path.exists(os.path.join(self.data_dir, "data_rollout")):
            os.makedirs(os.path.join(self.data_dir, "data_rollout"))

        #initialize active_envs
        active_envs = np.ones(self.n_rollout_threads) if active_envs is None else active_envs
        rewards = np.zeros(self.n_rollout_threads) if rewards is None else rewards
        rollout_ids = np.zeros(self.n_rollout_threads) if rollout_ids is None else rollout_ids

        # get the array of states across all processes
        assert active_envs is not None
        self.global_agent_pos = [] #list of list of tuples indicating the agents position during simulation
        #variable storing the state space for all the
        for i, env_i in enumerate(np.where(active_envs)[0]):
            self.global_state[env_i, :] = state[i, :]
            #if rewards is not None:
            self.global_rewards[i] = rewards[i]

        if need_reset is not None:
            for n in range(self.n_rollout_threads):
                if need_reset[n] and bool_rollout is not None and bool_rollout[n] == False: #initialize when the need reset is True, but bool_rollout is not incremented yet
                    print(f"*****************************Resetting for environemt {n}***********************************")
                    self.global_state_mat[n, :, :, :] = 0

        #determine wheteher to save or not
        step_range = [i for i in range(int(step), int(step + self.n_rollout_threads))]
        bool_range = np.asarray(step_range)% save_interval == 0

        for n in range(self.n_rollout_threads):
            #create dirctory if not already generated
            process_path = os.path.join(self.data_dir, "data_rollout_{}".format(rollout_ids[n]), "fig_process_{}".format(n))
            if not os.path.exists(process_path):
                os.makedirs(process_path)
            agent_cords = self.get_coords_nagents(n) #get the position of the agents in a given thread
            self.global_agent_pos.append(agent_cords)
            for agent, loc in enumerate(agent_cords):
                row_loc = loc[0]
                col_loc = loc[1]
                #add need reset here
                if need_reset is None or not need_reset[n]:
                    if bool_rollout is None or not bool_rollout[n]:
                        self.global_state_mat[n, agent, row_loc, col_loc] += 1
            #add block to store data:
            data_dict = self.list_of_dicts[n]
            #append
            data_dict["step"].append(step)
            data_dict["reward"].append(self.global_rewards[n])
            for a, agent_id in enumerate(self.agent_ids):
                _ag_temp = data_dict["Agent_{}".format(agent_id)]
                data_dict["Agent_{}".format(agent_id)].append(agent_cords[a])

            if (step > 0 and need_reset is not None and need_reset[n]) or np.any(bool_range):
                df = pd.DataFrame(data_dict)
                df.to_csv(os.path.join(process_path, "data_rollout_{}.csv".format(rollout_ids[n])))

        return None


    def get_wall_cords(self):

        """
        read txt file
        :param txt_file: (str) - full path with string name to the .txt file
        :return:
        """
        assert self.map_file is not None

        with open(self.map_file) as f:
            lines = f.readlines()

        num_of_rows = len(lines)  # number of rows in the grid
        num_of_cols = len(list(lines[0].rstrip("\n")))

        # initialize wall matrix
        wall_mat = np.zeros((num_of_rows, num_of_cols))
        for row, line in enumerate(lines):
            cells = list(line.rstrip("\n"))
            for c, cell in enumerate(cells):
                if cell == "#":
                    wall_mat[row, c] = 1

        return wall_mat

    #method for visualizing
    def visualize(
            self,
            step,
            rollout_ids=None,
            init_flag=False,
            full_dones=None,
            target_pos=None,
            bool_rollout=None,
            viz_all=True
    ):
        process_ids = np.arange(self.n_rollout_threads).tolist()
        # for mat the target position based on the environment id
        if target_pos is not None:
            #print(f"target_pos: {target_pos}")
            target_list = self.format_target_pos(target_pos=target_pos)

        for n in process_ids:
            #creating the paths for the figures
            rollout_path = "data_rollout" if rollout_ids is None else "data_rollout_{}".format(int(rollout_ids[n]))
            rollout_path = os.path.join(self.data_dir, rollout_path)
            #print(f"rollout_path: {rollout_path}")
            if not os.path.exists(rollout_path):
                os.makedirs(rollout_path)
            process_path = os.path.join(self.data_dir, "data_rollout_{}".format(int(rollout_ids[n])), "fig_process_{}".format(n))
            #print(f"process path: {process_path}")
            if not os.path.exists(process_path):
                os.makedirs(process_path)
            #get the agent coordinates
            agent_cords = self.get_coords_nagents(n)

            #get the treasure coordinates for environment "n"
            target_pos_n = target_list[n] if target_pos is not None else None
            if init_flag:
                fig_name = os.path.join(process_path, "init.png")
                #print(f"fig_name: {fig_name}")
                self.plot_grid(
                    agent_cords=agent_cords,
                    state_mat=None,
                    fig_name=fig_name,
                    agent_nums=range(self.num_agents),
                    target_pos=target_pos_n
                )
            else:
                #plot everything together in the same plot
                if viz_all:
                    if full_dones is not None and full_dones[n]: #not rollout_bool -> because we havent incremented the rollout id yet
                        if bool_rollout is not None and not bool_rollout[n]:
                            fig_name = os.path.join(process_path, "x_vizall_final_{}.png".format(step))
                            print(f"fig_name: {fig_name}")
                            self.plot_grid(
                                agent_cords=agent_cords,
                                state_mat=self.global_state_mat[n, :, :, :],
                                fig_name=fig_name,
                                agent_nums=range(self.num_agents),
                                target_pos=target_pos_n
                            )
                    elif bool_rollout is not None and bool_rollout[n]:
                        pass
                    else:
                        fig_name = os.path.join(process_path, "vizall_state_{}.png".format(step))
                        self.plot_grid(
                            agent_cords=agent_cords,
                            state_mat=self.global_state_mat[n, :, :, :],
                            fig_name=fig_name,
                            agent_nums=range(self.num_agents),
                            target_pos=target_pos_n
                        )
                else:
                    #individual plots
                    for i in range(self.num_agents):
                        if full_dones is not None and full_dones[n]: #not rollout_bool -> because we havent incremented the rollout id yet
                            if bool_rollout is not None and not bool_rollout[n]:
                                fig_name = os.path.join(process_path, "final_{}_{}.png".format(step, i))
                                print(f"fig_name: {fig_name}")
                                self.plot_grid(
                                    agent_cords=agent_cords,
                                    state_mat=self.global_state_mat[n, :, :, :],
                                    fig_name=fig_name,
                                    agent_nums=[i],
                                    target_pos=target_pos_n
                                )
                        elif bool_rollout is not None and bool_rollout[n]:
                            pass
                        else:
                            fig_name = os.path.join(process_path, "state_{}_{}.png".format(step, i))
                            self.plot_grid(
                                agent_cords=agent_cords,
                                state_mat=self.global_state_mat[n, :, :, :],
                                fig_name=fig_name,
                                agent_nums=[i],
                                target_pos=target_pos_n
                            )

        return None


    def format_target_pos(self, target_pos):

        """
        method to convert target_pos into an array
        """
        #the format for target_pos is a list of size (2), each list contains 2 arrays, corresponding to x and y coordinates
        #need to convert it to a list of lists, each list for an environment
        out_list = []
        for n in range(self.n_rollout_threads):
            out_target = []
            for n_obj in range(self.num_objects):
                target_row = target_pos[n_obj][0][n]
                target_col = target_pos[n_obj][1][n]
                out_tuple = (target_row, target_col)
                out_target.append(out_tuple)
            out_list.append(out_target)

        return out_list

    def get_coords_nagents(
            self,
            select_process
    ):

        """
        function to get coordinates from n agents
        :param state: np.array (number of processes X M), M-> total numnber of state variables
        :return list_of_cords: (list_of_tuple) - list of tuples
        """
        list_of_cords = []
        x_max = self.size[0]
        y_max = self.size[1]
        state = self.global_state

        for n in range(self.num_agents):
            select_state = state[select_process, :]  # select the parallel process we want to visualize
            idx_x1 = -(n + 1) * x_max - n * y_max
            idx_x2 = -n * x_max - n * y_max
            x_val = select_state[idx_x1:idx_x2] if n != 0 else select_state[idx_x1:]
            y_val = select_state[-(n + 1) * y_max - (n + 1) * x_max:-n * y_max - (n + 1) * x_max]
            x_ag, y_ag = np.where(x_val > 0)[0][0], np.where(y_val > 0)[0][0]
            list_of_cords.append((y_ag, x_ag))

        return list_of_cords[::-1]  # (row, col) format, [::-1] to match it up with the current cords

    def update_state_matrix(self, state_mat, agent_coords):

        for agent, loc in enumerate(agent_coords):
            row_loc = loc[0]
            col_loc = loc[1]
            state_mat[agent, row_loc, col_loc] += 1

        return state_mat

    def compute_final_reward(self, threshold_visits=1):
        """Method to compute reward based on

        """
        #coop_agents = [n for n in range(self.num_agents) if n not in self.rogue_agents]
        if self.rogue_agents is not None:
            coop_agents = [n for n in range(self.num_agents) if n not in self.rogue_agents]
        else:
            coop_agents = [n for n in range(self.num_agents)]
        rewards = [np.count_nonzero(self.global_state_mat[n, coop_agents, :, :]) for n in range(self.n_rollout_threads)]
        #compute joint state matrix for all cooperative agents
        coop_state_mat = np.sum(self.global_state_mat[:, coop_agents, :, :], axis=1)
        redundant_state_mat = coop_state_mat > threshold_visits
        adversarial_rewards = [np.count_nonzero(redundant_state_mat[n, :, :]) for n in
                           range(self.n_rollout_threads)]
        # adversarial_rewards =  [np.c
        #redundant_state_mat = self.global_state_mat > threshold_visits
        #adversarial_rewards =  [np.count_nonzero(redundant_state_mat[n, coop_agents, :, :]) for n in range(self.n_rollout_threads)]
        return np.asarray(rewards), np.asarray(adversarial_rewards)
        #return np.asarray(rewards)/(self.size[0]*self.size[1]), np.asarray(adversarial_rewards)/(self.size[0]*self.size[1])

    @staticmethod
    def compute_beta(
            t,
            threshold_t,
            max_episode_length,
            beta_0=1.0,
            beta_t = 0.01
    ):

        k = -np.log(beta_t/beta_0)

        return None

    #plot_Grid for visualization
    def plot_grid(
            self,
            x_delta=1,
            y_delta=1,
            fig_name="test.svg",
            agent_cords=None,
            state_mat=None,
            alpha_max=0.9,
            alpha_min=0.1,
            agent_nums=[0, 1],
            target_pos= None
    ):
        x_max = self.size[0]
        y_max = self.size[1]
        wall_mat = self.wall_mat
        treasure_cords = self.target_locs if target_pos is None else target_pos

        x_min, y_min = 0.0, 0.0
        n_x = int((x_max - x_min) / x_delta + 1)
        n_y = int((y_max - y_min) / y_delta + 1)

        xs = np.linspace(0, x_max, n_x)
        ys = np.linspace(0, y_max, n_y)
        ax = plt.gca()

        w, h = xs[1] - xs[0], ys[1] - ys[0]

        agent_colors = ["#008000", "#FFA500", "#0000FF", "#30D5C8"]

        for i, x in enumerate(xs[:-1]):
            for j, y in enumerate(ys[:-1]):
                if wall_mat is not None:
                    # add a wall matrix to add a darker shade to the wall patches
                    if wall_mat[j, i] == 1:
                        # ax.add_patch(Rectangle((x, y), w, h, fill=True, color='#000000'))
                        #print(f"Troubleshoot: {x}, {y_max}, {y}")
                        ax.add_patch(Rectangle((x, y_max - 1 - y), w, h, fill=True, color='#000000', alpha=1))
                elif i % 2 == j % 2:  # racing flag style
                    ax.add_patch(Rectangle((x, y), w, h, fill=True, color='#008610', alpha=.1))
                # add a wall matrix to add a darker shade to the wall patches

                if state_mat is not None:
                    # find the sum across all visits
                    #sum_visits = np.sum(state_mat)
                    #alpha_val = alpha_max * (state_mat / sum_visits)

                    for ag in agent_nums:
                        sum_visits = np.sum(state_mat[ag, :, :])
                        if state_mat[ag, j, i] > 0:
                            alpha_val = alpha_min + ((alpha_max - alpha_min) / alpha_max) * (
                                    state_mat[ag, :, :] / sum_visits)
                            # ax.add_patch(Rectangle((x, y_max - 1 - y), w, h, fill=True, color=agent_colors[ag], alpha=0.1))
                            ax.add_patch(Rectangle((x, y_max - 1 - y), w, h, fill=True, color=agent_colors[ag],
                                                   alpha=alpha_val[j, i]))
                            # ax.add_patch(Rectangle((x, y), w, h, fill=True, color=agent_colors[ag], alpha=.1))

        # grid lines
        for x in xs:
            plt.plot([x, x], [ys[0], ys[-1]], color='black', alpha=.33, linestyle=':')
        for y in ys:
            plt.plot([xs[0], xs[-1]], [y, y], color='black', alpha=.33, linestyle=':')

        # agent_markers = ['go', 'bo']
        agent_markers = ['o', 'o']

        if agent_cords is not None:
            # for k, loc in enumerate(agent_cords):
            for k in agent_nums:
                # plt.plot(loc[0], loc[1], markers[k], markersize=10)
                # (y_max - 1) because we want to match up with the map coords
                loc = agent_cords[k]
                plt.plot(loc[1] + 0.5 * x_delta, (y_max - 1 - loc[0]) + 0.5 * y_delta, marker="o",
                         color=agent_colors[k], markersize=14)

        if treasure_cords is not None and not self.explore_mode:
            if state_mat is not None:
                for k, loc in enumerate(treasure_cords):
                    # check if the treasure coordinates have been visited by the agents
                    for ag in range(state_mat.shape[0]):
                        if state_mat[ag, loc[0], loc[1]] > 0:
                            marker_sym = "m*"
                            break
                        else:
                            marker_sym = "r*"
                    plt.plot(loc[1] + 0.5 * x_delta, (y_max - 1 - loc[0]) + 0.5 * y_delta, marker_sym, markersize=14)

            else:
                marker_sym = "r*"
                for k, loc in enumerate(treasure_cords):
                    plt.plot(loc[1] + 0.5 * x_delta, (y_max - 1 - loc[0]) + 0.5 * y_delta, marker_sym, markersize=14)


        plt.axis('off')  # turning off the axis for gridworld
        plt.savefig(fig_name)
        plt.close()

        return None


#---------------wrapper for
def search_and_rescue(
        seed,
        rogue_agents,
        num_objects=2,
        env_type="gridworld",
        map_ind=None,
        comm_radius=20,
        rogue_reward_factor=1.0,
        task_config=1,
        size_L=20,
        size_W=20,
        need_get=False,
        stay_act=False,
        parallel_threads=True,
        random_target=False,
        random_epsilon=None,
        number_of_random_targets=None,
        n_rollout_threads=12,
        data_dir="magw_data",
        explore_mode=False
):
    """
    :param seed:
    """

    SRObj = SearchRescue(
        seed=seed,
        rogue_agents=rogue_agents,
        num_objects=num_objects,
        env_type=env_type,
        map_ind=map_ind,
        comm_radius=comm_radius,
        rogue_reward_factor=rogue_reward_factor,
        task_config=task_config,
        size_L=size_L,
        size_W=size_W,
        need_get=need_get,
        stay_act=stay_act,
        random_target=random_target,
        random_epsilon=random_epsilon,
        number_of_random_targets=number_of_random_targets,
        parallel_threads=parallel_threads,
        n_rollout_threads=n_rollout_threads,
        data_dir=data_dir,
        explore_mode=explore_mode
    )

    env = SRObj.make_parallel_env()

    return env, SRObj



