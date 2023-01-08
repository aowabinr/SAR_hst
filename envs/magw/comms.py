import numpy as np
import gym
import torch
import os
import copy
import pygame
from collections import deque, OrderedDict
from itertools import product
from gym import Env, spaces
from gym.utils import seeding

import matplotlib.pyplot as plt


def get_count_based_novelties_comm(env, config, state_inds, agents_in_comm=None, device='cpu'):
    env_visit_counts = env.get_visit_counts()

    # samp_visit_counts[i,j,k] is # of times agent j has visited the state that agent k occupies at time i
    samp_visit_counts = np.concatenate(
        [np.concatenate(
            [env_visit_counts[j][tuple(zip(*state_inds[k]))].reshape(-1, 1, 1)
             for j in range(config.num_agents)], axis=1)
         for k in range(config.num_agents)], axis=2)


    novelties = np.power(np.maximum(samp_visit_counts, 1), -config.decay)


    return torch.tensor(novelties, device=device, dtype=torch.float32)



def get_intrinsic_rewards_with_comm(
        novelties, config,
        intr_rew_rms,
        infos=None,
        update_irrms=False,
        active_envs=None,
        device='cpu'
):

    #agents_comm = np.arange(config.num_agents) if agents_comm is None else agents_comm
    if update_irrms:
        assert active_envs is not None
    intr_rews = []

    for i, exp_type in enumerate(config.explr_types):
        if exp_type == 0:  # independent
            intr_rews.append([novelties[:, ai, ai] for ai in range(config.num_agents)])
        elif exp_type == 1:  # min
            if infos is None:
                intr_rews.append([novelties[:, :, ai].min(axis=1)[0] for ai in range(config.num_agents)])
            else:
                #intr_rews.append([novelties[:, agents_comm[ai], ai].min(axis=1)[0] for ai in range(config.num_agents)])
            #modifying code to get consensus from agents that are ONLY in range
                type_rews = []
                for ai in range(config.num_agents):
                    rollout_list = []
                    for n in range(novelties.shape[0]):
                        #print(infos[n]["agents_in_comm"])
                        agents_comm = np.asarray(infos[n]["agents_in_comm"][ai]) # - 1 #- 1 #agents in communication
                        rollout_list.append(novelties[n, agents_comm - 1, ai].min(axis=0)[0].tolist())
                    type_rews.append(torch.FloatTensor(rollout_list))
                    #type_rews.append(out_rew)
                intr_rews.append(type_rews)

        elif exp_type == 2:  # covering
            type_rews = []
            for ai in range(config.num_agents):
                rew = novelties[:, ai, ai] - novelties[:, :, ai].mean(axis=1)
                if infos is None:
                    rew[rew > 0.0] += novelties[rew > 0.0, :, ai].mean(axis=1)
                else:
                    for n in range(novelties.shape[0]):
                        agents_comm = np.asarray(infos[n]["agents_in_comm"][ai])
                        if rew[n] >  0.0:
                            rew[n] += novelties[n, agents_comm-1, ai].mean(axis=0)
                rew[rew < 0.0] = 0.0
                type_rews.append(rew)
            intr_rews.append(type_rews)
        elif exp_type == 3:  # burrowing
            type_rews = []
            for ai in range(config.num_agents):
                rew = novelties[:, ai, ai] - novelties[:, :, ai].mean(axis=1)
                rew[rew > 0.0] = 0.0
                if infos is None:
                    rew[rew < 0.0] += novelties[rew < 0.0, :, ai].mean(axis=1)
                else:
                    for n in range(novelties.shape[0]):
                        agents_comm = np.asarray(infos[n]["agents_in_comm"][ai])
                        if rew[n]< 0.0:
                            rew[n] += novelties[n, agents_comm-1, ai].mean(axis=0)
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


def get_agents_in_range(env, comm_range=(5,5)):

    """
    method to get
    :param env:
    :param comm_range:
    :return:
    """
    #map = env.m
    print(env.get_attr("m"))

    return None