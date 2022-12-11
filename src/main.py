#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import logging
from train_test import train, test
import warnings
from arg_parser import init_parser
from setproctitle import setproctitle as ptitle
from normalized_env import NormalizedEnv
import gym
from tensorboardX import SummaryWriter

if __name__ == "__main__":
    ptitle('WOLP_DDPG')
    warnings.filterwarnings('ignore')
    parser = init_parser('WOLP_DDPG')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)[1:-1]

    from util import get_output_folder, setup_logger
    from wolp_agent import WolpertingerAgent
    from swag_agent import SWAGAgent
    from ddpg import DDPG
    if args.mode == 'train':
        args.save_model_dir = get_output_folder('../output', args.env)
    else :
        args.save_model_dir = '../output' + '/'+ args.env + '/' + args.load_model_dir
    env = gym.make(args.env)
    continuous = None
    try:
        # continuous action
        nb_states = env.observation_space.shape[0]
        nb_actions = env.action_space.shape[0]
        action_high = env.action_space.high
        action_low = env.action_space.low
        continuous = True
        env = NormalizedEnv(env)
    except IndexError:
        # discrete action for 1 dimension
        nb_states = env.observation_space.shape[0]
        nb_actions = 1  # the dimension of actions, usually it is 1. Depend on the environment.
        max_actions = env.action_space.n
        continuous = False

    if args.seed > 0:
        np.random.seed(args.seed)
        try:
            env.seed(args.seed)
        except:
            pass
    
    if args.alg == 'ddpg':
        agent_args = {
            'nb_states': nb_states,
            'nb_actions': nb_actions,
            'args': args,
        }
        agent = DDPG(**agent_args)
    elif args.alg == 'swag':
        agent_args = {
            'nb_states': nb_states,
            'nb_actions': nb_actions,
            'args': args,
        }
        agent = SWAGAgent(**agent_args)
    else:
        if continuous:
            agent_args = {
                'continuous': continuous,
                'max_actions': None,
                'action_low': action_low,
                'action_high': action_high,
                'nb_states': nb_states,
                'nb_actions': nb_actions,
                'args': args,
            }
        else:
            agent_args = {
                'continuous': continuous,
                'max_actions': max_actions,
                'action_low': None,
                'action_high': None,
                'nb_states': nb_states,
                'nb_actions': nb_actions,
                'args': args,
            }

        agent = WolpertingerAgent(**agent_args)

    if args.load:
        agent.load_weights(args.load_model_dir)

    if args.gpu_ids[0] >= 0 and args.gpu_nums > 0:
        agent.cuda_convert()

    # set logger, log args here
    log = {}
    print(args.save_model_dir)
    if args.mode == 'train':
        print(args.save_model_dir)
        setup_logger('RS_log', r'{}/RS_train_log'.format(args.save_model_dir))
        setup_logger('RS_eval_log', r'{}/RS_eval_log'.format(args.save_model_dir))
    elif args.mode == 'test':
        setup_logger('RS_log', r'{}/RS_test_log'.format(args.save_model_dir))
    else:
        raise RuntimeError('undefined mode {}'.format(args.mode))
    log['RS_log'] = logging.getLogger('RS_log')
    try:
        log['RS_eval_log'] = logging.getLogger('RS_eval_log')
    except:
        pass
    d_args = vars(args)
    d_args['max_actions'] = args.max_actions
    for key in agent_args.keys():
        if key == 'args':
            continue
        d_args[key] = agent_args[key]
    for k in d_args.keys():
        log['RS_log'].info('{0}: {1}'.format(k, d_args[k]))
        try:
            log['RS_eval_log'].info('{0}: {1}'.format(k, d_args[k]))
        except:
            pass

    if args.mode == 'train':

        train_args = {
            'continuous': continuous,
            'env': env,
            'agent': agent,
            'max_episode': args.max_episode,
            'warmup': args.warmup,
            'save_model_dir': args.save_model_dir,
            'max_episode_length': args.max_episode_length,
            'log': log, #['RS_log'],
            'save_per_epochs': args.save_per_epochs,
            'swag':args.swag,
            'swag_start':args.swag_start,
            'alg':args.alg,
        }

        train(**train_args)

    elif args.mode == 'test':

        test_args = {
            'env': env,
            'agent': agent,
            'model_path': args.save_model_dir,
            'test_episode': args.test_episode,
            'max_episode_length': args.max_episode_length,
            'logger': log['RS_log'],
            'swag':args.swag,
            'alg':args.alg,
        }

        test(**test_args)

    else:
        raise RuntimeError('undefined mode {}'.format(args.mode))
