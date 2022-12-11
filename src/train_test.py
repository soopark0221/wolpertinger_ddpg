#!/usr/bin/env python
# -*- coding: utf-8 -*-

def train(continuous, env, agent, max_episode, warmup, save_model_dir, max_episode_length, log, save_per_epochs, swag_start, swag=False, alg='ddpg', evaluate=False):
    agent.is_training = True
    step = episode = episode_steps = 0
    episode_reward = 0.
    s_t = None
    evaluate = evaluate
    logger = log['RS_log']
    eval_logger = log['RS_eval_log']
    while episode < max_episode:
        agent.is_training = True
        while True:
            if s_t is None:
                s_t = env.reset()
                agent.reset(s_t)

            # agent pick action ...
            # args.warmup: time without training but only filling the memory
            if isinstance(s_t, tuple):
                s_t = s_t[0]
            #if step <= warmup:
            if alg == 'swag':
                if step < warmup:
                    action = agent.random_action()
                elif episode < swag_start:
                    action = agent.select_action(s_t)
                else:
                    action = agent.select_swag_action(s_t)
                    refaction = agent.select_action(s_t)
            else:
                if step < warmup:
                    action = agent.random_action()
                else:
                    action = agent.select_action(s_t)

            # env response with next_observation, reward, terminate_info
            if not continuous:
                action = action.reshape(1,).astype(int)[0]
            s_t1, r_t, done, _, _ = env.step(action)
            if max_episode_length and episode_steps >= max_episode_length - 1:
                done = True

            # agent observe and update policy
            agent.observe(r_t, s_t1, done)
            if step > warmup:
                agent.update_policy(episode)

            # update
            step += 1
            episode_steps += 1
            episode_reward += r_t
            s_t = s_t1
            # s_t = deepcopy(s_t1)
            
            if done:  # end of an episode
                logger.info(
                    "Ep:{0} | R:{1:.4f}".format(episode, episode_reward)
                )

                agent.memory.append(
                    s_t,
                    agent.select_action(s_t),
                    0., True
                )

                # reset
                s_t = None
                episode_steps =  0
                episode_reward = 0.
                episode += 1
                # break to next episode
                break
        # [optional] save intermideate model every run through of 32 episodes
        if step > warmup and episode > 0 and episode % save_per_epochs == 0:
            agent.save_model(save_model_dir)
            logger.info("### Model Saved before Ep:{0} ###".format(episode))
            #eval code
            if evaluate == True and episode > swag_start:
                agent.is_training = False
                agent.eval()
                s_t = None
                agent.swag_sample_param(agent.swag_model, agent.actor_sample)
                for i in range(100):
                    while True:
                        if s_t is None:
                            s_t = env.reset()
                            agent.reset(s_t)
                        if swag:
                            action = agent.select_swag_action(s_t)
                        else:
                            action = agent.select_action(s_t)  # to do : multiple swag policy 
                        s_t, r_t, done, _, _= env.step(action)
                        episode_steps += 1
                        episode_reward += r_t
                        if max_episode_length and episode_steps >= max_episode_length - 1:
                            done = True
                        if done:  # end of an episode
                            eval_logger.info(
                                "Ep:{0} | R:{1:.4f}".format(i+1, episode_reward/(i+1))
                            )
                            s_t = None
                            break    # save swag params
                episode_steps = 0
                episode_reward = 0.
    if alg == 'ddpg' and swag == True:
        agent.save_swag_model(save_model_dir)
        logger.info("### SWAG Model Saved before Ep:{0} ###".format(episode))

def test(env, agent, model_path, test_episode, max_episode_length, logger, swag=False, alg='ddpg'):
    
    # load swag params
    if swag:
        agent.load_swag_weights(model_path)
    else:
        agent.load_weights(model_path)
    agent.is_training = False
    agent.eval()

    #select swag action
    if swag:
        policy = lambda x: agent.select_swag_action(x, decay_epsilon=False)
    else:
        policy = lambda x: agent.select_action(x, decay_epsilon=False)

    episode_steps = 0
    episode_reward = 0.
    s_t = None
    for i in range(test_episode):
        while True:
            if s_t is None:
                s_t = env.reset()
                agent.reset(s_t)

            action = policy(s_t)  # to do : multiple swag policy 
            s_t, r_t, done, _, _= env.step(action)
            episode_steps += 1
            episode_reward += r_t
            if max_episode_length and episode_steps >= max_episode_length - 1:
                done = True
            if done:  # end of an episode
                logger.info(
                    "Ep:{0} | R:{1:.4f}".format(i+1, episode_reward/(i+1))
                )
                s_t = None
                break