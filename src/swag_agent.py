from ddpg import DDPG
import action_space
from util import *
import torch.nn as nn
import torch
from swag_misc import SWAG
import swag_utils
import time
criterion = nn.MSELoss()

criterion = nn.MSELoss()
class SWAGAgent(DDPG):

    def __init__(self, nb_states, nb_actions, args):
        super().__init__(args, nb_states, nb_actions)
        self.experiment = args.id
        self.swag_lr = args.swag_lr
        self.lr_init = args.lr_init
        self.swag_start = args.swag_start
        self.max_episode = args.max_episode
        self.swag = args.swag
        self.swag_model = SWAG(self.actor)
        self.eval_freq = args.eval_freq
        self.sample_freq = args.sample_freq

    def update_policy(self, episode):
        # Sample batch
        state_batch, action_batch, reward_batch, \
        next_state_batch, terminal_batch = self.memory.sample_and_split(self.batch_size)
        
        #next_state_batch = to_tensor(next_state_batch, volatile=True, gpu_used=self.gpu_used, gpu_0=self.gpu_ids[0])

        # Prepare for the target q batch
        next_q_values = self.critic_target([
            to_tensor(next_state_batch, volatile=True, gpu_used=self.gpu_used, gpu_0=self.gpu_ids[0]),
            self.actor_target(to_tensor(next_state_batch, volatile=True, gpu_used=self.gpu_used, gpu_0=self.gpu_ids[0])),
        ])
        next_q_values.volatile=False

        target_q_batch = to_tensor(reward_batch, gpu_used=self.gpu_used, gpu_0=self.gpu_ids[0]) + \
            self.gamma*to_tensor(terminal_batch.astype(np.float), gpu_used=self.gpu_used, gpu_0=self.gpu_ids[0])*next_q_values

        # Critic update
        self.critic.zero_grad()

        q_batch = self.critic([ to_tensor(state_batch, gpu_used=self.gpu_used, gpu_0=self.gpu_ids[0]), to_tensor(action_batch, gpu_used=self.gpu_used, gpu_0=self.gpu_ids[0]) ])
        
        value_loss = criterion(q_batch, target_q_batch)
        value_loss.backward()
        self.critic_optim.step()

        # Actor update
        self.actor.zero_grad()

        policy_loss = -self.critic([
            to_tensor(state_batch, gpu_used=self.gpu_used, gpu_0=self.gpu_ids[0]),
            self.actor(to_tensor(state_batch, gpu_used=self.gpu_used, gpu_0=self.gpu_ids[0]))
        ])

        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()

        # Target update
        soft_update(self.actor_target, self.actor, self.tau_update)
        soft_update(self.critic_target, self.critic, self.tau_update)
        # update lr
        lr = self.schedule(episode)
        self.adjust_learning_rate(self.actor_optim, lr)
        if self.swag and (episode+1) > self.swag_start:
            collect_start = time.process_time()
            self.swag_model.collect_model(self.actor)
            collect_end = time.process_time()
            # batch norm
            if episode == 0 or episode % self.eval_freq == self.eval_freq-1:
                eval_start = time.process_time()
                self.swag_eval(state_batch)
                eval_end = time.process_time()
            # sample and batch norm
            if episode % self.sample_freq == 0:
                swag_sample_start= time.process_time()
                self.swag_sample_param(state_batch)
                swag_sample_end= time.process_time()

    def cuda_convert(self):
        if len(self.gpu_ids) == 1:
            if self.gpu_ids[0] >= 0:
                with torch.cuda.device(self.gpu_ids[0]):
                    print('model cuda converted')
                    self.cuda()
        if len(self.gpu_ids) > 1:
            self.data_parallel()
            self.cuda()
            self.to_device()
            print('model cuda converted and paralleled')

    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

    def cuda(self):
        self.actor.cuda()
        self.actor_target.cuda()
        self.critic.cuda()
        self.critic_target.cuda()
        self.actor_sample.cuda()
        self.critic_sample.cuda()

    def data_parallel(self):
        self.actor = nn.DataParallel(self.actor, device_ids=self.gpu_ids)
        self.actor_target = nn.DataParallel(self.actor_target, device_ids=self.gpu_ids)
        self.critic = nn.DataParallel(self.critic, device_ids=self.gpu_ids)
        self.critic_target = nn.DataParallel(self.critic_target, device_ids=self.gpu_ids)
        self.actor_sample = nn.DataParallel(self.actor_sample, device_ids=self.gpu_ids)
        self.critic_sample = nn.DataParallel(self.critic_sample, device_ids=self.gpu_ids)


    def to_device(self):
        self.actor.to(torch.device('cuda:{}'.format(self.gpu_ids[0])))
        self.actor_target.to(torch.device('cuda:{}'.format(self.gpu_ids[0])))
        self.critic.to(torch.device('cuda:{}'.format(self.gpu_ids[0])))
        self.critic_target.to(torch.device('cuda:{}'.format(self.gpu_ids[0])))
        self.actor_sample.to(torch.device('cuda:{}'.format(self.gpu_ids[0])))
        self.critic_sample.to(torch.device('cuda:{}'.format(self.gpu_ids[0])))

    def observe(self, r_t, s_t1, done):
        if self.is_training:
            if isinstance(self.s_t, tuple):
                self.s_t = self.s_t[0]
            self.memory.append(self.s_t, self.a_t, r_t, done)
            self.s_t = s_t1

    def random_action(self):
        action = np.random.uniform(-1., 1., self.nb_actions)
        self.a_t = action
        return action

    def select_action(self, s_t, decay_epsilon=True):
        # proto action
        if isinstance(s_t, tuple):
            s_t = s_t[0]
        action = to_numpy(
            self.actor(to_tensor(np.array([s_t]), gpu_used=self.gpu_used, gpu_0=self.gpu_ids[0])),
            gpu_used=self.gpu_used
        ).squeeze(0)
        action += self.is_training * max(self.epsilon, 0) * self.random_process.sample()
        action = np.clip(action, -1., 1.)

        if decay_epsilon:
            self.epsilon -= self.depsilon
        
        self.a_t = action
        return action
        
    def select_swag_action(self, s_t, decay_epsilon=True, expl=False):
        # proto action
        if isinstance(s_t, tuple):
            s_t = s_t[0]
        action = to_numpy(
            self.actor_sample(to_tensor(np.array([s_t]), gpu_used=self.gpu_used, gpu_0=self.gpu_ids[0])),
            gpu_used=self.gpu_used
        ).squeeze(0)
        if expl:
            action += self.is_training * max(self.epsilon, 0) * self.random_process.sample()

        action = np.clip(action, -1., 1.)

        if decay_epsilon:
            self.epsilon -= self.depsilon
        
        self.a_t = action
        return action

    def reset(self, s_t):
        self.s_t = s_t
        self.random_process.reset_states()

    def load_weights(self, dir):
        if dir is None: return

        if self.gpu_used:
            # load all tensors to GPU (gpu_id)
            ml = lambda storage, loc: storage.cuda(self.gpu_ids)
        else:
            # load all tensors to CPU
            ml = lambda storage, loc: storage

        self.actor.load_state_dict(
            torch.load('output/{}/actor.pkl'.format(dir), map_location=ml)
        )

        self.critic.load_state_dict(
            torch.load('output/{}/critic.pkl'.format(dir), map_location=ml)
        )
        print('model weights loaded')


    def save_model(self,output):
        if len(self.gpu_ids) == 1 and self.gpu_ids[0] > 0:
            with torch.cuda.device(self.gpu_ids[0]):
                torch.save(
                    self.actor.state_dict(),
                    '{}/actor.pt'.format(output)
                )
                torch.save(
                    self.critic.state_dict(),
                    '{}/critic.pt'.format(output)
                )
        elif len(self.gpu_ids) > 1:
            torch.save(self.actor.module.state_dict(),
                       '{}/actor.pt'.format(output)
            )
            torch.save(self.actor.module.state_dict(),
                       '{}/critic.pt'.format(output)
                       )
        else:
            torch.save(
                self.actor.state_dict(),
                '{}/actor.pt'.format(output)
            )
            torch.save(
                self.critic.state_dict(),
                '{}/critic.pt'.format(output)
            )

    def seed(self,seed):
        torch.manual_seed(seed)
        if len(self.gpu_ids) > 0:
            torch.cuda.manual_seed_all(seed)

    def schedule(self, epoch):
        t = (epoch) / (self.swag_start if self.swag else self.max_episode)
        lr_ratio = self.swag_lr / self.lr_init if self.swag else 0.01
        if t <= 0.5:
            factor = 1.0
        elif t <= 0.9:
            factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
        else:
            factor = lr_ratio
        return self.lr_init * factor

    def adjust_learning_rate(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return 

    def swag_sample_param(self, state_batch):
        # swag bn 
        self.swag_model.sample(self.actor_sample, 0.5)
        swag_utils.bn_update(state_batch,self.swag_model)

    def swag_eval(self, state_batch):
        # swag bn 
        self.swag_model.set_swa(self.actor_sample)
        swag_utils.bn_update(state_batch,self.swag_model)