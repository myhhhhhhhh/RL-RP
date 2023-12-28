# import torch as T
import torch
import torch.nn.functional as F
import numpy as np
import torch.optim as opt
import torch.nn as nn
from common.td3_network import ActorNetwork, CriticNetwork
from common.td3_buffer import ReplayBuffer
from torch.autograd import Variable
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TD3:
    def __init__(self, args):
        self.args = args
        self.gamma = args.gamma
        self.tau = args.tau
        self.action_noise = args.action_noise
        self.policy_noise = args.policy_noise
        self.policy_noise_clip = args.policy_noise_clip  # 0.5
        self.update_time = args.update_time
        self.delay_time = args.delay_time
        self.lr_a = args.lr_a
        self.lr_c = args.lr_c
        # actor,64,64;critic,256,128
        """这里需要确认一下权重加载方式"""
        load_or_not = any([args.load_or_not, args.evaluate])
        # self.c_loss = 0
        # self.a_loss = 0

        # create the network
        self.actor = Actor(args)
        self.critic1 = Critic(args)
        self.critic2 = Critic(args)

        # build up the target network
        self.target_actor = Actor(args)
        self.target_critic1 = Critic(args)
        self.target_critic2 = Critic(args)

        # load the weights into the target networks
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        # create the optimizer
        self.actor_optimizer = opt.Adam(self.actor.parameters(), self.lr_a)
        self.critic1_optimizer = opt.Adam(self.critic1.parameters(), self.lr_c)
        self.critic2_optimizer = opt.Adam(self.critic2.parameters(), self.lr_c)
        # # learning rate scheduler
        # base_lr_a = 5e-5
        # base_lr_c = 5e-5
        # # base_lr_a_fixed = 5e-4  # 1e-2  # 5e-3  # 1e-3
        # # lr_a = 5e-4
        # # lr_c = 5e-3
        # lr_a = 1e-2
        # lr_c = 1e-2
        # step_size = int(args.max_episodes / 10)
        # self.scheduler_lr_a = opt.lr_scheduler.CyclicLR(self.actor_optimizer,
        #                                                 base_lr=base_lr_a, max_lr=lr_a, step_size_up=step_size,
        #                                                 mode="triangular2", cycle_momentum=False)
        # self.scheduler_lr_c = opt.lr_scheduler.CyclicLR(self.critic_optimizer,
        #                                                 base_lr=base_lr_c, max_lr=lr_c, step_size_up=step_size,
        #                                                 mode="triangular2", cycle_momentum=False)

        # create the direction for store the model
        if load_or_not is False:
            if not os.path.exists(self.args.save_dir):
                os.mkdir(self.args.save_dir)
            # path to save the model
            self.model_path = self.args.save_dir + '/' + self.args.scenario_name
            if not os.path.exists(self.model_path):
                os.mkdir(self.model_path)
        else:
            # load model
            load_path = self.args.load_dir + '/' + self.args.load_scenario_name + '/net_params'
            actor_pkl = '/actor_params_ep%d.pkl' % self.args.load_episode
            critic_pkl1 = '/critic1_params_ep%d.pkl' % self.args.load_episode
            critic_pkl2 = '/critic2_params_ep%d.pkl' % self.args.load_episode
            load_a = load_path + actor_pkl
            load_c1 = load_path + critic_pkl1
            load_c2 = load_path + critic_pkl2
            if os.path.exists(load_a):
                self.actor.load_state_dict(torch.load(load_a))
                self.critic1.load_state_dict(torch.load(load_c1))
                self.critic1.load_state_dict(torch.load(load_c2))
                print('Agent successfully loaded actor_network: {}'.format(load_a))
                print('Agent successfully loaded critic_network: {}'.format(load_c1))
                print('Agent successfully loaded critic_network: {}'.format(load_c2))

        ######
        # def __init__(self, alpha, beta, state_dim, action_dim, actor_fc1_dim, actor_fc2_dim,
        #              critic_fc1_dim, critic_fc2_dim, ckpt_dir, gamma=0.99, tau=0.005, action_noise=0.1,
        #              policy_noise=0.2, policy_noise_clip=0.5, delay_time=2, max_size=1000000,
        #              batch_size=256):
        #     self.gamma = gamma
        #     self.tau = tau
        #     self.action_noise = action_noise
        #     self.policy_noise = policy_noise
        #     self.policy_noise_clip = policy_noise_clip
        #     self.delay_time = delay_time
        #     self.update_time = 0
        #     self.checkpoint_dir = ckpt_dir
        #
        #     self.actor = ActorNetwork(alpha=alpha, state_dim=state_dim, action_dim=action_dim,
        #                               fc1_dim=actor_fc1_dim, fc2_dim=actor_fc2_dim)
        #     self.critic1 = CriticNetwork(beta=beta, state_dim=state_dim, action_dim=action_dim,
        #                                  fc1_dim=critic_fc1_dim, fc2_dim=critic_fc2_dim)
        #     self.critic2 = CriticNetwork(beta=beta, state_dim=state_dim, action_dim=action_dim,
        #                                  fc1_dim=critic_fc1_dim, fc2_dim=critic_fc2_dim)
        #
        #     self.target_actor = ActorNetwork(alpha=alpha, state_dim=state_dim, action_dim=action_dim,
        #                                      fc1_dim=actor_fc1_dim, fc2_dim=actor_fc2_dim)
        #     self.target_critic1 = CriticNetwork(beta=beta, state_dim=state_dim, action_dim=action_dim,
        #                                         fc1_dim=critic_fc1_dim, fc2_dim=critic_fc2_dim)
        #     self.target_critic2 = CriticNetwork(beta=beta, state_dim=state_dim, action_dim=action_dim,
        #                                         fc1_dim=critic_fc1_dim, fc2_dim=critic_fc2_dim)
        """这里也需要更改，与DDPG一致"""
        max_size, state_dim, action_dim, batch_size = \
            self.args.buffer_size, self.args.obs_dim, self.args.action_dim, self.args.batch_size
        self.memory = ReplayBuffer(max_size=max_size, state_dim=state_dim, action_dim=action_dim,
                                   batch_size=batch_size)

        self.update_network_parameters(tau=self.tau)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        for actor_params, target_actor_params in zip(self.actor.parameters(),
                                                     self.target_actor.parameters()):
            target_actor_params.data.copy_(tau * actor_params + (1 - tau) * target_actor_params)

        for critic1_params, target_critic1_params in zip(self.critic1.parameters(),
                                                         self.target_critic1.parameters()):
            target_critic1_params.data.copy_(tau * critic1_params + (1 - tau) * target_critic1_params)

        for critic2_params, target_critic2_params in zip(self.critic2.parameters(),
                                                         self.target_critic2.parameters()):
            target_critic2_params.data.copy_(tau * critic2_params + (1 - tau) * target_critic2_params)

    def remember(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def choose_action(self, observation, train=True):
        self.actor.eval()
        state = torch.tensor([observation], dtype=torch.float).to(device)
        action = self.actor.forward(state)

        if train:
            # exploration noise
            noise = torch.tensor(np.random.normal(loc=0.0, scale=self.action_noise),
                                 dtype=torch.float).to(device)
            action = torch.clamp(action + noise, -1, 1)
        self.actor.train()
        # return action.data.numpy()
        return action.squeeze().detach().cpu().numpy()

    def learn(self, transition):
        if not self.memory.ready():
            return
        state_batch = transition[0]
        action_batch = transition[1]
        reward_batch = transition[2]
        next_state_batch = transition[3]
        terminal_batch = transition[4]
        states = Variable(torch.from_numpy(state_batch))
        actions = Variable(torch.from_numpy(action_batch))
        rewards = Variable(torch.from_numpy(reward_batch))
        states_ = Variable(torch.from_numpy(next_state_batch))
        terminals = Variable(torch.from_numpy(terminal_batch))
        # states, actions, rewards, states_, terminals = self.memory.sample_buffer()
        states_tensor = torch.tensor(states, dtype=torch.float).to(device)
        actions_tensor = torch.tensor(actions, dtype=torch.float).to(device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float).to(device)
        next_states_tensor = torch.tensor(states_, dtype=torch.float).to(device)
        terminals_tensor = torch.tensor(terminals).to(device)

        with torch.no_grad():
            next_actions_tensor = self.target_actor.forward(next_states_tensor)
            action_noise = torch.tensor(np.random.normal(loc=0.0, scale=self.policy_noise),
                                        dtype=torch.float).to(device)
            # smooth noise
            action_noise = torch.clamp(action_noise, -self.policy_noise_clip, self.policy_noise_clip)
            next_actions_tensor = torch.clamp(next_actions_tensor + action_noise, -1, 1)
            q1_ = self.target_critic1.forward(next_states_tensor, next_actions_tensor).view(-1)
            q2_ = self.target_critic2.forward(next_states_tensor, next_actions_tensor).view(-1)
            q1_[terminals_tensor] = 0.0
            q2_[terminals_tensor] = 0.0
            critic_val = torch.min(q1_, q2_)
            target = rewards_tensor + self.gamma * critic_val
        q1 = self.critic1.forward(states_tensor, actions_tensor).view(-1)
        q2 = self.critic2.forward(states_tensor, actions_tensor).view(-1)

        critic1_loss = F.mse_loss(q1, target.detach())
        critic2_loss = F.mse_loss(q2, target.detach())
        critic_loss = critic1_loss + critic2_loss
        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()

        self.update_time += 1
        if self.update_time % self.delay_time != 0:
            return

        new_actions_tensor = self.actor.forward(states_tensor)
        q1 = self.critic1.forward(states_tensor, new_actions_tensor)
        actor_loss = -torch.mean(q1)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        actor_loss = -actor_loss.data
        critic_loss = critic_loss.data
        self.update_network_parameters()
        # print(critic_loss, actor_loss)
        return critic_loss, actor_loss
        # this part need to be coordinated

    def save_net(self, save_episode):
        model_path = os.path.join(self.args.save_dir, self.args.scenario_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, 'net_params')
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.actor.state_dict(), model_path +
                   '/actor_params_ep%d.pkl' % save_episode)
        torch.save(self.critic1.state_dict(), model_path +
                   '/critic_params_ep%d.pkl' % save_episode)
        torch.save(self.critic2.state_dict(), model_path +
                   '/critic_params_ep%d.pkl' % save_episode)

    # def save_models(self, episode):
    #     self.actor.save_checkpoint(self.checkpoint_dir + 'Actor/TD3_actor_{}.pth'.format(episode))
    #     print('Saving actor network successfully!')
    #     self.target_actor.save_checkpoint(self.checkpoint_dir +
    #                                       'Target_actor/TD3_target_actor_{}.pth'.format(episode))
    #     print('Saving target_actor network successfully!')
    #     self.critic1.save_checkpoint(self.checkpoint_dir + 'Critic1/TD3_critic1_{}.pth'.format(episode))
    #     print('Saving critic1 network successfully!')
    #     self.target_critic1.save_checkpoint(self.checkpoint_dir +
    #                                         'Target_critic1/TD3_target_critic1_{}.pth'.format(episode))
    #     print('Saving target critic1 network successfully!')
    #     self.critic2.save_checkpoint(self.checkpoint_dir + 'Critic2/TD3_critic2_{}.pth'.format(episode))
    #     print('Saving critic2 network successfully!')
    #     self.target_critic2.save_checkpoint(self.checkpoint_dir +
    #                                         'Target_critic2/TD3_target_critic2_{}.pth'.format(episode))
    #     print('Saving target critic2 network successfully!')

    def load_models(self, episode):
        self.actor.load_checkpoint(self.checkpoint_dir + 'Actor/TD3_actor_{}.pth'.format(episode))
        print('Loading actor network successfully!')
        self.target_actor.load_checkpoint(self.checkpoint_dir +
                                          'Target_actor/TD3_target_actor_{}.pth'.format(episode))
        print('Loading target_actor network successfully!')
        self.critic1.load_checkpoint(self.checkpoint_dir + 'Critic1/TD3_critic1_{}.pth'.format(episode))
        print('Loading critic1 network successfully!')
        self.target_critic1.load_checkpoint(self.checkpoint_dir +
                                            'Target_critic1/TD3_target_critic1_{}.pth'.format(episode))
        print('Loading target critic1 network successfully!')
        self.critic2.load_checkpoint(self.checkpoint_dir + 'Critic2/TD3_critic2_{}.pth'.format(episode))
        print('Loading critic2 network successfully!')
        self.target_critic2.load_checkpoint(self.checkpoint_dir +
                                            'Target_critic2/TD3_target_critic2_{}.pth'.format(episode))
        print('Loading target critic2 network successfully!')


# define the actor network
class Actor(nn.Module):
    def __init__(self, args):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(args.obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        # self.fc3 = nn.Linear(128, 64)
        self.action_out = nn.Linear(64, args.action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        action = torch.tanh(self.action_out(x))  # tanh value section: [-1, 1]
        return action

    def save_checkpoint(self, check_file):
        torch.save(self.state_dict(), check_file, _use_new_zipfile_serialization=False)


class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(args.obs_dim + args.action_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 128)
        self.q_out = nn.Linear(128, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)
        return q_value
