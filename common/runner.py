import datetime
import os
import sys
import numpy as np
from numpy.random import normal  # normal distribution
import random
import scipy.io as scio
import torch
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from tqdm import tqdm
# sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
# print(sys.path)
from common.memory import MemoryBuffer
from common.dqn_model import DQN_model, Memory
from common.agentENV import RoutePlanning


class Runner:
    def __init__(self, args, env):
        self.args = args
        self.env = env
        # self.buffer = MemoryBuffer(args)
        self.memory = Memory(memory_size=args.buffer_size, batch_size=args.batch_size)
        self.DQN_agent = DQN_model(args, s_dim=self.env.obs_dim, a_dim=self.env.act_dimension, a_num=self.env.act_num)     
        print(self.DQN_agent.dqn)
        summary(model=self.DQN_agent.dqn, input_size=(1, 6, 34), device="cpu") 
        # configuration
        self.episode_num = args.max_episodes
        self.episode_step = args.episode_steps
        self.save_path = self.args.save_dir + '/' + self.args.scenario_name
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.save_path_episode = self.save_path + '/episode_data'
        if not os.path.exists(self.save_path_episode):
            os.makedirs(self.save_path_episode)
        # tensorboard
        fileinfo = args.scenario_name
        self.writer = SummaryWriter(args.log_dir + '/{}_{}_{}_seed{}'
                                    .format(datetime.datetime.now().strftime("%m_%d_%H_%M"),
                                            fileinfo, self.args.DRL, self.args.seed))
        self.DONE = {}

    def set_seed(self):
        seed = self.args.seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        print("\n Random seeds have been set to %d !\n" % seed)

    def run_DQN(self):
        # save_path = str(self.args.save_dir) + "_DQN_" + "_LR" + str(self.args.lr_DQN)        
        average_reward = []  # average_reward of each episode
        # c_loss = []
        loss = []
        travel_dis = []
        # travel_time = []
        # travel_cost = []
        lr_recorder = {'lrcr': []}  # 动态变化的学习率
        epsilon_list = []
        updates = 0  # for tensorboard counter, 记录总共多少个time-step
        initial_epsilon = 1.0
        finial_epsilon = 0.2
        # epsilon_decent = (initial_epsilon - finial_epsilon) / 200
        epsilon_decent = []
        decent_i = int(0)
        
        train_flag = False 
        
        for i in range(int(round(self.args.max_episodes / 3, 0))):
            epsilon_decent.append((1 - (0.01 * i) ** 2) - (1 - (0.01 * (i + 1)) ** 2))
        epsilon = initial_epsilon
        
        for episode in tqdm(range(self.episode_num)):
            state, goal = self.env.reset()  # reset the environment
            # if noise_decrease:
            #     noise_rate *= self.args.noise_discount_rate
            episode_reward = []
            loss_one_ep = []
            info = {}
            # data being saved in .mat
            episode_info = {'travel_dis': [], 'current_location': []}
            episode_step = 0
            while True:
                with torch.no_grad():  # 节省计算量
                    action, epsilon_using = self.DQN_agent.e_greedy_action(state, goal, epsilon)
                    print("action: ", action)
                if self.env.EnvPlayer.get_action_effect(action) is True:
                    state_next, reward, _, done, info, path = self.env.step(episode_step, action, goal, self.args.w1,
                                                                   self.args.w2, self.args.w3, self.args.w4)
                    self.memory.store_transition(state, action, reward, state_next, done, goal)
                else:
                    state_next, reward, _, done, info, path = self.env.step(episode_step, action, goal, self.args.w1,
                                                                   self.args.w2, self.args.w3, self.args.w4)
                    state_next = state  # 将没进入step的state赋给0矩阵state_next
                    self.memory.store_transition(state, action, reward, state_next, done, goal)
                state = state_next
                
                k = int(4)     # between [4, 8]
                # goal_prime = self.get_goal_full(k)  # k个元素的列表，元素为2*34的np数组
                # goal_prime = self.get_goal_past(k, path)
                goal_prime = self.get_goal_future(k, path)  
                for i in goal_prime:
                    reward_prime = self.env.EnvPlayer.get_reward_prime(action, goal_prime, self.args.w1, self.args.w2, 
                                                                       self.args.w3, self.args.w4)
                    self.memory.store_transition(state, action, reward_prime, state_next, done, i)
                    # print('--------------HER is operated-------------------')


                # save data
                for key in episode_info.keys():
                    episode_info[key].append(info[key])
                episode_reward.append(reward)
                # save interaction data in .mat
                if self.episode_num - episode <= 200 and done is True:
                    # save network parameters
                    self.DQN_agent.save_model(self.save_path, episode)
                    # save all data in one episode
                    datadir = self.save_path_episode + '/data_ep%d.mat' % episode
                    scio.savemat(datadir, mdict=episode_info)

                if done:  # 把到达终点的情况保存下来
                    print('到了 in step %d of episode %d' % (episode_step, episode))
                    self.DONE.update({episode: episode_step})
                    break  # 要结束当前循环
                    # 此处若done, 则结束当前while循环，进入下一个episode; 此时应该先保存交互数据。
                    # 故调整代码顺序，将保存数据的代码放到done前面，将learn的代码放在done的后面

                # learn
                if episode >= 1:
                # if self.memory.current_size >= 20 * self.args.batch_size:
                    # noise_decrease = True
                    train_flag = True
                    transition = self.memory.uniform_sample()
                    loss_step = self.DQN_agent.train(transition)
                    # save to tensor board
                    self.writer.add_scalar('loss/Q_loss', loss_step, updates)
                    self.writer.add_scalar('reward/step_reward', reward, updates)
                    updates += 1
                    # save in .mat
                    loss_one_ep.append(loss_step)
                    # print('loss step: ', loss_step)

                episode_step += 1

            if episode <= self.args.max_episodes / 3:
                epsilon = 1.0
            elif self.args.max_episodes / 3 < episode <= self.args.max_episodes / 3 * 2:
                epsilon -= float(epsilon_decent[decent_i])
                decent_i += 1
                epsilon = max(epsilon, finial_epsilon)
            else:
                epsilon = 0.2
            epsilon_list.append(epsilon)
            
            # show episode data
            # 查看74行，可知episode_info['travel_cost']是一个[]，保存了一个episode的运行情况
            # 若charge_cost.append(episode_info['charge_cost'])，则charge_cost将是一个二维[[]]
            # 但是173行的语句只能保存一维[]，故无法保存数据
            # 可做如下修改，视具体情况而定
            travel_dis.append(episode_info['travel_dis'][-1])
            # travel_time.append(episode_info['travel_time'][-1])
            # travel_cost.append(episode_info['travel_cost'][-1])
            
            # lr scheduler
            lr0 = self.DQN_agent.scheduler_lr.get_lr()[0]
            lr_recorder['lrcr'].append(lr0)
            self.DQN_agent.scheduler_lr.step()

            # print
            # soc = info['SOC']
            t_d = info['travel_dis']
            print('\nepi %d: travel_dis: %.4f'
                  % (episode, t_d))

            # save loss and reward on the average
            ep_r = np.mean(episode_reward)
            ep_c1 = np.mean(loss_one_ep)
            print('epi %d: ep_r: %.3f,  loss: %.4f,  lr: %.6f, epsilon: %.6f'
                  % (episode, ep_r, ep_c1, lr0, epsilon_using))
            average_reward.append(ep_r)
            loss.append(ep_c1)

        scio.savemat(self.save_path + '/ep_reward.mat', mdict={'ep_reward': average_reward})
        scio.savemat(self.save_path + '/Q_loss.mat', mdict={'Q_loss': loss})
        scio.savemat(self.save_path + '/lr_recorder.mat', mdict=lr_recorder)
        scio.savemat(self.save_path + '/travel_dis.mat', mdict={'travel_dis': travel_dis})
        # scio.savemat(self.save_path + '/travel_time.mat', mdict={'travel_time': travel_time})
        # scio.savemat(self.save_path + '/travel_cost.mat', mdict={'travel_cost': travel_cost})
        scio.savemat(self.save_path + '/epsilon.mat', mdict={'epsilon': epsilon_list})

    def memory_info(self):
        print('\nbuffer counter:', self.memory.counter)
        print('buffer current size:', self.memory.current_size)
        print('replay ratio: %.3f' % (self.memory.counter / self.memory.current_size))
        print('arrive:', self.DONE)
        
    def get_goal_full(self, k):
        map_point = []
        goal_prime = []
        for i in range(34):
            map_point.append(i)
        goal_prime_location = random.sample(map_point, k)
        for i in goal_prime_location:
            goal_i = self.env.EnvPlayer.generate_goal_array(i)
            goal_prime.append(goal_i)
        return goal_prime
        
    def get_goal_past(self, k, path):
        goal_prime = []        
        goal_prime_location = random.sample(path, k)
        for i in goal_prime_location:
            goal_i = self.env.EnvPlayer.generate_goal_array(i)
            goal_prime.append(goal_i)
        return goal_prime      
        
    def get_goal_future(self, k, path):        
        map_point = []
        goal_prime = []
        for i in range(34):
            map_point.append(i)
        future_point = list(set(map_point) - set(path))
        kk = min(k, len(future_point))      # prevent sample error 
        goal_prime_location = random.sample(future_point, kk)
        for i in goal_prime_location:
            goal_i = self.env.EnvPlayer.generate_goal_array(i)
            goal_prime.append(goal_i)
        return goal_prime      
        
