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
from common.dqn_model import DQN_model
from common.data.map import map_matrix
from common.Priority_Replay import Memory_PER


class Runner:
    def __init__(self, args, env):
        self.args = args
        self.env = env        
        # self.memory = Memory(memory_size=args.buffer_size, batch_size=args.batch_size)
        self.memory = Memory_PER(args)
        self.DQN_agent = DQN_model(args, s_dim=self.env.obs_dim, a_dim=self.env.act_dimension, a_num=self.env.act_num)     
        print(self.DQN_agent.dqn)
        summary(model=self.DQN_agent.dqn, input_size=(1, 5, 34), device="cpu") 
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
        decent_i_max = int(200)
        
        train_flag = False        
        
        # for i in range(int(round(self.args.max_episodes / 3, 0))):
        # 以decent_i_max为e降低的速度，设置从0到decent_i_max回合间，e从1降低到0
        for i in range(decent_i_max):
            # epsilon_decent.append((1 - (0.01 * i) ** 2) - (1 - (0.01 * (i + 1)) ** 2))
            epsilon_decent.append((1 - (i / decent_i_max) ** 2) - (1 - ((i + 1) / decent_i_max) ** 2))
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
            repeat_times = int(0)
            while True:
                with torch.no_grad():  # 节省计算量
                    action, epsilon_using = self.DQN_agent.e_greedy_action(state, goal, epsilon)
                    print("action: ", action)
                if self.env.EnvPlayer.get_action_effect(action) is True:
                    state_next, reward, _, done, info, path = self.env.step(episode_step, action, goal, self.args.w1,
                                                                   self.args.w2, self.args.w3, self.args.w4)
                    self.memory.store_transition(state, action, reward, state_next, done, goal)
                    repeat_times = int(0)
                else:
                    state_next, reward, _, done, info, path = self.env.step(episode_step, action, goal, self.args.w1,
                                                                   self.args.w2, self.args.w3, self.args.w4)
                    state_next = state  # 将没进入step的state赋给0矩阵state_next                    
                    # self.memory.store_transition(state, action, reward, state_next, done, goal)
                    repeat_times += 1
                # state = state_next
                
                k = int(4)     # between [4, 8]
                # goal_prime = self.get_goal_full(k)  # k个元素的列表，元素为1*34的np数组
                # goal_prime = self.get_goal_past(k, path)
                # goal_prime = self.get_goal_future(k, path) 
                goal_prime = self.get_goal_from_son(k, state)
                if repeat_times == 0:
                    for i in goal_prime:
                        reward_prime = self.env.EnvPlayer.get_reward_prime(action, goal_prime, self.args.w1, self.args.w2, 
                                                                        self.args.w3, self.args.w4)                    
                        self.memory.store_transition(state, action, reward_prime, state_next, done, i)
                    print('--------------HER is operated-------------------')
                state = state_next                
                
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
                    
                # 在同一位置连续原地踏步超过10次，结束回合，防止经验池中的优秀经验被覆盖
                if repeat_times >= 30 and train_flag is True:
                    print('失败 in step %d of episode %d' % (episode_step, episode))
                    break
                
                episode_step += 1
                
            # learn 
            if self.memory.current_size >= 20 * self.args.batch_size:   # 128 *
                # noise_decrease = True
                train_flag = True
                for _ in range(20):
                    # transition = self.memory.uniform_sample()
                    tree_index, transition, ISWeights = self.memory.sample(self.args.batch_size)  # PER
                    loss_step, td_error_abs = self.DQN_agent.train(transition, ISWeights)  # PER
                    self.memory.batch_update(tree_index, td_error_abs)  # PER
                    # save to tensor board
                    self.writer.add_scalar('loss/Q_loss', loss_step, updates)
                    self.writer.add_scalar('reward/step_reward', reward, updates)
                    # self.writer.add_scalar('loss/ISWeights', ISWeights, updates)
                    updates += 1
                    # save in .mat
                    loss_one_ep.append(loss_step)
                    # print('loss step: ', loss_step)                              

            if train_flag is False:
                epsilon = 1.0
            elif decent_i < decent_i_max:
                epsilon -= float(epsilon_decent[decent_i])          
                epsilon = max(epsilon, finial_epsilon)                    
                decent_i += 1
            else:
                epsilon = finial_epsilon        
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
        print('dqn loss update steps: %d' % self.DQN_agent.num_updates)
        print('dqn target policy update steps: %d' % self.DQN_agent.num_target_updates)
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
    
    def get_goal_from_son(self, k, state):
        cur = ((np.argwhere(state[0] == 1))[0].tolist())[0]
        goal_prime = []
        son_point = set()  # 不重复的集合
        son = np.argwhere(map_matrix[cur] == 1)        
        for i in son:
            son_id = i.tolist()[0]
            son_point.add(son_id)            
            son_son = np.argwhere(map_matrix[son_id] == 1)
            for j in son_son:
                son_point.add(j.tolist()[0])
        goal_prime_location = random.sample(son_point, k)
        for i in goal_prime_location:
            goal_i = self.env.EnvPlayer.generate_goal_array(i)
            goal_prime.append(goal_i)
        return goal_prime      
    