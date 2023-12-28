import datetime
import os
import numpy as np
from numpy.random import normal  # normal distribution
import scipy.io as scio
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from common.memory import MemoryBuffer
from common.dqn_model import DQN_model, Memory
from common.agentENV import RoutePlanning


class Runner:
    def __init__(self, args, env):
        self.args = args
        self.env = env
        # self.buffer = MemoryBuffer(args)
        self.memory = Memory(memory_size=args.buffer_size, batch_size=args.batch_size)
        self.DQN_agent = DQN_model(args, s_dim=self.env.obs_num, a_dim=self.env.act_dim)
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
                                    .format(datetime.datetime.now().strftime("%m-%d_%H-%M"),
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
        # todo:args中修改保存路径/添加其他需要的信息，如起讫点
        average_reward = []  # average_reward of each episode
        DONE = {}
        # c_loss = []
        loss = []
        travel_dis = []
        travel_time = []
        travel_cost = []
        lr_recorder = {'lrcr': []}  # 动态变化的学习率
        updates = 0  # for tensorboard counter, 记录总共多少个time-step
        initial_epsilon = 1.0
        finial_epsilon = 0.2
        # epsilon_decent = (initial_epsilon - finial_epsilon) / 200
        epsilon_decent = []
        decent_i = int(0)
        for i in range(100):
            epsilon_decent.append((1 - (0.01 * i) ** 2) - (1 - (0.01 * (i + 1)) ** 2))
        epsilon = initial_epsilon
        for episode in tqdm(range(self.episode_num)):
            state = self.env.reset()  # reset the environment
            # if noise_decrease:
            #     noise_rate *= self.args.noise_discount_rate
            episode_reward = []
            loss_one_ep = []
            info = {}
            # data being saved in .mat
            episode_info = {'travel_dis': [], 'travel_time': [], 'travel_cost': [],  'SOC': [],
                            'current_location': [], 'P_mot': [], 'V_batt': [], 'I_batt': [], 'SOC_delta': []
                            }
            episode_step = 0
            while True:
                with torch.no_grad():  # 节省计算量
                    action_id, epsilon_using = self.DQN_agent.e_greedy_action(state, epsilon)
                actions = self.env.act_list[action_id]
                while actions == 0:
                    action_id = self.DQN_agent.random_action()
                    actions = self.env.act_list[action_id]

                state_next, reward, done, info = self.env.step(episode_step, actions, self.args.w1,
                                                               self.args.w2, self.args.w3, self.args.w4)
                # 把存储的三维动作转化成一维的动作序号0-23
                self.memory.store_transition(state, actions, reward, state_next)
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

                # learn
                if self.memory.current_size >= 10 * self.args.batch_size:
                    # noise_decrease = True
                    transition = self.memory.uniform_sample()
                    loss_step = self.DQN_agent.train(transition)
                    # save to tensorboard
                    self.writer.add_scalar('loss/critic', loss_step, updates)
                    self.writer.add_scalar('reward/step_reward', reward, updates)
                    updates += 1
                    # save in .mat
                    loss_one_ep.append(loss_step)

                episode_step += 1

            if episode <= 100:
                epsilon = 1
            elif 101<= episode <= 200:
                epsilon -= float(epsilon_decent[decent_i])
                decent_i += 1
                epsilon = max(epsilon, finial_epsilon)
            else:
                epsilon = 0.2

            # show episode data
            # 查看74行，可知episode_info['travel_cost']是一个[]，保存了一个episode的运行情况
            # 若charge_cost.append(episode_info['charge_cost'])，则charge_cost将是一个二维[[]]
            # 但是173行的语句只能保存一维[]，故无法保存数据
            # 可做如下修改，视具体情况而定
            travel_dis.append(episode_info['travel_dis'][-1])
            travel_time.append(episode_info['travel_time'][-1])
            travel_cost.append(episode_info['travel_cost'][-1])
            # lr scheduler
            lr0 = self.DQN_agent.scheduler_lr.get_lr()[0]
            lr_recorder['lrcr'].append(lr0)
            self.DQN_agent.scheduler_lr.step()
            # print
            soc = info['SOC']
            t_d = info['travel_dis']

            print('\nepi %d: SOC-end: %.4f, travel_dis: %.4f'
                  % (episode, soc, t_d))
            # save loss and reward on the average
            ep_r = np.mean(episode_reward)
            ep_c1 = np.mean(loss_one_ep)
            print('epi %d: ep_r: %.3f,  loss: %.4f,  lr: %.6f, epsilon: %.6f'
                  % (episode, ep_r, ep_c1, lr0, epsilon_using))
            average_reward.append(ep_r)
            loss.append(ep_c1)

        scio.savemat(self.save_path + '/reward.mat', mdict={'reward': average_reward})
        scio.savemat(self.save_path + '/critic_loss.mat', mdict={'loss': loss})
        scio.savemat(self.save_path + '/lr_recorder.mat', mdict=lr_recorder)
        scio.savemat(self.save_path + '/travel_dis.mat', mdict={'travel_dis': travel_dis})
        scio.savemat(self.save_path + '/travel_time.mat', mdict={'travel_time': travel_time})
        scio.savemat(self.save_path + '/travel_cost.mat', mdict={'travel_cost': travel_cost})
        scio.savemat(self.save_path + '/epsilon.mat', mdict={'epsilon': epsilon_using})

    def memory_info(self):
        print('\nbuffer counter:', self.memory.counter)
        print('buffer current size:', self.memory.current_size)
        print('replay ratio: %.3f' % (self.memory.counter / self.memory.current_size))
        print('arrive:', self.DONE)

    def get_index(self, actions_id):
        index = actions_id[2] + actions_id[1] * self.env.act2_dim + \
                actions_id[0] * self.env.act2_dim * self.env.act1_dim
        return index
