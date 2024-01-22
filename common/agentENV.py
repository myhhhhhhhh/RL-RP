import numpy as np
from math import exp
import random

import torch

from common.data.map import map_matrix
from common.data.map import dis_matrix
from common.data.map import dis_matrix_km
from common.data.map import map_point_num
from common.astar import Astar
from common.EV_model import EV_model


class RoutePlanning:
    def __init__(self, args):
        self.args = args
        self.time_step = 1  # 仿真时间步
        self.obs_dim = (4, 34)
        self.act_num = np.shape(map_matrix)[0]
        self.act_dimension = 1

        self.map = map_matrix
        self.start_id = self.args.start_id
        self.end_id = self.args.end_id  # todo,去哪填哪
        self.current_location = self.args.start_id  # 初始化
        self.next_location = self.args.start_id

        # self.travel_time = 0.0  # 汽车行驶产生的状态变化
        self.travel_dis = 0.0
        self.left_dis = self.calculate_travel_dis(self.next_location, self.end_id)
        self.travel_dis_max = self.calculate_travel_dis(self.start_id, self.end_id)

        self.done = False
        self.info = {}
        self.path = [self.start_id]  # 记录已经到达的点

    def reset(self):
        self.done = False
        self.current_location = self.args.start_id
        self.next_location = self.args.start_id
        self.travel_dis = 0.0  # 汽车行驶产生的状态变化
        self.path = [self.start_id]

        # obs = np.zeros(self.obs_dim, dtype=np.float32)  # np.array
        # obs[0] = self.current_location
        # # obs[1] = map_matrix[self.current_location]  # todo:如何归一化?→如何利用信息？矩阵拼接+特征提取
        # # obs[2] = dis_matrix[self.current_location]
        # obs[1] = self.travel_dis / self.travel_dis_max
        # obs[2] = self.left_dis / self.travel_dis_max
        obs_0 = self.current_location_vector(self.current_location)
        obs_1 = dis_matrix_km[self.current_location]       # change unit to km, normalization 
        obs_2 = self.current_location_vector(self.next_location)
        obs_3 = dis_matrix_km[self.next_location]
        obs_0 = obs_0[np.newaxis, :]
        obs_1 = obs_1[np.newaxis, :]
        obs_2 = obs_2[np.newaxis, :]
        obs_3 = obs_3[np.newaxis, :]
        obs = np.concatenate((obs_0, obs_1, obs_2, obs_3), axis=0)
        # obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        init_goal = np.random.randint(0,33)  # 设置一个goal的初始状态用于选择动作，后续更新方式更加重要，因此设置为随机        
        goal = self.generate_goal_array(init_goal)
        return obs, goal

    def execute(self, action):
        if self.get_action_effect(action) is True:
        # if action != 0:
            self.next_location = action  # 按照训练好的Q网络找出当前状态对应的下一个动作
            if self.next_location != self.path[-1]:
                self.path.append(self.next_location)
            # mappoint = self.map[self.current_location]  # current_location仅用于计算状态的更新
            self.travel_dis += dis_matrix[self.current_location][self.next_location] / 1000  # km

            self.info.update({'travel_dis': self.travel_dis,
                              'current_location': self.next_location})
            
            obs_0 = self.current_location_vector(self.current_location)
            obs_1 = dis_matrix_km[self.current_location]
            obs_2 = self.current_location_vector(self.next_location)
            obs_3 = dis_matrix_km[self.next_location]
            obs_0 = obs_0[np.newaxis, :]
            obs_1 = obs_1[np.newaxis, :]
            obs_2 = obs_2[np.newaxis, :]
            obs_3 = obs_3[np.newaxis, :]
            obs = np.concatenate((obs_0, obs_1, obs_2, obs_3), axis=0)
            # obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            self.current_location = self.next_location
        else:
            obs = np.zeros(self.obs_dim, dtype=np.float32)
            # obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            # 不对path进行更新
            self.info.update({'travel_dis': self.travel_dis,
                              'current_location': self.current_location})
            # step的过程中与goal无关，也无需返回，goal仅参与选择动作、计算r、存储环节
        return obs

    def get_reward(self, action, w1, w2, w3, w4):
        # w1 w2 w3 w4为权重系数，作用是归一化，均为正常数
        reward = 0.0
        # 第一部分：基础行驶产生的代价
        # reward = -  0.1 * self.travel_dis
        # 第四部分：路径规划相关奖惩
        # ①防止选择不可行动作
        if self.get_action_effect(action) is True:
            reward -= 1
        else:
            # ②到达终点奖励
            if self.done is True:
                reward += 50
                reward += 10 * (1 / self.travel_dis)
            # ③步数尽量少
            else:
                reward -= 0.1
        
        # ④防止走回头路
        if self.current_location in self.path:
            reward -= 0.1        
        
        # ⑤通过A*,离终点越近给越大的奖励
        dis_left = self.calculate_travel_dis(self.current_location, self.end_id) / self.calculate_travel_dis(
            self.start_id, self.end_id)
        reward += 15 * (1 - dis_left)
        return float(reward)
    
    def get_reward_prime(self, action, goal, w1, w2, w3, w4):        
        reward = 0.0
        goal_location = self.generate_goal_location(goal)
        
        if self.get_action_effect(action) is True:
            reward -= 1
        else:
            # ②到达终点奖励
            if action == goal_location:
                reward += 50
                if self.travel_dis != 0:
                    reward += 10 * (1 / self.travel_dis)
                else:
                    reward += 10 * (1 / (self.travel_dis + 1))
            # ③减少步数
            else:
                reward -= 0.1
        
        # ④防止走回头路
        if self.current_location in self.path:
            reward -= 0.1        
        
        # ⑤通过A*,离终点越近给越大的奖励
        if self.current_location != goal_location:
            dis_left = self.calculate_travel_dis(self.current_location, goal_location) / self.calculate_travel_dis(
                self.start_id, goal_location)
        else:
            dis_left = self.calculate_travel_dis(self.current_location, goal_location) / (self.calculate_travel_dis(
                self.start_id, goal_location) + 1)
        reward += 15 * (1 - dis_left)
        return float(reward)
    
    def get_info(self):
        return self.info

    def get_done(self):
        if self.current_location == self.end_id:
            self.done = True
        return self.done
    
    def get_path(self):
        return self.path

    # 由输入的action编号索引得到该动作对应的是1（可行）还是0（不可行）
    def get_action_effect(self, action):
        action_effect = map_matrix[self.current_location][action]
        return bool(action_effect)

    # 由位置数字0-map_point_num输出表示位置的数组[0,0,...1, 0,...,0]
    def current_location_vector(self, location):
        location_vector = np.zeros(self.act_num, dtype=int)
        location_vector[location] = 1
        return location_vector
    
    # 由2*map_point_num的goal数组提取出代表的位置编号
    def generate_goal_location(self, goal):
        index = np.argwhere(goal[0] == 1)
        return (index[0].tolist())[0]
    
    # 由位置编号产生一个2*map_point_num的goal数组
    def generate_goal_array(self, goal_location):
        goal_0 = self.current_location_vector(goal_location)
        goal_1 = dis_matrix_km[goal_location]
        goal_0 = goal_0[np.newaxis, :]
        goal_1 = goal_1[np.newaxis, :]
        goal = np.concatenate((goal_0, goal_1), axis=0)
        return goal
    
    @staticmethod
    def calculate_travel_dis(start_id, end_id):
        if start_id != end_id:
            astar = Astar(dis_matrix, start_id, end_id)
            dis = astar.shortest_path() / 1000  # km
            return dis
        else:
            return 0.0

    # 输入位置相关信息。输出限速信息
    def get_car_spd(self, current_location):
        # todo:如何简便地获取道路等级信息，需要再列一个表格？
        return 10


if __name__ == '__main__':
    # rp1 = RoutePlanning()
    # rp1.execute([2, 0, 0, 0])
    # print(rp1.travel_dis, rp1.travel_time, rp1.travel_cost)
    # print(rp1.calculate_travel_dis(29, 27))
    print(map_matrix)
    print(dis_matrix)
