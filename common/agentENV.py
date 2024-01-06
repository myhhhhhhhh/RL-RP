import numpy as np
from math import exp
import random
from common.data.map import map_matrix
from common.data.map import dis_matrix
from common.data.map import map_point_num
from common.astar import Astar
from common.EV_model import EV_model


class RoutePlanning:
    def __init__(self):
        self.time_step = 1  # 仿真时间步
        self.obs_num = 5
        self.act_num = 1

        # self.args = args
        self.map = map_matrix
        self.start_id = 29
        self.end_id = 27  # todo,去哪填哪
        self.current_location = int(30)  # 初始化
        self.next_location = int(30)

        self.travel_time = 0.0  # 汽车行驶产生的状态变化
        self.travel_dis = 0.0
        self.astar_total = Astar(self.map, self.start_id, self.end_id)
        self.left_dis = self.astar_total.shortest_path()
        self.travel_dis_max = 30  # km

        self.done = False
        self.info = {}
        self.path = []  # 记录已经到达的点

    def reset(self):
        self.done = False
        self.current_location = int(30)
        self.next_location = int(30)
        self.travel_time = 0.0  # 汽车行驶产生的状态变化
        self.path = []

        obs = np.zeros(self.obs_num, dtype=np.float32)  # np.array
        # obs[0] = self.current_location / map_point_num  # todo:这样归一化是否妥当？
        # obs[0] = self.current_location
        # obs[1] = map_matrix[self.current_location]
        # obs[2] = dis_matrix[self.current_location]
        obs[0] = self.next_location
        obs[1] = map_matrix[self.next_location]  # todo:如何归一化?如何利用信息？
        obs[2] = dis_matrix[self.next_location]
        obs[3] = self.travel_dis / self.travel_dis_max
        obs[4] = self.left_dis / self.travel_dis_max
        return obs

    def execute(self, action):
        if action != 0:
            self.next_location = action  # 按照训练好的Q网络找出当前状态对应的下一个动作
            mappoint = self.map[self.current_location]  # current_location仅用于计算状态的更新
            self.travel_dis += dis_matrix[self.current_location][self.next_location]

            self.info.update({'travel_dis': self.travel_dis,
                              'current_location': self.next_location})

            obs = np.zeros(self.obs_num, dtype=np.float32)  # np.array
            obs[0] = self.next_location
            obs[1] = map_matrix[self.next_location]
            obs[2] = dis_matrix[self.next_location]
            obs[3] = self.travel_dis / self.travel_dis_max
            obs[4] = self.left_dis / self.travel_dis_max
        else:
            obs = np.zeros(self.obs_num, dtype=np.float32)
        return obs

    def get_reward(self, w1, w2, w3, w4):
        # w1 w2 w3 w4为权重系数，作用是归一化，均为正常数
        # 第一部分：基础行驶产生的代价
        reward = -w1 * self.travel_dis
        # 第四部分：路径规划相关奖惩
        # ①防止选择不可行动作
        if self.next_location == 0:
            reward -= 50
        else:
            # ②到达终点奖励
            if self.done is True:
                reward += 20
            # ③步数尽量少
            else:
                reward -= 0.1
        # ④防止走回头路
        if self.current_location in self.path:
            reward -= 0.1
        else:
            self.path.append(self.current_location)
        # ⑤通过A*,离终点越近给越大的奖励
        dis_left = self.calculate_travel_dis(self.current_location, self.end_id) / self.calculate_travel_dis(
            self.start_id, self.end_id)
        reward += 10 * (1 - dis_left)
        return float(reward)

    def get_info(self):
        return self.info

    def get_done(self):
        if self.current_location == self.end_id:
            self.done = True
        return self.done

    def calculate_travel_dis(self, start_id, end_id):
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
