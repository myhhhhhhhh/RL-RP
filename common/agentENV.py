import numpy as np
from math import exp
import random
from common.data.mapdata import MapData
from common.data.mapdata import map_point_num
from common.data.mapdata import directed_graph
from common.astar import Astar
from common.EV_model import EV_model


class RoutePlanning:
    def __init__(self, args):
        self.time_step = 1  # 仿真时间步
        self.obs_num = 5
        self.act_num = 1
        self.act_dim = 3

        self.args = args
        self.map = MapData
        self.start_id = 30
        self.end_id = 28  # todo,去哪填哪
        self.current_location = int(30)  # 初始化
        self.next_location = int(30)

        self.travel_time = 0.0  # 汽车行驶产生的状态变化
        self.travel_dis = 0.0
        self.travel_cost = 0.0
        self.SOC = 1  # SOC初始值，从仿真软件得到

        self.travel_dis_max = 30  # km
        self.travel_time_max = 30  # min
        self.travel_cost_max = 50  # ￥

        self.done = False
        self.info = {}
        self.path = []  # 记录已经到达的点

    def reset(self):
        self.done = False
        self.current_location = int(30)
        self.next_location = int(30)
        self.travel_time = 0.0  # 汽车行驶产生的状态变化
        self.travel_dis = 0.0
        self.travel_cost = 0.0
        self.SOC = 1
        self.path = []

        obs = np.zeros(self.obs_num, dtype=np.float32)  # np.array
        obs[0] = self.current_location / map_point_num
        obs[1] = self.travel_dis / self.travel_dis_max
        obs[2] = self.travel_time / self.travel_time_max
        obs[3] = self.travel_cost / self.travel_cost_max
        obs[4] = self.SOC
        return obs

    def reset_act_list(self):
        act_list = self.map[self.current_location].to_id
        return act_list

    def execute(self, actions):
        # 暂时确定为是否进入需要充电状态(need_charge)是人为规定的
        self.next_location = actions  # 由路径规划决定，在需要充电状态下可以不完全按照路径规划结果走

        mappoint = self.map[self.current_location]  # current_location仅用于计算状态的更新
        self.travel_dis += mappoint.get_travel_dis(self.next_location)  # km
        self.travel_time += mappoint.get_travel_time(self.next_location)  # min
        self.travel_cost += mappoint.get_travel_cost(self.next_location)
        _, SOC_consume_c_n, batt_info = self.calculate_batt_consume(self.current_location,
                                                                    self.next_location, 0.0, self.SOC)

        if SOC_consume_c_n > 0.1:
            print('—————SOC consume error, SOC_consume:%.4f ' % SOC_consume_c_n)
        self.SOC -= SOC_consume_c_n
        self.current_location = self.next_location

        self.info.update({'travel_dis': self.travel_dis,
                          'travel_time': self.travel_time,
                          'travel_cost': self.travel_cost,
                          'current_location': self.current_location,
                          'SOC': self.SOC
                          }
                         )
        self.info.update(batt_info)

        obs = np.zeros(self.obs_num, dtype=np.float32)  # np.array
        obs[0] = self.current_location / map_point_num
        obs[1] = self.travel_dis / self.travel_dis_max
        obs[2] = self.travel_time / self.travel_time_max
        obs[3] = self.travel_cost / self.travel_cost_max
        obs[4] = self.SOC
        return obs

    def get_reward(self, w1, w2, w3, w4):
        # w1 w2 w3 w4为权重系数，作用是归一化，均为正常数
        # 第一部分：基础行驶产生的代价
        reward = -(w1 * self.travel_dis + w2 * self.travel_time)
        # 第四部分：路径规划相关奖惩
        # ①防止选择不可行动作
        if self.next_location == 0:
            reward -= 10
        else:
            # ②到达终点奖励
            if self.done is True:
                reward += 20
            # ③步数尽量少
            else:
                reward -= 0.8
        # ④防止走回头路
        if self.current_location in self.path:
            reward -= 3
        else:
            self.path.append(self.current_location)
        # ⑤通过A*,离终点越近给越大的奖励
        dis_left = self.calculate_travel_dis(self.current_location, self.end_id)[0] / self.calculate_travel_dis(
            self.start_id, self.end_id)[0]
        reward += 10 * (1 - dis_left)
        return float(reward)

    def get_info(self):
        return self.info

    def get_done(self):
        if self.current_location == self.end_id:
            self.done = True
        return self.done

    def charge(self, charge_mode):
        # todo: dis变化暂不考虑，若需要则在charge station dic里先加上
        print('在节点%d充电' % self.charge_point)
        if charge_mode == 0:
            print('快充')
        else:
            print('慢充')
        # print('当前位置%d' % self.current_location)
        # time
        if charge_mode == 0:
            queue_time = self.map[self.current_location].fast_queue_time
            target_SOC = 0.8
            charge_speed = 1  # todo
        else:
            queue_time = self.map[self.current_location].slow_queue_time
            target_SOC = 0.4
            charge_speed = 0.1  # todo
        self.fill_SOC_time = (target_SOC - self.SOC) * charge_speed  # min
        self.charge_time += queue_time + self.fill_SOC_time  # 因充电引起的时间变化仅包括这两项，奖励函数中的delay等仅做判断用
        # cost
        charge_price = 1  # todo
        charge_cost = (target_SOC - self.SOC) * charge_price
        INC = 79600 / 525600  # income
        wait_cost = self.charge_time * INC
        # dis_cost
        self.charge_cost += charge_cost + wait_cost
        self.SOC = target_SOC

        # 不需要包含charge_dis,含义有区别

    # 输入self.SOC（目标SOC）, self.charge_time, self.charge_cost, self.charge_mode, 充电时长？←id,num,充电速率
    # 输出更新self.SOC, self.charge_time, self.charge_cost

    def calculate_travel_dis(self, start_id, end_id):
        # dis = Astar(directed_graph, str(start_id), str(end_id)).shortest_path()
        if start_id != end_id:
            astar = Astar(directed_graph, str(start_id), str(end_id))
            dis = astar.shortest_path() / 1000  # km
            # v = 10 * random.random()  # km/h
            v = 10
            time = dis / v * 60  # min
            return dis, time
        else:
            return 0.0, 0.0

    # 输入位置相关信息。输出限速信息
    def get_car_spd(self, current_location):
        # todo:如何简便地获取道路等级信息，需要再列一个表格？
        return 10

    def calculate_batt_consume(self, start_id, end_id, car_acc, initial_SOC):
        car_spd = self.get_car_spd(self.current_location)  # km/h
        time = self.calculate_travel_dis(start_id, end_id)[1]  # min
        batt_consume_per_second, SOC_delta, out_info = EV_model().run(car_spd, car_acc, initial_SOC)
        batt_consume = batt_consume_per_second * time * 60
        SOC_consume = SOC_delta * time * 60
        return batt_consume, SOC_consume, out_info


if __name__ == '__main__':
    rp1 = RoutePlanning()
    rp1.execute([2, 0, 0, 0])
    print(rp1.travel_dis, rp1.travel_time, rp1.travel_cost)
