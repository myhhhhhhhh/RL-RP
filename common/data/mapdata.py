import numpy as np
import random


class MapPoint:
    def __init__(self, point_id):
        self.to_where_max = 4
        self.point_id = point_id
        # self.to_N = to_N
        self.to_id = np.zeros(self.to_where_max, np.int16)
        self.to_id_dis = np.zeros(self.to_id.shape, np.float16)
        self.to_id_time = np.zeros(self.to_id.shape, np.float16)
        self.to_id_cost = np.zeros(self.to_id.shape, np.float16)

        to_id = to_id_table[point_id]
        # self.to_N = len(to_id)
        if len(to_id) == self.to_where_max:
            self.to_id = np.array(to_id)
        else:
            for i in range(len(to_id)):
                self.to_id[i] = to_id[i]
        for i in range(len(self.to_id)):
            if self.to_id[i] == 0:
                self.to_id_dis[i] = -1  # can not go?表示无效
            else:
                to_id_dis_i = dis_table[point_id][i] / 1000
                self.to_id_dis[i] = to_id_dis_i  # km
                v = 10 * random.random()
                self.to_id_time[i] = dis_table[point_id][i] / v / 60  # todo, min, 从仿真软件导入, 暂未考虑交通量
                self.to_id_cost[i] = self.cost_calculation(to_id_dis_i, self.to_id_time[i])

    @staticmethod
    def cost_calculation(dis, time):  # todo
        dis = dis / 1000  # m
        Q = 250
        C = 500
        ui = (6.61 * 1e+4) / 2200
        cost1 = time * ui
        Q_batt_consume = 1
        cost2 = Q_batt_consume * 0.5 * time
        return cost1 + cost2

    def get_travel_dis(self, next_id):
        next_id_turple = np.where(self.to_id == next_id)
        # print(type(next_id_turple))
        # print(next_id_turple[0], type(next_id_turple[0]))
        # print(self.point_id, self.to_id)
        next_id_list = next_id_turple[0].tolist()
        next_id_index = next_id_list[0]
        # print(next_id_index, type(next_id_index))
        return self.to_id_dis[next_id_index]

    def get_travel_time(self, next_id):
        next_id_index = np.where(self.to_id == next_id)
        next_id = int(next_id_index[0])
        return self.to_id_time[next_id]

    def get_travel_cost(self, next_id):
        next_id_index = np.where(self.to_id == next_id)
        next_id = int(next_id_index[0])
        return self.to_id_cost[next_id]


to_id_table = {
    1: [2, 10],
    2: [1, 3, 9],
    3: [2, 4, 5],
    4: [3, 6, 8],
    5: [3, 6],
    6: [4, 5, 7],
    7: [6, 33],
    8: [4, 32, 33],
    9: [2, 17],
    10: [1, 11, 12],
    11: [10, 14],
    12: [10, 14, 15],
    13: [14, 16, 17],
    14: [11, 12, 13],
    15: [12, 16, 18],
    16: [13, 15, 19],
    17: [9, 13, 21],
    18: [15, 19, 27],
    19: [16, 18, 20],
    20: [19, 21, 28],
    21: [17, 20, 22],
    22: [21, 23, 29],
    23: [22, 24, 32],
    24: [23, 25],
    25: [24, 26, 30],
    26: [25, 30],
    27: [18, 28, 31],
    28: [20, 27, 29],
    29: [22, 28, 31],
    30: [25, 26],
    31: [27, 29],
    32: [8, 23, 34],
    33: [7, 8, 34],
    34: [32, 33]
}
dis_table = {
    1: [510, 391],
    2: [510, 125, 241],
    3: [125, 278, 154],
    4: [278, 175, 273],
    5: [154, 280],
    6: [175, 280, 271],
    7: [271, 364],
    8: [273, 697, 181],
    9: [241, 839],
    10: [391, 221, 77],
    11: [221, 78],
    12: [77, 242, 221],
    13: [51, 286, 328],
    14: [78, 221, 51],
    15: [242, 113, 195],
    16: [286, 113, 226],
    17: [839, 328, 278],
    18: [195, 156, 410],
    19: [226, 156, 259],
    20: [259, 94, 324],
    21: [278, 94, 96],
    22: [96, 228, 414],
    23: [338, 202, 232],
    24: [202, 123],
    25: [123, 202, 353],
    26: [202, 141],
    27: [410, 485, 761],
    28: [324, 79, 485],
    29: [414, 79, 113],
    30: [353, 141],
    31: [761, 113],
    32: [697, 232, 134],
    33: [364, 181, 750],
    34: [134, 750]
}


map_point_num = 34
MapData = {}
for i in range(map_point_num):
    map_point = MapPoint(i + 1)
    MapData.update({i + 1: map_point})

directed_graph = {'1': {'2': 510.34, '10': 391.5},
                  '2': {'1': 510.34, '9': 241.79, '3': 125.27},
                  '3': {'2': 125.27, '4': 278.71, '5': 154.71},
                  '4': {'3': 278.71, '6': 175.67, '8': 273.37},
                  '5': {'3': 154.71, '6': 280.42},
                  '6': {'4': 175.67, '5': 280.42, '7': 271.8},
                  '7': {'6': 271.8, '33': 364.02},
                  '8': {'4': 273.37, '32': 697.47, '33': 181.34},
                  '9': {'2': 241.79, '17': 839.68},
                  '10': {'1': 391.5, '12': 77.93, '11': 221.4},
                  '11': {'10': 221.4, '14': 78.87},
                  '12': {'10': 77.93, '15': 242.59, '14': 221.84},
                  '13': {'14': 51.54, '16': 286.52, '17': 328.82},
                  '14': {'11': 78.87, '12': 221.84, '13': 51.54},
                  '15': {'12': 242.59, '18': 195.08, '16': 113.03},
                  '16': {'15': 113.03, '19': 226.12, '13': 286.52},
                  '17': {'9': 839.68, '13': 328.82, '21': 278.27},
                  '18': {'15': 195.08, '19': 156.38, '27': 410.69},
                  '19': {'16': 226.12, '18': 156.38, '20': 259.53},
                  '20': {'19': 259.53, '28': 324.97, '21': 94.15},
                  '21': {'17': 278.27, '20': 94.15, '22': 96.65},
                  '22': {'21': 96.65, '29': 414.99, '23': 338.82},
                  '23': {'22': 338.82, '24': 202.74, '32': 232.06},
                  '24': {'23': 202.74, '32': 81.43, '25': 123.09},
                  '25': {'24': 123.09, '26': 202.94, '30': 353.94},
                  '26': {'30': 141.58, '25': 202.94},
                  '27': {'18': 410.69, '28': 485.68, '31': 761.62},
                  '28': {'20': 324.97, '29': 79.88, '27': 485.68},
                  '29': {'22': 414.99, '28': 79.88, '31': 113.95},
                  '30': {'26': 141.58, '25': 353.94},
                  '31': {'29': 113.95, '27': 761.62},
                  '32': {'8': 697.47, '23': 232.06, '24': 81.43, '34': 134.31},
                  '33': {'8': 181.34, '7': 364.02, '34': 750.5},
                  '34': {'33': 750.5, '32': 134.31}
                  }
if __name__ == '__main__':
    point1 = MapPoint(1)
    point2 = MapPoint(2)
    point3 = MapPoint(3)
    point4 = MapPoint(4)
    point5 = MapPoint(5)
    MapData = {1: point1, 2: point2, 3: point3, 4: point4, 5: point5}
    print(point1.to_id, point1.to_id_dis, point1.to_id_time, point1.to_id_cost)
    print(point2.to_id, point2.to_id_dis, point2.to_id_time, point2.to_id_cost)
    print(point3.to_id, point3.to_id_dis, point3.to_id_time, point3.to_id_cost)
    print(point1.to_id, point1.to_charge_station_id, point1.fast_num, point1.fast_queue_time, point1.slow_num,
          point1.slow_queue_time)

    # print(MapData)
    # print('6')
    #
    # start = '1'
    # goal = '3'
    # astar = Astar(directed_graph, start, goal)
    # astar.shortest_path()
