import numpy as np
import scipy.io as scio


class MapPoint:
    def __init__(self, point_id):
        self.point_id = point_id

    def get_travel_dis(self, next_id):
        distance = dis_matrix[self.point_id, next_id]
        return distance


to_id_table = {
    0: [1, 9],
    1: [0, 2, 8],
    2: [1, 3, 4],
    3: [2, 5, 7],
    4: [2, 5],
    5: [3, 4, 6],
    6: [5, 32],
    7: [3, 31, 32],
    8: [1, 16],
    9: [0, 10, 11],
    10: [9, 13],
    11: [9, 13, 14],
    12: [13, 15, 16],
    13: [10, 11, 12],
    14: [11, 15, 17],
    15: [12, 14, 18],
    16: [8, 12, 20],
    17: [14, 18, 26],
    18: [15, 17, 19],
    19: [18, 20, 27],
    20: [16, 19, 21],
    21: [20, 22, 28],
    22: [21, 23, 31],
    23: [22, 24],
    24: [23, 25, 29],
    25: [24, 29],
    26: [17, 27, 30],
    27: [19, 26, 28],
    28: [21, 27, 30],
    29: [24, 25],
    30: [26, 28],
    31: [7, 22, 33],
    32: [6, 7, 33],
    33: [31, 32]
}

dis_table = {
    0: [510, 391],
    1: [510, 125, 241],
    2: [125, 278, 154],
    3: [278, 175, 273],
    4: [154, 280],
    5: [175, 280, 271],
    6: [271, 364],
    7: [273, 697, 181],
    8: [241, 839],
    9: [391, 221, 77],
    10: [221, 78],
    11: [77, 242, 221],
    12: [51, 286, 328],
    13: [78, 221, 51],
    14: [242, 113, 195],
    15: [286, 113, 226],
    16: [839, 328, 278],
    17: [195, 156, 410],
    18: [226, 156, 259],
    19: [259, 94, 324],
    20: [278, 94, 96],
    21: [96, 228, 414],
    22: [338, 202, 232],
    23: [202, 123],
    24: [123, 202, 353],
    25: [202, 141],
    26: [410, 485, 761],
    27: [324, 79, 485],
    28: [414, 79, 113],
    29: [353, 141],
    30: [761, 113],
    31: [697, 232, 134],
    32: [364, 181, 750],
    33: [134, 750]
}

map_point_num = len(to_id_table)
# 创建一个34*34的零矩阵
map_matrix = np.zeros((34, 34), dtype=np.int16)
dis_matrix = np.zeros((34, 34), dtype=np.float16)

# 填入连接关系得到0和1组成的Map matrix
for point, connections in to_id_table.items():
    for connection in connections: 
        map_matrix[point][connection] = 1 

    # 更新距离矩阵
    distances = dis_table[point]
    for i, connection in enumerate(connections): 
        dis_matrix[point][connection] = distances[i]
        
dis_matrix_km = dis_matrix * 0.001  # 用于状态和goal的归一化


if __name__ == '__main__':
    print("map_matrix: ", map_matrix)
    # print("dis_matrix: ", dis_matrix)
    print(map_point_num)
    # print(map_matrix[18][26])
    # scio.savemat('./common/data/map_matrix.mat', mdict={'map_matrix': map_matrix})
    # scio.savemat('./common/data/dis_matrix.mat', mdict={'dis_matrix': dis_matrix})
    # print(dis_matrix_km)
