import numpy as np

to_id_table = {
    0: [],
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
    0: [],
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

dis_matrix = np.full((35, 35), np.inf)
for point, connections in to_id_table.items():    
    distances = dis_table[point]
    for i, connection in enumerate(connections):
        # 因为矩阵的索引是从0开始的，所以需要减1
        dis_matrix[point-1][connection-1] = distances[i]
# 将对角线上的元素设置为0
np.fill_diagonal(dis_matrix, 0)
print(dis_matrix)

INF = float('inf')

def dijkstra(src, target):
    """
    src : 起点索引
    dist: 终点索引
    ret:  最短路径的长度
    """
    # 未到的点
    u = [i for i in range(35)]
    # 距离列表
    dist = dis_matrix[src][:]
    # 把起点去掉
    u.remove(src)

    # 用于记录最后更新结点
    last_update = [src if i != INF else -1 for i in dist]

    while True:
        idx = 0
        min_dist = INF

        # 找最近的点
        for i in range(35):
            if i in u and dist[i] < min_dist:
                min_dist = dist[i]
                idx = i

        # 从未到列表中去掉这个点
        u.remove(idx)

        # 更新dist（借助这个点连接的路径更新dist）
        for j in range(35):
            if j in u and dis_matrix[idx][j] + min_dist < dist[j]:
                dist[j] = dis_matrix[idx][j] + min_dist

                # 记录更新该结点的结点编号
                last_update[j] = idx
        if src == target:
            break
    # 输出从起点到终点的路径结点
    tmp = target
    path = []
    while tmp != src:
        path.append(tmp)
        tmp = last_update[tmp]
    path.append(src)
    print("->".join([str(i) for i in reversed(path)]))

    return dist[target]


dijkstra(30, 28)
