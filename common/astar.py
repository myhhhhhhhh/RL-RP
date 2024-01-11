import numpy as np
import sys
import common.data.map as map


class Astar:
    def __init__(self, graph, start, goal):
        self.graph = graph  # 邻接表
        self.start = start  # 起点
        self.goal = goal  # 终点

        self.open_list = {start: 0}  # open 表,将起点放入 open_list 中
        self.closed_list = {}  # closed 表

        # self.open_list[start] = 0.0  # 将起点放入 open_list 中

        self.parent = {start: None}  # 存储节点的父子关系。键为子节点，值为父节点。方便做最后路径的回溯
        self.min_dis = 0  # 最短路径的长度

    def SelectPoint(self):
        # key_list = self.open_list.keys()
        min_distance = sys.maxsize
        min_node = -1
        for node, distance in self.open_list.items():
            if distance < min_distance:
                min_distance = distance
                min_node = node
        return min_distance, min_node

    def shortest_path(self):
        while True:
            if len(self.open_list) == 0:
                print('搜索失败， 结束！')
                break

            distance, min_node = self.SelectPoint()
            self.open_list.pop(min_node)  # 将其从 open_list 中去除
            self.closed_list[min_node] = distance  # 将节点加入 closed_list 中

            for node, weight in enumerate(self.graph[min_node]):  # 遍历当前节点的邻接节点
                if weight == 0:
                    continue
                if node not in self.closed_list.keys():  # 邻接节点不在 closed_list 中
                    if node in self.open_list.keys():  # 如果节点在 open_list 中
                        if weight + distance < self.open_list[node]:
                            self.open_list[node] = distance + weight  # 更新节点的值
                            self.parent[node] = min_node  # 更新继承关系
                    else:  # 如果节点不在 open_list 中
                        self.open_list[node] = distance + weight  # 计算节点的值，并加入 open_list 中
                        self.parent[node] = min_node  # 更新继承关系

            if min_node == self.goal:  # 如果节点为终点
                self.min_dis = distance
                break
                # return shortest_path[::-1], self.min_dis			# 返回最短路径和最短路径长度
        self.back_track()
        return self.min_dis

    def back_track(self):
        shortest_path = [self.goal]  # 记录从终点回溯的路径
        father_node = self.parent[self.goal]
        while father_node != self.start:
            shortest_path.append(father_node)
            father_node = self.parent[father_node]
        shortest_path.append(self.start)

        # print(shortest_path[::-1])  # 逆序
        # print('最佳路径的代价为：{}'.format(self.min_dis))
        # print('找到最短路径， 结束！')
        print('.')
        shortest_path.reverse()
        path = []
        for n in shortest_path:
            path.append(int(n))
        return path


if __name__ == '__main__':
    to_id_table_test = {
        0: [1, 3, 4],
        1: [0, 2, 3, 4],
        2: [1, 3, 4],
        3: [0, 1, 2],
        4: [0, 1, 2],
    }
    dis_table_test = {
        0: [6, 5, 7],
        1: [6, 4, 6, 2],
        2: [4, 8, 8],
        3: [5, 6, 8],
        4: [7, 2, 8],
    }

    # dis_matrix = np.full((5, 5), np.inf)
    dis_matrix = np.zeros((map.map_point_num, map.map_point_num))

    for point, connections in map.to_id_table.items():
        distances = map.dis_table[point]
        for i, connection in enumerate(connections):
            # 因为矩阵的索引是从0开始的，所以需要减1
            dis_matrix[point][connection] = distances[i]
    # np.fill_diagonal(dis_matrix, 0)
    print(dis_matrix)
    print(np.shape(dis_matrix)[0])
    start = 29
    goal = 27
    # 创建一个Astar对象
    astar = Astar(dis_matrix, start, goal)

    # 计算最短路径
    min_dis = astar.shortest_path()

    # 输出最短路径和路程
    print("最短路径：", astar.back_track())
    print("路程：", min_dis)
