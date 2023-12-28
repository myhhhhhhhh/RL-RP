import numpy as np
import sys
import time


class Astar:
    def __init__(self, graph, start, goal):
        self.graph = graph  # 邻接表
        self.start = start  # 起点
        self.goal = goal  # 终点

        self.open_list = {}  # open 表
        self.closed_list = {}  # closed 表

        self.open_list[start] = 0.0  # 将起点放入 open_list 中

        self.parent = {start: None}  # 存储节点的父子关系。键为子节点，值为父节点。方便做最后路径的回溯
        self.min_dis = 0  # 最短路径的长度

    def SelectPoint(self):
        key_list = self.open_list.keys()
        distance = sys.maxsize
        min_node = '0'
        for v in key_list:
            if self.open_list[v] < distance:
                distance = self.open_list[v]
                min_node = v
        return distance, min_node

    def \
            shortest_path(self):
        while True:
            if self.open_list is None:
                print('搜索失败， 结束！')
                break

            distance, min_node = self.SelectPoint()
            self.open_list.pop(min_node)  # 将其从 open_list 中去除
            self.closed_list[min_node] = distance  # 将节点加入 closed_list 中

            for node in self.graph[min_node].keys():  # 遍历当前节点的邻接节点
                if node not in self.closed_list.keys():  # 邻接节点不在 closed_list 中
                    if node in self.open_list.keys():  # 如果节点在 open_list 中
                        if self.graph[min_node][node] + distance < self.open_list[node]:
                            self.open_list[node] = distance + self.graph[min_node][node]  # 更新节点的值
                            self.parent[node] = min_node  # 更新继承关系
                    else:  # 如果节点不在 open_list 中
                        self.open_list[node] = distance + self.graph[min_node][node]  # 计算节点的值，并加入 open_list 中
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