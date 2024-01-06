from collections import deque
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import os


class Memory:
    def __init__(self, memory_size, batch_size):
        memory_size = int(memory_size)
        self.memory_size = memory_size
        self.batch_size = int(batch_size)
        self.counter = 0
        self.current_size = 0
        self.memory_buffer = deque(maxlen=memory_size)

    def store_transition(self, s, a, r, s_):
        transition = (s, a, r, s_)
        self.memory_buffer.append(transition)
        self.counter += 1
        self.current_size = min(self.counter, self.memory_size)

    def uniform_sample(self):
        temp_buffer = []
        idx = np.random.randint(0, self.current_size, self.batch_size)
        for i in idx:
            temp_buffer.append(self.memory_buffer[i])
        return temp_buffer


class DQN_net(nn.Module):
    def __init__(self, s_dim, a_num):
        super(DQN_net, self).__init__()
        self.fc1 = nn.Linear(s_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        # self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, a_num)

        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x


class DQN_model:
    def __init__(self, args, s_dim, a_num, target_update_freq=300):
        self.args = args
        self.gamma = args.gamma
        self.a_num = a_num
        self.dqn = DQN_net(s_dim=s_dim, a_num=a_num)
        self.dqn_optimizer = torch.optim.Adam(self.dqn.parameters())
        self.dqn_target = DQN_net(s_dim=s_dim, a_num=a_num)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        step_size_up = int(args.max_episodes / 10)
        self.scheduler_lr = torch.optim.lr_scheduler.CyclicLR(self.dqn_optimizer,
                                                              base_lr=args.base_lrs,
                                                              max_lr=args.lr_DQN, step_size_up=step_size_up,
                                                              mode="triangular2", cycle_momentum=False)

        # reset number of update
        self.num_updates = 0
        self.target_update_freq = target_update_freq
        self.batch_size = args.batch_size

    def train(self, minibatch):
        # obtain minibatch
        state = []
        action = []
        reward = []
        state_next = []
        for tt in minibatch:
            state.append(tt[0])
            action.append(tt[1])
            reward.append(tt[2])
            state_next.append(tt[3])
        state = Variable(torch.FloatTensor(state)).type(dtype=torch.float32)
        action = Variable(torch.Tensor(action)).type(dtype=torch.int64)
        reward = Variable(torch.FloatTensor(reward)).type(dtype=torch.float32)
        state_next = Variable(torch.FloatTensor(state_next)).type(dtype=torch.float32)

        Q_value = self.dqn.forward(state)  # 64,34
        action = torch.unsqueeze(action, dim=1)  # 64,1
        Q_value = Q_value.gather(1, action)  # Q(s, a)使Q_value不止与state有关，还与action有关,dim=?

        Q_next = self.dqn_target(state_next).detach()
        Q_next_max, idx = Q_next.max(1)  # greedy policy
        Q_target = reward + self.gamma * Q_next_max
        Q_target = torch.unsqueeze(Q_target, dim=1)  #
        loss_fn = nn.MSELoss(reduction='mean')

        loss = loss_fn(Q_value, Q_target)
        self.dqn_optimizer.zero_grad()
        loss.backward()
        self.dqn_optimizer.step()

        loss_np = loss.data.numpy()
        self.num_updates += 1
        if self.num_updates % self.target_update_freq == 0:
            self.dqn_target.load_state_dict(self.dqn.state_dict())
            print("total training steps: %d, update target network!" % self.num_updates)
        return loss_np

    def e_greedy_action(self, s, epsilon):
        s = Variable(torch.from_numpy(s)).type(dtype=torch.float32)
        # s = Variable(torch.FloatTensor(s))      # 这两行都是要把nparray->tensor，
        Q_value = self.dqn.forward(s)
        if np.random.random() < epsilon:
            action = np.random.randint(0, self.a_num)
            return action, epsilon
        else:
            return torch.argmax(Q_value), epsilon  # todo:3选一

    def random_action(self):
        action = np.random.randint(0, self.a_num)
        return action

    def save_model(self, save_path, save_episode):
        model_path = save_path + '/net_params'
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.dqn.state_dict(), model_path +
                   '/dqn_params_ep%d.pkl' % save_episode)


if __name__ == '__main__':
    from common.arguments import get_args

    args = get_args()
    dqn = DQN_model(args, 10, 24, 3, 4, 2, 300)
    for i in range(24):
        a_id = dqn.index_get_actions(i)
        print(i, a_id)
