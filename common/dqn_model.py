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

    # 在store函数中将s|g,s`|g拼接，这样runner中可以节省一次拼接步骤，实际存储的还是(s,a,r,s`)形式。后续训练时也直接取出输入网络即可，不用再拼
    def store_transition(self, s, a, r, s_, done, goal):
        # transition = (s, a, r, s_, done, goal)
        state_goal = np.concatenate((s, goal), axis=0)
        statenext_goal = np.concatenate((s_, goal), axis=0)
        transition = (state_goal, a, r, statenext_goal)
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

        self.layers_cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 6), stride=(1, 2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 6), stride=(1, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Flatten()
        )
        
        with torch.no_grad():
            x = torch.randn((1, 1, int(s_dim[0] + 1), s_dim[1]))
            y = self.layers_cnn(x)
            # print(y.shape)
            in_features = y.shape[1]
        
        self.layers_linear = nn.Sequential(
            nn.Linear(in_features, a_num),
            nn.ReLU(),
            # nn.Linear(32, a_num),
            # nn.ReLU()       # TODO activation function
        ) 

    def forward(self, x): 
        x = self.layers_cnn(x)
        x = self.layers_linear(x)
        return x


class DQN_model:
    def __init__(self, args, s_dim, a_dim, a_num):
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
        self.num_target_updates = 0
        self.target_update_freq = args.target_update_freq   # default 200
        self.batch_size = args.batch_size

    def train(self, minibatch, ISWeights):
        # obtain minibatch
        state = []
        action = []
        reward = []
        state_next = []        
        for tt in minibatch:
            state.append(tt[0][np.newaxis, :])
            action.append(tt[1])
            reward.append(tt[2])
            state_next.append(tt[3][np.newaxis, :])
        state = torch.tensor(np.array(state), dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.int64)
        reward = torch.tensor(reward, dtype=torch.float32)
        state_next = torch.tensor(np.array(state_next), dtype=torch.float32)        
        
        print('训练了')
        Q_value = self.dqn.forward(state)  # 64,34
        action = torch.unsqueeze(action, dim=1)  # 64,1
        Q_value = Q_value.gather(1, action)  # Q(s, a)使Q_value不止与state有关，还与action有关,dim=?

        Q_next = self.dqn_target(state_next).detach()
        Q_next_max, idx = Q_next.max(1)  # greedy policy
        Q_target = reward + self.gamma * Q_next_max
        Q_target = torch.unsqueeze(Q_target, dim=1)  #      
                  
        # ISWeights是batch size*1数组，不能和mse过后的loss(数)相乘，因此要先两数组相乘再mse得到最终backward的loss
        ISWeights = torch.tensor(ISWeights, dtype=torch.float32)   # batch size*1
        td_error = Q_value - Q_target   # batch size*1
        loss = torch.mul(ISWeights, td_error ** 2).mean()  # 数；**2是对tensor中的元素进行运算
        
        # loss_fn = nn.MSELoss(reduction='mean')
        # loss = loss_fn(Q_value, Q_target)           
        # weighted_loss = torch.mul(loss, ISWeights) 
        
        self.dqn_optimizer.zero_grad()
        loss.backward()
        self.dqn_optimizer.step()

        loss_np = loss.data.numpy()
        # td_error_abs = abs(loss_np)
        td_error_abs = abs(td_error).detach().numpy()
         # 返回的td_error_abs后续要用作zip运算和循环，应该是一个n*1数组，并且需要从tensor转化为numpy
         
        self.num_updates += 1
        if self.num_updates % self.target_update_freq == 0:
            self.dqn_target.load_state_dict(self.dqn.state_dict())
            self.num_target_updates += 1
            print("total training steps: %d, update target network!" % self.num_updates)         
       
        return loss_np, td_error_abs

    def e_greedy_action(self, s, g, epsilon):        
        # s = torch.tensor(s, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # change np.array to Tensor
        
        # s = torch.tensor(s, dtype=torch.float32)  # invalid
        # g = torch.tensor(g, dtype=torch.float32)
        # state_goal = torch.cat((s, g), dim=1)
        
        state_goal = np.concatenate((s, g), axis=0)
        state_goal = torch.tensor(state_goal, dtype=torch.float32).unsqueeze(0)
        Q_value = self.dqn.forward(state_goal)
        if np.random.random() < epsilon:
            print('随机action')
            action = np.random.randint(0, self.a_num)
            return action, epsilon
        else:
            print('DQN action')
            action = torch.argmax(Q_value)
            action = action.item()
            return action, epsilon

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
