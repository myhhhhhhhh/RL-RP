from common.utils import get_driving_cycle, get_acc_limit
from common.agentENV import RoutePlanning


class Env:
    """ environment for route planning"""

    def __init__(self, args):
        self.args = args
        self.EnvPlayer = RoutePlanning(args)
        self.obs_num = self.EnvPlayer.obs_num
        self.action_num = self.EnvPlayer.act_num  # 3
        self.act_dim = self.EnvPlayer.act_dim
        self.act_list = self.EnvPlayer.reset_act_list()

    def reset(self):
        obs = self.EnvPlayer.reset()
        self.act_list = self.EnvPlayer.reset_act_list()
        return obs

    def step(self, episode_step, actions, w1, w2, w3, w4):
        obs = self.EnvPlayer.execute(actions)
        self.act_list = self.EnvPlayer.reset_act_list()
        print('episode_step: %d, current_location: %d, SOC: %.6f'
              % (episode_step, self.EnvPlayer.current_location, self.EnvPlayer.SOC))
        print()
        # print('act0_list: ', self.act0_list)
        reward = self.EnvPlayer.get_reward(w1, w2, w3, w4)
        done = self.EnvPlayer.get_done()
        info = self.EnvPlayer.get_info()
        return obs, reward, done, info


def make_env(args):
    env = Env(args)
    args.obs_dim = env.obs_num
    args.action_dim = env.act_dim
    args.action_num = env.action_num
    return env, args
