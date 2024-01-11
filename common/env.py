from common.agentENV import RoutePlanning


class Env:
    """ environment for route planning"""

    def __init__(self, args):
        self.args = args
        self.EnvPlayer = RoutePlanning(args)
        self.obs_dim = self.EnvPlayer.obs_dim
        self.act_num = self.EnvPlayer.act_num
        self.act_dimension = self.EnvPlayer.act_dimension

    def reset(self):
        obs = self.EnvPlayer.reset()
        # self.act_list = self.EnvPlayer.reset_act_list()
        return obs

    def step(self, episode_step, action, w1, w2, w3, w4):
        obs = self.EnvPlayer.execute(action)
        print('episode_step: %d, current_location: %d'
              % (episode_step, self.EnvPlayer.current_location))
        reward = self.EnvPlayer.get_reward(action, w1, w2, w3, w4)
        done = self.EnvPlayer.get_done()
        info = self.EnvPlayer.get_info()
        return obs, reward, done, info


def make_env(args):
    env = Env(args)
    args.obs_dim = env.obs_dim
    args.act_num = env.act_num
    args.act_dimension = env.act_dimension
    return env, args
