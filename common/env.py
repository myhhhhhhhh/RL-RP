from common.agentENV import RoutePlanning


class Env:
    """ environment for route planning"""

    def __init__(self, args):
        self.args = args
        self.EnvPlayer = RoutePlanning()
        self.obs_num = self.EnvPlayer.obs_num
        self.act_num = self.EnvPlayer.act_num  # 3

    def step(self, episode_step, action, w1, w2, w3, w4):
        obs = self.EnvPlayer.execute(action)
        print('episode_step: %d, current_location: %d'
              % (episode_step, self.EnvPlayer.current_location))
        reward = self.EnvPlayer.get_reward(w1, w2, w3, w4)
        done = self.EnvPlayer.get_done()
        info = self.EnvPlayer.get_info()
        return obs, reward, done, info


def make_env(args):
    env = Env(args)
    args.obs_dim = env.obs_num
    args.act_num = env.act_num
    return env, args
