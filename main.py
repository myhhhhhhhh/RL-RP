import sys
import warnings
from common.runner import Runner
from common.arguments import get_args
from common.env import make_env
from common.evaluate import Evaluator
from common.utils import Logger

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    args = get_args()
    # args.save_dir = "./test3"
    args.save_dir = args.save_dir + "_" + args.DRL + "_" + args.MODE
    args.log_dir = args.log_dir + "_" + args.DRL + "_" + args.MODE
    args.eva_dir = args.eva_dir + "_" + args.DRL + "_" + args.MODE
    env, args = make_env(args)
    if args.evaluate:
        args.scenario_name += '_' + args.load_dir + '_' + args.load_scenario_name + '_%d' % args.load_episode
        sys.stdout = Logger(filepath=args.eva_dir + "/" + args.scenario_name + "/", filename='evaluate_log.log')
        print('max_episodes: ', args.evaluate_episode)
    else:
        args.scenario_name = args.scenario_name + "_w%d" % args.w_soc + '_' \
                             + args.file_v + "_LR" + str('{:.0e}'.format(args.lr_DQN))
        sys.stdout = Logger(filepath=args.save_dir + "/" + args.scenario_name + "/", filename='train_log.log')
        # print('\n weight coefficient: w_soc = %.1f' % args.w_soc)
        print('max_episodes: ', args.max_episodes)
    # print('cycle name: ', args.scenario_name)
    print('episode_steps: ', args.episode_steps)
    # print('abs_spd_MAX: %.3f m/s' % args.abs_spd_MAX)
    # print('abs_acc_MAX: %.3f m/s2' % args.abs_acc_MAX)
    print("DRL method: ", args.DRL)
    print('obs_dim: ', args.obs_dim)
    print('action_dim: ', args.act_num)
    # print('initial SOC: ', args.soc0)
    # print('SOC-MODE: ', args.MODE)

    if args.evaluate:
        print("\n-----Start evaluating!-----")
        evaluator = Evaluator(args, env)
        evaluator.evaluate()
        print("-----Evaluation is finished!-----")
        print('-----Data saved in: <%s>-----' % (args.eva_dir + "/" + args.scenario_name))
    else:
        print("\n-----Start training-----")
        runner = Runner(args, env)
        runner.set_seed() 
        if args.DRL == 'DQN':
            runner.run_DQN()
        # elif args.DRL == 'DDPG':
        #     runner.run_DDPG()
        else:
            print("\n[ERROR]: No such DRL method in this program! Exit Now!\n")
        runner.memory_info()
        print("-----Training is finished!-----")
        print('-----Data saved in: <%s>-----' % (args.save_dir + "/" + args.scenario_name))
