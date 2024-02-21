import argparse


def get_args():
    parser = argparse.ArgumentParser("Soft Actor Critic Implementation")
    # Environment
    parser.add_argument("--max_episodes", type=int, default=300, help="number of episodes ")
    parser.add_argument("--episode_steps", type=int, default=None, help="number of time steps in a single episode")
    # Core training parameters
    parser.add_argument("--lr_critic", type=float, default=0.0001, help="learning rate of critic")
    parser.add_argument("--lr_DQN", type=float, default=5e-3, help="learning rate of alpha")
    parser.add_argument("--base_lrs", type=float, default=5e-5, help="base_lr of CyclicLR")
    parser.add_argument("--gamma", type=float, default=0.9, help="discount factor")
    parser.add_argument("--target_update_freq", type=int, default=200, help="dqn target_update_freq")
    # for DDPG
    # parser.add_argument("--noise_rate", type=float, default=0.1,
    #                     help="initial noise rate for sampling from a standard normal distribution ")
    # parser.add_argument("--noise_discount_rate", type=float, default=0.999)
    # memory buffer
    parser.add_argument("--buffer_size", type=int, default=int(5000),    # DDPG: 1e4 for 200 ep,  3e4 for 500 ep
                        help="number of transitions can be stored in buffer")
    parser.add_argument("--batch_size", type=int, default=128, help="number of episodes to optimize at the same time")
    # random seeds
    parser.add_argument("--seed", type=int, default=1)
    # device
    parser.add_argument("--cuda", type=bool, default=True, help="True for GPU, False for CPU")
    # method
    parser.add_argument("--DRL", type=str, default='DQN', help="DQN or D3QN")
    # about EMS, weight coefficient
    parser.add_argument("--w_soc", type=float, default=50, help="weight coefficient for SOC reward")
    parser.add_argument("--soc0", type=float, default=0.2, help="initial value of SOC")
    parser.add_argument("--MODE", type=str, default='CD', help="CS or CD, charge-sustain or charge-depletion")
    # save model under training     # Standard_ChinaCity  Standard_IM240
    parser.add_argument("--scenario_name", type=str, default="myh",
                        help="name of driving cycle data")
    # CLTC_P  Standard_ChinaCity  # validation: CTUDC_WVUSUB_HWFET
    parser.add_argument("--save_dir", type=str, default="./test1", help="directory in which saves training data and "
                                                                        "model")
    parser.add_argument("--log_dir", type=str, default="./logs1", help="directory in which saves logs")
    parser.add_argument("--file_v", type=str, default='v1', help="每次训练都须重新指定")
    # load learned model to train new model or evaluate
    parser.add_argument("--load_or_not", type=bool, default=False)
    parser.add_argument("--load_episode", type=int, default=189)
    parser.add_argument("--load_scenario_name", type=str, default="CLTC_P_w100.0_v2_LR0.001")
    parser.add_argument("--load_dir", type=str, default="./test1_SAC_CD")
    # evaluate
    parser.add_argument("--evaluate", type=bool, default=False)
    parser.add_argument("--evaluate_episode", type=int, default=1)
    parser.add_argument("--eva_dir", type=str, default="./eva")
    # reward
    parser.add_argument("--w1", type=float, default=0.1)  # 0.1rmb per km
    parser.add_argument("--w2", type=float, default=690/5256)  # INC/T1 = 69000/(365*25*60),约0.13
    parser.add_argument("--w3", type=int, default=1)  # rmb
    parser.add_argument("--w4", type=float, default=0.5)
    # start & end
    parser.add_argument("--start_id", type=int, default=29)
    parser.add_argument("--end_id", type=int, default=27)
    # all above
    args = parser.parse_args()
    return args
    