import os
import sys
import gym
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

from DDQN.ddqn import DDQN

from keras.backend.tensorflow_backend import set_session
from keras.utils import to_categorical

from utils.networks import get_session

import gym_stage

gym.logger.set_level(40)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def parse_args(args):
    """ Parse arguments from command line input
    """
    parser = argparse.ArgumentParser(description='Training parameters')
    #
    parser.add_argument('--type', type=str, default='DDQN',help="Algorithm to train from {A2C, A3C, DDQN, DDPG}")
    # parser.add_argument('--is_atari', dest='is_atari', action='store_true', help="Atari Environment")
    parser.add_argument('--with_PER', dest='with_per', action='store_true', help="Use Prioritized Experience Replay (DDQN + PER)")
    parser.add_argument('--dueling', dest='dueling', action='store_true', help="Use a Dueling Architecture (DDQN)")
    #
    parser.add_argument('--nb_episodes', type=int, default=5000, help="Number of training episodes")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size (experience replay)")
    parser.add_argument('--consecutive_frames', type=int, default=4, help="Number of consecutive frames (action repeat)")
    parser.add_argument('--training_interval', type=int, default=30, help="Network training frequency")
    parser.add_argument('--update_target_freq', type=int, default=200, help="Target network update frequency")
    # parser.add_argument('--n_threads', type=int, default=8, help="Number of threads (A3C)")
    #
    parser.add_argument('--gather_stats', dest='gather_stats', action='store_true',help="Compute Average reward per episode (slower)")
    # parser.add_argument('--render', dest='render', action='store_true', help="Render environment while training")
    parser.add_argument('--env', type=str, default='StageCampCarLidarNnEnv-v0',help="OpenAI Gym Environment")
    # parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    #
    parser.set_defaults(render=False)
    return parser.parse_args(args)


if __name__ == '__main__':
    args = None
    args = parse_args(args)
    set_session(get_session())

    env = gym.make(args.env)
    env._max_episode_steps = 25  # env returns done after _max_episode_steps
    outdir = '/tmp/stage_gym_experiments/'
    env = gym.wrappers.Monitor(env, outdir, force=True, resume=False)

    if args.env == 'StageSimpleRoomsLidarNnEnv-v0':
        state_dim = (176, 432, 3) #(40, 64, 2) (120, 160, 3)
        action_dim = 1189 #41
        action_shape = (22, 54)
    elif args.env == 'StageHospitalSectionPartOneNnEnv-v0':
        state_dim = (144, 216, 3)
        action_dim = 487
        action_shape = (18, 27)
    elif args.env == 'StageCampCarLidarNnEnv-v0':
        state_dim = (200, 200, 3)
        action_dim = 11
        action_shape = (13, 13)
    elif args.env == 'StageICRALidarNnEnv-v0':
        state_dim = (40, 64, 3)
        action_dim = 41
        action_shape = (5, 8)
    elif args.env == 'StageMazeCarLidarNnEnv-v0':
        state_dim = (25, 55, 3)
        action_dim = 9
        action_shape = (13, 13)
    else:
        state_dim = (72, 104, 3)
        action_dim = 118
        action_shape = (9, 13)

    beta_list = [0, 50, 100, 150, 200, 250, 300]
    beta = 200
    summary_name = "/tensorboard_{}_{}".format(args.env, beta)
    summary_writer = tf.summary.FileWriter(args.type + summary_name)

    model_path = None
    algo = DDQN(action_dim, state_dim, args, summary_writer, load_network_path=model_path)

    # # Train
    # stats = algo.train(env, args)

    # # # Export results to CSV
    # print "writing result ..."
    # df = pd.DataFrame(np.array(stats))
    # df.to_csv(args.type + summary_name + "/logs.csv",
    #           header=['Episode', 'score_mean', 'score_stddev', 'grid_mean', 'grid_stddev', 'path_mean',
    #                   'path_stddev'], float_format='%10.5f')
    
    model_path = './DDQN/ddqn200.h5' # icra
    # model_path = './DDQN_with_edge/ddqn249.h5' # for hospital
    algo.evaluate(env, model_path)

