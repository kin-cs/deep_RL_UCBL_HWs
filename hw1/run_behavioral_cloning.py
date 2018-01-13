
#!/usr/bin/env python

"""
Code referred from https://github.com/txizzle/drl/blob/master/hw1/run_cloning.py

Behavorial cloning of an expert policy using a simple Feedforward Neural Network.
Example usage:
    python run_behavioral_cloning.py experts/Humanoid-v1.pkl Humanoid-v1 data/Humanoid-v1_100_data.pkl \
    --render --num_rollouts 20
"""

import pickle
import numpy as np
import tensorflow as tf
import tf_util
import gym
import load_policy
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.utils import np_utils
from sklearn.utils import shuffle

def load_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.loads(f.read())
    return data

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('data_file', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    parser.add_argument("--max_timesteps", type=int)
    args = parser.parse_args()

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')

    # Set Parameters
    # task = 'Humanoid-v1'
    # task = 'HalfCheetah-v1'
    # task = 'Hopper-v1' 
    # task = 'Reacher-v1'
    # task = 'Ant-v1'
    # task = 'Walker2d-v1'
    task = args.envname
    task_data = args.data_file

    # Load in expert policy observation data
    -----------


    # Split data into train and test set
    -----------

    # Create a feedforward neural network
    -----------

    with tf.Session():
        ---------------

if __name__ == '__main__':
    main()