import pandas as pd
import numpy as np
import sys
from keras import backend as K
import matplotlib.pyplot as plt
import time
import itertools
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from collections import deque

import os
import random
from collections import namedtuple
import collections
import copy

# Keras and backend for neural networks
import keras
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf
from tensorboardX import SummaryWriter

sys.path.append('virl')
import virl


class NN:
    def __init__(self, alpha, d_states, n_actions, nn_config):
        self.alpha = alpha
        self.nn_config = nn_config  # size of hidden layer
        self.n_actions = n_actions
        self.d_states = d_states
        self.n_layers = len(nn_config)
        self.model = self._build_model()

    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
        """
        Huber loss (for use in Keras), see https://en.wikipedia.org/wiki/Huber_loss
        The huber loss tends to provide more robust learning in RL settings where there are
        often "outliers" before the functions has converged.
        """
        error = y_true - y_pred
        cond = K.abs(error) <= clip_delta
        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)
        loss = K.mean(tf.where(cond, squared_loss, quadratic_loss))
        return loss

    def _build_model(self):
        model = Sequential()
        for layer in self.nn_config:
            model.add(Dense(layer, input_dim=self.d_states, activation='relu'))
        model.add(Dense(self.n_actions, activation='linear'))
        model.compile(loss='mean_squared_error',
                      optimizer=Adam(lr=self.alpha,
                                     clipnorm=10.))  # specify the optimiser, we clip the gradient of the norm which can make traning more robust
        return model

    def predict(self, s, a=None):
        if a is None:
            return self._predict_nn(s)
        else:
            return self._predict_nn(s)[a]

    def _predict_nn(self, state_hat):
        """
        Predict the output of the neural netwwork (note: these can be vectors)
        """
        x = self.model.predict(state_hat)
        return x

    def update(self, states, td_target):
        self.model.fit(states, td_target, epochs=1, verbose=0)  # take one gradient step usign Adam
        return


EpisodeStats = namedtuple("Stats", ["episode_rewards"])


# Main Q-learner
def q_learning_nn(env, func_approximator, func_approximator_target, num_episodes, visualization, GAMMA=0.9,
                  epsilon_init=0.1, epsilon_decay=0.99995,
                  epsilon_min=0.01, use_batch_updates=True, fn_model_in=None, fn_model_out=None):
    """
    Q-Learning algorithm for Q-learning using Function Approximations.
    Finds the optimal greedy policy while following an explorative greedy policy.

    Args:
        env: covid environment.
        func_approximator: Action-Value function estimator, behavior policy (i.e. the function which determines the next action)
        func_approximator_target: Action-Value function estimator, updated less frequenty than the behavior policy
        num_episodes: Number of episodes to run for.
        visualization: visulization by tensorboardX
        GAMMA: Gamma discount factor.
        epsilon_init: Exploration strategy; chance the sample a random action. Float between 0 and 1.
        epsilon_decay: Each episode, epsilon is decayed by this factor
        epislon_min: Min epsilon value
        use_batch_updates=True,
        fn_model_in: Load the model from the file if not None
        fn_model_out: File name of the saved model, saves the best model in the last 200 episodes

    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    memory = ReplayMemory(BUFFER_SIZE)  # init the replay memory
    n_actions = env.action_space.n
    d_states = env.observation_space.shape[0]
    best_reward = -np.inf

    # Synch the target and behavior network
    if fn_model_in is not None:
        func_approximator.model.load_weights(fn_model_in)
    func_approximator_target.model.set_weights(func_approximator.model.get_weights())

    # Keeps track of useful statistics
    stats = EpisodeStats(episode_rewards=np.zeros(num_episodes))

    epsilon = epsilon_init

    for i_episode in range(num_episodes):
        sys.stdout.flush()

        state = env.reset()
        state = np.reshape(state, [1, d_states])  # reshape to (1, d_state)
        done = False
        inner_count = 0

        # One step in the environment
        while not done:
            inner_count += 1

            # Select an action usign and epsilon greedy policy based on the main behavior network
            if np.random.rand() <= epsilon:
                action = random.randrange(n_actions)
            else:
                act_values = func_approximator.predict(state)[0]
                action = np.argmax(act_values)  # returns action

            # Take a step
            next_state, reward, done, _ = env.step(action)
            # state_list = next_state.tolist()
            # visualization.add_scalars('data', {'susceptibles': state_list[0],
            #                                    'infectious': state_list[1],
            #                                    'quarantined': state_list[2],
            #                                    'recovereds': state_list[3]}, inner_count)
            # visualization.add_scalar('reward', reward, inner_count)
            # visualization.add_scalar('action', action, inner_count)
            next_state = np.reshape(next_state, [1, d_states])

            # Add observation to the replay buffer
            if done:
                memory.push(state, action, next_state, reward)
            else:
                memory.push(state, action, next_state, reward)

            # Update current episode reward
            stats.episode_rewards[i_episode] += reward

            # Update network
            if func_approximator.alpha > 0.0 and len(memory) >= BATCH_SIZE:
                # Fetch a bacth from the replay buffer and extract as numpy arrays
                transitions = memory.sample(BATCH_SIZE)
                batch = Transition(*zip(*transitions))
                train_rewards = np.array(batch.reward)
                train_states = np.array(batch.state)
                train_next_state = np.array(batch.next_state)
                train_actions = np.array(batch.action)

                if use_batch_updates:
                    # Do a single gradient step computed based on the full batch
                    train_td_targets = func_approximator.predict(
                        train_states.reshape(BATCH_SIZE, d_states))  # predict current values for the given states
                    q_values_next = func_approximator_target.predict(
                        train_next_state.reshape(BATCH_SIZE, d_states))
                    train_td_targetstmp = train_rewards + GAMMA * np.max(q_values_next, axis=1)
                    train_td_targets[
                        (np.arange(BATCH_SIZE), train_actions.reshape(BATCH_SIZE, ).astype(int))] = train_td_targetstmp
                    func_approximator.update(train_states.reshape(BATCH_SIZE, d_states),
                                             train_td_targets)  # Update the function approximator using our target
                if epsilon > epsilon_min:
                    epsilon *= epsilon_decay

            state = next_state

            if done:
                # Synch the target and behavior network
                func_approximator_target.model.set_weights(func_approximator.model.get_weights())

                print("\repisode: {}/{}, reward: {}, epsilon: {:.2}".format(i_episode, num_episodes,
                                                                            stats.episode_rewards[i_episode], epsilon),
                      end="")

                # Save the best model
                if fn_model_out is not None and (stats.episode_rewards[i_episode] >= best_reward):
                    func_approximator.model.save_weights(fn_model_out)
                    best_reward = stats.episode_rewards[i_episode]

                # visualization.add_scalar('episode_reward', stats.episode_rewards[i_episode], i_episode)
                break

    return stats


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory:
    """
    Implement a replay buffer using the deque collection
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        """Saves a transition."""
        self.memory.append(Transition(*args))

    def pop(self):
        return self.memoery.pop()

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def test_model(_val_writer):
    nn_func_approximator.alpha = 0.0
    epsilon_fixed = 0.0
    stats_show = q_learning_nn(env, nn_func_approximator, nn_func_approximator_target, 1, _val_writer,
                               epsilon_init=epsilon_fixed, epsilon_decay=1.0, epsilon_min=epsilon_fixed,
                               fn_model_in="covid.h5")


if __name__ == '__main__':
    # get virl env
    env = virl.Epidemic()

    d_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    alpha = 0.01
    nn_config = [24, 24]
    BATCH_SIZE = 128
    BUFFER_SIZE = 500
    run_writer = SummaryWriter('runs/test')
    val_writer = SummaryWriter('vals/exp3')

    nn_func_approximator = NN(alpha, d_states, n_actions, nn_config)
    nn_func_approximator_target = NN(alpha, d_states, n_actions, nn_config)

    stats = q_learning_nn(env, nn_func_approximator, nn_func_approximator_target, 200, run_writer,
                          epsilon_init=0.1, epsilon_decay=0.99995, epsilon_min=0.001, use_batch_updates=True,
                          fn_model_in=None, fn_model_out="covid.h5")

    # test_model(val_writer)
