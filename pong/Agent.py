import numpy as np
import random
import math
from collections import defaultdict


def reshape_obs(observation):
    """
    Reshapes and 'discretizes' an observation for Q-table read/write
    Make sure the state space is not too large!

    :param observation: The to-be-reshaped/discretized observation. Contains the position of the
    'players', as well as the position and movement.
    direction of the ball.
    :return: The reshaped/discretized observation
    """
    _reshaped_obs = np.zeros((len(observation),10), dtype=int)
    for i in range(len(observation)):
        for j in range(10):
            if j <5 :
                _reshaped_obs[i][j] = math.trunc(observation[i][j],1)
            else :
                _reshaped_obs[i][j] = observation[i][j]   
    print(_reshaped_obs) 
    return f'{_reshaped_obs.reshape(-1, 10)}'


class Agent:
    """
    Skeleton q-learner agent that the students have to implement
    """

    def __init__(
            self, id, actions_n, obs_space_shape,
            gamma=0.9, # pick reasonable values for all of these!
            epsilon=0.2,
            min_epsilon=0.1,
            epsilon_decay=0.8,
            alpha=1
    ):
        """
        Initiates the agent

        :param id: The agent's id in the game environment
        :param actions_n: The id of actions in the agent's action space
        :param obs_space_shape: The shape of the agents observation space
        :param gamma: Depreciation factor for expected future rewards
        :param epsilon: The initial/current exploration rate
        :param min_epsilon: The minimal/final exploration rate
        :param epsilon_decay: The rate of epsilon/exploration decay
        :param alpha: The learning rate
        """
        self.id = id
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.actions_n = actions_n
        self.obs_space_shape = obs_space_shape
        self.alpha = alpha
        self.q = defaultdict(lambda: np.zeros(self.actions_n))

    def determine_action_probabilities(self, observation):
        """
        A function that takes the state as an input and returns the probabilities for each
        action in the form of a numpy array of length of the action space.
        :param observation: The agent's current observation
        :return: The probabilities for each action in the form of a numpy
        array of length of the action space.
        """
        print(observation)
        print("new object",reshape_obs(observation))
        # print(self.q[0])
        # print(self.q[1])
        # print(self.q[6])
        return np.array[self.q[observation]]

    def act(self, observation):
        """
        Determines and action, given the current observation.
        :param observation: the agent's current observation of the state of
        the world
        :return: the agent's action
        """
        
        # Action Space: 0: Noop, 1: Up, 2: Down
        # Agent Observation : Agent Coordinate + Ball location ( head and tail)
        # epsilon greedy rule
        print("the act started")
        if random.random() < self.epsilon:
            action =  random.randint(0,2)
        # pick best reward action
        else :
            print("let s format the data")
            action = np.argmax(self.determine_action_probabilities(observation))
        return action

    def update_history(
            self, observation, action, reward, new_observation
    ):
        """
        Updates the agent's Q-table

        :param observation: The observation *before* the action
        :param action: The action that has been executed
        :param reward: The reward the action has yielded
        :param new_observation: The observation *after* the action
        :return:
        """
        # counterfactual next action, to later backpropagate reward to current action
        next_action = np.argmax(self.q[reshape_obs(new_observation)])
        td_target = reward + self.gamma * self.q[reshape_obs(new_observation)][next_action]
        td_delta = td_target - self.q[reshape_obs(observation)][action]
        self.q[reshape_obs(observation)][action] += self.alpha * td_delta

