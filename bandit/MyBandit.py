# epsilon-greedy example implementation of a multi-armed bandit
import random

class Bandit:
    """
    Generic epsilon-greedy bandit that you need to improve
    """
    def __init__(self, arms, epsilon=0.7, epsilon_decrease=0.95, epsilon_min=0.05, history_window=100):
        """
        Initiates the bandits

        :param arms: List of arms
        :param epsilon: Epsilon value for random exploration
        :param epsilon_decrease: Decrease epsilon value over time
        """
        self.arms = arms
        self.epsilon = epsilon
        self.frequencies = [0] * len(arms)
        self.sums = [0] * len(arms)
        self.expected_values = [0] * len(arms)
        self.epsilon_decrease = epsilon_decrease
        self.epsilon_min = epsilon_min

        self.history = {arm: [0] for arm in range(0, len(arms))}
        self.history_window = history_window
        

    def run(self):
        """
        Asks the bandit to recommend the next arm

        :return: Returns the arm the bandit recommends pulling
        """
        if min(self.frequencies) == 0:
            return self.arms[self.frequencies.index(min(self.frequencies))]
        if random.random() < self.epsilon:
            return self.arms[random.randint(0, len(self.arms) - 1)]
        return self.arms[self.expected_values.index(max(self.expected_values))]

    def give_feedback(self, arm, reward):
        """
        Sets the bandit's reward for the most recent arm pull

        :param arm: The arm that was pulled to generate the reward
        :param reward: The reward that was generated
        """
        arm_index = self.arms.index(arm)
        
        frequency = self.frequencies[arm_index] + 1
        self.frequencies[arm_index] = frequency
        sum_ = self.sums[arm_index] + reward
        self.sums[arm_index] = sum_
        # expected_value = sum / frequency
        # self.expected_values[arm_index] = expected_value

        self.history[arm_index].append(reward)
        freq = min(self.history_window, len(self.history[arm_index]))
        self.expected_values[arm_index] = sum(self.history[arm_index][len(self.history[arm_index]) - freq:])/freq

        # Decrease epsilon value over time
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decrease
