import math
import random
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor
from display_caronthehill import save_caronthehill_image


class Domain:

    def __init__(self, discount, actions, integration_step, discretisation_step):
        self.discount = discount
        self.actions = actions
        self.integration_step = integration_step
        self.discretisation_step = discretisation_step
        self.ratio = discretisation_step / integration_step
        self.stuck = False

    def _nextPosition(self, position, speed):
        """
        Computes the next position using the Euler integration method
        of a position and a speed.
        Arguments:
        ----------
        - position : position of the agent in the domain
        - speed : speed of the agent in the domain

        Returns:
        ----------
        - float value for the position
        """

        next_position = position + self.integration_step * speed

        # Check if you reach a terminal state
        if abs(next_position) > 1:
            self.stuck = True
        return next_position

    def _nextSpeed(self, position, speed, action):
        """
        Computes the next speed using the Euler integration method
        using the position, speed and action taken of the agent.
        Arguments:
        ----------
        - position : position of the agent in the domain
        - speed : speed of the agent in the domain
        - agent : action that the agent took in the domain

        Returns:
        ----------
        - float value for the speed
        """
        next_speed = speed + self.integration_step * \
            self._speedDiff(position, speed, action)

        # Check if you reach a terminal state
        if abs(next_speed) > 3:
            self.stuck = True
        return next_speed

    def _speedDiff(self, position, speed, action):
        """ 
        Derivative used for the speed dyanmics 
        Arguments:
        ----------
        - position : position of the agent in the domain
        - speed : speed of the agent in the domain
        - agent : action that the agent took in the domain

        Returns:
        ----------
        - float value for the speed derivative
        """
        return (action/(1 + self._hill_diff(position)**2)
                - 9.81 * self._hill_diff(position) /
                (1 + self._hill_diff(position)**2)
                - ((self._hill_diff(position) * self._hill_diff_diff(position)) * (speed**2))/(1 + self._hill_diff(position)**2))

    def _hill_diff(self, position):
        """ 
        Derivative of the Hill(position) function which represents
        the hill in the car on the hill problem 
        Arguments:
        ----------
        - position : position of the agent in the domain
        Returns:
        ----------
        - float value for the hill derivative
        """
        if position < 0:
            return 2 * position + 1
        else:
            return (1/math.sqrt(1 + 5 * position ** 2)
                    - 5 * position ** 2 * (1 + 5 * position ** 2)**-1.5)

    def _hill_diff_diff(self, position):
        """ 
        Second derivative of the Hill(position) function which represents
        the hill in the car on the hill problem 
        Arguments:
        ----------
        - position : position of the agent in the domain
        Returns:
        ----------
        - float value for the second hill derivative
        """
        if position < 0:
            return 2
        else:
            return position * ((75 * (position ** 2)/((1 + 5 * position**2)**2.5)) - 5/((1 + 5 * position ** 2)**2.5)) \
                - 10 * position/((1 + 5 * position ** 2)**1.5)

    def isStuck(self):
        """ Return true if the agent reached a terminal state in the domain"""
        return self.stuck

    def reset(self):
        """ Reset the terminal state lock of the domain"""
        self.stuck = False

    def expectedReward(self, starting_position, starting_speed, policy, N):
        """ 
        Computes the expected reward of a state given a policy after N iterations 
        Arguments:
        ----------
        - starting_position : position of the agent in the domain
        - starting_speed : speed of the agent in the domain
        - policy : policy that the agent will follow

        Returns:
        ----------
        - float value for the expected reward
        """
        total_reward = 0
        position = starting_position
        speed = starting_speed
        #TODO:   action = policy[(starting_position, starting_speed)]
        action = policy
        for i in range(N):
            if not self.stuck:
                reward, position, speed = self.rewardAtState(
                    position, speed, action)
                total_reward += (self.discount ** i) * reward
        return total_reward

    def nextState(self, position, speed, action):
        """ 
        Computes the next state (position and speed) for a given position,
        speed and action that the agent is going to take

        Arguments:
        ----------
        - position : position of the agent in the domain
        - speed : speed of the agent in the domain
        - agent : action that the agent took in the domain

        Returns:
        ----------
        - float value, float value
        """
        next_position = position
        next_speed = speed
        for _ in range(math.floor(self.ratio)):
            next_position = self._nextPosition(position, speed)
            next_speed = self._nextSpeed(position, speed, action)
            position = next_position
            speed = next_speed
        return next_position, next_speed

    def rewardAtState(self, position, speed, action):
        """
        Computes the reward that the agent is going to get 
        when starting at the given position and speed and taking the
        given action
        ----------
        - position : position of the agent in the domain
        - speed : speed of the agent in the domain
        - agent : action that the agent took in the domain

        Returns:
        ----------
        - integer, float value, float value
        """

        # If terminal state, return 0 for subsequent moves and
        # do not update the position nor speed
        if(self.stuck):
            return 0, position, speed

        new_pos, new_speed = self.nextState(position, speed, action)

        if (new_pos < -1 or abs(new_speed) > 3):
            return -1, new_pos, new_speed
        elif (new_pos > 1 and abs(new_speed) <= 3):
            return 1, new_pos, new_speed
        else:
            return 0, new_pos, new_speed


class Agent:
    def __init__(self, domain):
        self.domain = domain

    def computeJ(self, policy, N):
        """
        Computes the cumulative expected reward for policy
        Arguments:
        ----------
        - policy: dictionary containing the action to take at each state
        - N : number of samples to take
        Returns:
        ----------
        - 
        """
        sum_J = []
        for n in range(N):
            print("Sample ", n)
            position = random.uniform(-1.0, 1.0)
            speed = random.uniform(-3.0, 3.0)
            J = self.domain.expectedReward(position, speed, policy, 250)
            self.domain.reset()
            sum_J.append(J)
        return np.mean(np.asarray(sum_J)), np.std(np.asarray(sum_J)), sum_J




class FittedQIterationAgent():

    def __init__(self, domain, n_episodes):
        self.domain = domain
        self.training_set = {}
        self.initTrainingSet(n_episodes)
        self.Q_model = None


    def initTrainingSet(self, n_episodes):
        # 1000 episodes with starting state (p,s) = (-0.5, 0)
        for ep in range(n_episodes):
            position = 0.5
            speed = 0
            while not self.domain.isStuck():
                action = random.choice(self.domain.actions)
                reward, next_pos, next_speed = self.domain.rewardAtState(position, speed, action)
                self.training_set[(position, speed, action )] = reward
                position = next_pos
                speed = next_speed
            self.domain.reset()

    def createHeatmap(self, n_iter, modeltype):
        print("Creating heatmap at iteration ", n_iter)
        positions = np.arange(-1, 1, 0.1)
        speeds = np.arange(-3, 3 ,0.1)

        f_samples = []
        b_samples = []
        for p in positions:
            for s in speeds:
                forward, backward = self.getBestAction(p,s)
                f_samples.append(forward)
                b_samples.append(backward)

        f_samples = np.asarray(f_samples).reshape(len(positions), len(speeds))

        fig, ax = plt.subplots()
        im = ax.imshow(f_samples)

        # We want to show all ticks...
        ax.set_xticks(np.arange(len(speeds)))
        ax.set_yticks(np.arange(len(positions)))

        # ... and label them with the respective list entries
        ax.set_xticklabels(["{:0.1f}".format(i) for i in speeds])
        ax.set_yticklabels(["{:0.1f}".format(i) for i in positions])

        # Create colorbar
        cbar = fig.colorbar(im, orientation="horizontal")
        cbar.ax.set_xlabel("Reward", rotation= 0, va="top")

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
                rotation_mode="anchor")

        ax.set_title("Q(p, s, +4) after {} iterations".format(n_iter))
        fig.tight_layout()
        plt.savefig("FittedQ_{}_forward_{}_iter.svg".format(modeltype, n_iter))


        b_samples = np.asarray(b_samples).reshape(len(positions), len(speeds))

        fig, ax = plt.subplots()
        im = ax.imshow(b_samples)

        # We want to show all ticks...
        ax.set_xticks(np.arange(len(speeds)))
        ax.set_yticks(np.arange(len(positions)))
        # ... and label them with the respective list entries
        ax.set_xticklabels(["{:0.1f}".format(i) for i in speeds])
        ax.set_yticklabels(["{:0.1f}".format(i) for i in positions])

        # Create colorbar
        cbar = fig.colorbar(im, orientation="horizontal")
        cbar.ax.set_xlabel("Reward", rotation= 0, va="top")

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
                rotation_mode="anchor")

        ax.set_title("Q(p, s, -4) after {} iterations".format(n_iter))
        fig.tight_layout()
        plt.savefig("FittedQ_{}_backwardward_{}_iter.svg".format(modeltype, n_iter))

    def qIterationAlgo(self, n_iter):
        for i in range(n_iter):
            print("Training iteration ",i)
            X = np.asarray(list(self.training_set.keys()))
            y = np.asarray(list(self.training_set.values())).reshape(-1)
            self.Q_model.fit(X,y)
            self.updateTrainingSet()

            if i in [0,4,9,19,49]:
                self.createHeatmap(i + 1)

    def updateTrainingSet(self):
        pass

class FittedQIterationLinearAgent(FittedQIterationAgent):

    def __init__(self, domain, n_episodes):
        super().__init__(domain, n_episodes)
        self.n_episodes = n_episodes
        self.Q_model = LinearRegression(n_jobs=-2)

    def updateTrainingSet(self):
        new_training_set = {}
        for ep in range(self.n_episodes):
            position = 0.5
            speed = 0
            while not self.domain.isStuck():
                action = random.choice(self.domain.actions)
                reward, next_pos, next_speed = self.domain.rewardAtState(position, speed, action)
                
                forward_reward = self.Q_model.predict(np.array([next_pos, next_speed, 4]).reshape(1,-1))[0]
                backward_reward = self.Q_model.predict(np.array([next_pos, next_speed, -4]).reshape(1,-1))[0]
                max_reward = backward_reward if forward_reward < backward_reward else forward_reward

                new_training_set[(position, speed, action)] = reward + self.domain.discount * max_reward
                position = next_pos
                speed = next_speed
            self.domain.reset()
        self.training_set = new_training_set


    def getBestAction(self, position, speed):
        forward_reward = self.Q_model.predict(np.array([position, speed, 4]).reshape(1,-1))[0]
        backward_reward = self.Q_model.predict(np.array([position, speed, -4]).reshape(1,-1))[0]
        return forward_reward, backward_reward
        # if forward_reward > backward_reward:
        #     return forward_reward, 4
        # else:
        #     return backward_reward, -4

    def computePolicy():
        forward_reward = self.Q_model.predict(np.array([position, speed, 4]).reshape(1,-1))[0]
        backward_reward = self.Q_model.predict(np.array([position, speed, -4]).reshape(1,-1))[0]

        return forward_reward, backward_reward
        # if forward_reward > backward_reward:
        #     return forward_reward, 4
        # else:
        #     return backward_reward, -4


class FittedQIterationTreesAgent(FittedQIterationAgent):

    def __init__(self, domain, n_episodes):
        super().__init__(domain, n_episodes)
        self.n_episodes = n_episodes
        self.Q_model = ExtraTreesRegressor(n_jobs=-2)

    def updateTrainingSet(self):
        new_training_set = {}
        for ep in range(self.n_episodes):
            print("Episode", ep)
            position = 0.5
            speed = 0
            while not self.domain.isStuck():
                action = random.choice(self.domain.actions)
                reward, next_pos, next_speed = self.domain.rewardAtState(position, speed, action)
                
                forward_reward = self.Q_model.predict(np.array([next_pos, next_speed, 4]).reshape(1,-1))
                backward_reward = self.Q_model.predict(np.array([next_pos, next_speed, -4]).reshape(1,-1))
                max_reward = backward_reward if forward_reward < backward_reward else forward_reward

                new_training_set[(position, speed, action)] = reward + self.domain.discount * max_reward
                position = next_pos
                speed = next_speed
            self.domain.reset()
        self.training_set = new_training_set

    def getBestAction(self, position, speed):
        forward_reward = self.Q_model.predict(np.array([position, speed, 4]).reshape(1,-1))
        backward_reward = self.Q_model.predict(np.array([position, speed, -4]).reshape(1,-1))

        return forward_reward, backward_reward
        # if forward_reward > backward_reward:
        #     return forward_reward, 4
        # else:
        #     return backward_reward, -4

    def computePolicy():
        forward_reward = self.Q_model.predict(np.array([position, speed, 4]).reshape(1,-1))
        backward_reward = self.Q_model.predict(np.array([position, speed, -4]).reshape(1,-1))

        return forward_reward, backward_reward
        # if forward_reward > backward_reward:
        #     return forward_reward, 4
        # else:
        #     return backward_reward, -4



class ParametricQLearningAgent:

    def __init__(self, domain):
        self.domain = domain

if __name__ == '__main__':

    def makeVideo():
        """ 
        Creates a 10 fps video in mp4 format called 'caronthehill_clip'
        using frames whose names follow the regular expression 'img%05d.jpg'
        and are issued from a directory called video that has to be created beforehand

        Arguments : /
        -----------

        Returns :
        ---------

        -video in the current working directory
        """
        os.system("cd video && ffmpeg -r 10 -i img%05d.jpg -vcodec mpeg4 -y caronthehill_clip.mp4")


    # CONSTANTS
    PLUS4 = 4
    MINUS4 = -4
    ACTIONS = [PLUS4, MINUS4]
    DISCOUNT = 0.95
    DISCRETE_STEP = 0.1
    INTEGRATION_STEP = 0.001

    """ ========================== Total expected reward for a policy =========================="""
    # det_domain = Domain(DISCOUNT, ACTIONS, INTEGRATION_STEP, DISCRETE_STEP)
    # agent = Agent(Domain(DISCOUNT, ACTIONS, INTEGRATION_STEP, DISCRETE_STEP))
    # mean, std, points = agent.computeJ(PLUS4, 10000)
    # print("Expected reward of policy 'PLUS4': ", mean)

    # # Plot histogram
    # plt.hist(points, bins=40)
    # plt.show()

    """ ========================== Graph of position, speeds and rewards over time =========================="""
    # det_domain = Domain(DISCOUNT, ACTIONS, INTEGRATION_STEP, DISCRETE_STEP)

    # positions = []
    # speeds = []
    # rewards = []
    # position = 2
    # speed = 3

    # print(det_domain.nextState(position, speed, MINUS4))

    # n_iter = 0
    # while not det_domain.isStuck():
    # # for i in range(10000):
    #     save_caronthehill_image(position, speed, "video/img{:05d}.jpg".format(n_iter))
    #     reward, next_pos, next_speed = det_domain.rewardAtState(position, speed, PLUS4)
    #     rewards.append(reward)
    #     positions.append(next_pos)
    #     speeds.append(next_speed)
    #     position = next_pos
    #     speed = next_speed
    #     n_iter += 1

    # makeVideo()

    # x = list(range(n_iter))
    # plt.plot(x, positions, x, speeds, x, rewards)
    # plt.title("Evolution of the car on the hill over time")
    # plt.xlabel("Time (in 0.1s)")
    # plt.gca().legend(('position','speed', 'reward'))
    # plt.savefig("evolution.png")
    # plt.show()


    """ ================= Fitted Q Iteration ============================"""
    NB_ITER = 50
    Q_iteration_agent = FittedQIterationLinearAgent(Domain(DISCOUNT, ACTIONS, INTEGRATION_STEP, DISCRETE_STEP), 1000)
    Q_iteration_agent.qIterationAlgo(NB_ITER)

