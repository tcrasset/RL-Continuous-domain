import os
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam, sgd

from tensorflow import ConfigProto, Session
from keras.backend import set_session

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="0";  

# Just disables the warning for CPU instruction set,
#  doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


config = ConfigProto( device_count = {'GPU': 1 , 'CPU': 8} ) 
sess = Session(config=config) 
set_session(sess)

import math
import random
import click
import numpy as np
import pandas as pd

from collections import deque
from matplotlib import pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error

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
        next_speed = speed + self.integration_step * self._speedDiff(position, speed, action)

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
                - ((self._hill_diff(position) * self._hill_diff_diff(position)) 
                * (speed**2))/(1 + self._hill_diff(position)**2))

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

    def expectedReward(self, starting_position, starting_speed, computePolicy, N):
        """ 
        Computes the expected reward of a state given a policy after N iterations 
        Arguments:
        ----------
        - starting_position : position of the agent in the domain
        - starting_speed : speed of the agent in the domain
        - computePolicy : function that takes position and speed as argument and return
            an action

        Returns:
        ----------
        - float value for the expected reward
        """
        total_reward = 0
        position = starting_position
        speed = starting_speed
        for i in range(N):
            if self.stuck:
                break
            action = computePolicy(position, speed)
            reward, position, speed = self.rewardAtState(position, speed, action)
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
        for _ in range(int(self.ratio)):
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

        with click.progressbar(range(N)) as bar:
            for n in bar:
                print("Sample ", n)
                position = random.uniform(-1.0, 1.0)
                speed = random.uniform(-3.0, 3.0)
                J = self.domain.expectedReward(position, speed, policy, 200)
                self.domain.reset()
                sum_J.append(J)
        return np.mean(np.asarray(sum_J)), np.std(np.asarray(sum_J)), sum_J


class FittedQIterationAgent():

    def __init__(self, domain, model, n_episodes):
        self.domain = domain
        self.training_set = None
        self.four_tuples = None
        self.initTrainingSet(n_episodes)
        self.Q_model = model


    def initTrainingSet(self, n_episodes):
        # 1000 episodes with starting state (p,s) = (-0.5, 0)
        test_tuples = []
        for ep in range(n_episodes):
            position = -0.5
            speed = 0
            # Explore the domain until reaching a terminal state
            while not self.domain.isStuck():
                action = random.choice(self.domain.actions)
                reward, next_pos, next_speed = self.domain.rewardAtState(position, speed, action)
                test_tuples.append([position, speed, action, reward, next_pos, next_speed])
                position = next_pos
                speed = next_speed
            # One additional step
            action = random.choice(self.domain.actions)
            reward, next_pos, next_speed = self.domain.rewardAtState(position, speed, action)
            test_tuples.append([position, speed, action, reward, next_pos, next_speed])
            self.domain.reset()
        # Fill training set
        self.four_tuples = np.array(test_tuples)
        self.training_set = self.four_tuples[:,:4].copy()


    def createRewardHeatmap(self, n_iter, modeltype):
        print("Creating reward heatmap at iteration ", n_iter)
        positions = np.linspace(-1, 1, 100)
        speeds = np.linspace(-3, 3 ,100)

        forward_test_set = []
        backward_test_set = []

        for p in positions:
            for s in speeds:
                forward_test_set.append([p, s, 4])
                backward_test_set.append([p, s, -4])
        
        forward_rewards = self.Q_model.predict(forward_test_set).reshape(len(positions), len(speeds))
        backward_rewards = self.Q_model.predict(backward_test_set).reshape(len(positions), len(speeds))
        
        
        # Heatmap of Q(.,4)
        cs = plt.contourf(positions, speeds, forward_rewards, cmap='Spectral')
        plt.colorbar(cs)
        plt.title("Q(p, s, +4) after {} iterations".format(n_iter))
        plt.xlabel("Position")
        plt.ylabel("Speed")
        plt.savefig("Figures/{}/FittedQ_{}_forward_reward_{}_iter.svg".format(modeltype,modeltype, n_iter))
        plt.clf()

        # Heatmap of Q(.,-4)
        cs = plt.contourf(positions, speeds, backward_rewards, cmap='Spectral')
        plt.colorbar(cs)
        plt.title("Q(p, s, -4) after {} iterations".format(n_iter))
        plt.xlabel("Position")
        plt.ylabel("Speed")
        plt.savefig("Figures/{}/FittedQ_{}_backward_reward_{}_iter.svg".format(modeltype,modeltype, n_iter))
        plt.clf()


    def createPolicyHeatmap(self, n_iter, modeltype):
        print("Creating policy heatmap at iteration ", n_iter)
        positions = np.linspace(-1, 1, 100)
        speeds = np.linspace(-3, 3 ,100)

        policy = []
        forward_samples = []
        backward_samples = []
        for p in positions:
            for s in speeds:
                forward_samples.append([p,s,4])
                backward_samples.append([p,s,-4])
        forward_pred = self.Q_model.predict(np.array(forward_samples))
        backward_pred = self.Q_model.predict(np.array(backward_samples))

        for f, b in zip(forward_pred, backward_pred):
            policy.append(4 if f > b else -4)
        
        policy = np.array(policy).reshape(len(positions), len(speeds))

        # Heatmap of policy
        cs = plt.contourf(positions, speeds, policy, cmap='Spectral')
        plt.colorbar(cs)
        plt.title("Policy after {} iterations".format(n_iter))
        plt.xlabel("Position")
        plt.ylabel("Speed")
        plt.savefig("Figures/{}/FittedQ_{}_policy_{}_iter.svg".format(modeltype, modeltype, n_iter))
        plt.clf()


    def qIterationAlgo(self, n_iter, modeltype):
        avg_exp_rewards = []
        Q_diff = []
        for i in range(n_iter):
            print("Training iteration {} ...".format(i+1))
            X = self.training_set[:,:3]
            y = self.training_set[:,3].reshape(-1)
            # # Computing Q difference (old_Q)
            # if i > 0:
            #     old_Q_predict = self.Q_model.predict(self.four_tuples[:,:3])
            
            self.Q_model.fit(X,y)

            # # Computing Q difference (new_Q)
            # new_Q_predict = self.Q_model.predict(self.four_tuples[:,:3])
            # if i > 0:
            #     Q_diff.append(mean_squared_error(old_Q_predict,  new_Q_predict))

            self.updateTrainingSet()

            # Saving machine learning model
            filename = "Models/{}/{}{}.pkl".format(modeltype, modeltype,i)
            print("Saving machine learning model as {}".format(filename))
            joblib.dump(self.Q_model, filename)

            # # Creating heatmaps
            # if i in [0,4,9,19,49]:
            #     self.createRewardHeatmap(i + 1, modeltype)
            #     self.createPolicyHeatmap(i + 1, modeltype)
            
            if i in [0,4,9,14,19,24,29,34,39,44,49]:
                print("Computing average expected reward ...")
                avg_expected_reward = self.computeJ()
                print("Average expected reward at iteration {} : {}".format(i+1, avg_expected_reward))
                avg_exp_rewards.append(avg_expected_reward)

        # Plot average expected reward J
        plt.plot([0,4,9,14,19,24,29,34,39,44,49],avg_exp_rewards)
        plt.xlabel("Iteration")
        plt.ylabel("Expected reward")
        plt.savefig("Figures/{}/FittedQ_{}_expReward.svg".format(modeltype, modeltype))

        # # Plot Q difference wrt. the iteration
        # print("Plotting Q difference for ", modeltype)
        # plt.figure()
        # plt.plot(range(1,n_iter), Q_diff)
        # plt.xlabel("Iteration")
        # plt.ylabel("Q difference")
        # plt.savefig("Figures/{}/FittedQ_{}_qdiff.svg".format(modeltype,modeltype))
        # plt.clf()   


    def updateTrainingSet(self):
        # Find the max reward between Q(x_t+1, 4) and Q(x_t+1, -4)
        forward_set = np.c_[self.four_tuples[:,-2:], np.repeat(4,self.training_set.shape[0])]
        backward_set = np.c_[self.four_tuples[:,-2:], np.repeat(-4,self.training_set.shape[0])]
        forward_reward = self.Q_model.predict(forward_set)
        backward_reward = self.Q_model.predict(backward_set)
        max_reward = np.maximum(forward_reward, backward_reward)
        #Update training set rewards
        self.training_set[:,-1] = self.four_tuples[:,3] + self.domain.discount * max_reward


    def computeJ(self):
        """
        Computes the average expected reward for policy using a sequence 
        of initial states
        Arguments:
        ----------
        Returns:
        ----------
        - 
        """
        sum_states = []
        positions = np.arange(-1, 1, 0.125)
        speeds = np.arange(-3, 3 ,0.375)

        
        with click.progressbar(positions) as pos:
            for p in pos:
                for s in speeds:
                    J = self.domain.expectedReward(p, s, self.computePolicy, 100)
                    self.domain.reset()
                    sum_states.append(J)
        return np.mean(np.asarray(sum_states))


    def computePolicy(self, position, speed):
        to_pred = [[position, speed, 4], [position, speed, -4]]
        prediction = self.Q_model.predict(to_pred)
        return 4 if prediction[0] > prediction[1] else -4


    def plotTrajetory(self, modeltype):

        positions= []
        speeds = []

        position = -0.5
        speed = 0
        positions.append(position)
        speeds.append(speed)
        i = 0
        print("Plotting trajectory")
        # Computing trajectory with policy of Q_max_iter
        while not self.domain.isStuck() and i < 100:
            action = self.computePolicy(position, speed)
            position, speed = self.domain.nextState(position, speed, action)
            positions.append(position)
            speeds.append(speed)
            i += 1
        self.domain.reset()

        #Plotting
        plt.figure()
        plt.plot(positions, speeds)
        plt.xlabel("Position")
        plt.xlim(-1,1)
        plt.ylim(-3,3)
        plt.ylabel("Speed")
        plt.savefig("Figures/{}/FittedQ_{}_trajectory.svg".format(modeltype, modeltype))
        plt.clf()


class ParametricQLearningAgent:

    def __init__(self, domain, n_episodes):


        self.domain = domain
        self.n_episodes = n_episodes
        self.learning_rate = 0.05
        self.batch_size = 32
        self.epsilon = 0.1  # exploration rate
        self.epsilon_decay = 0.99
        self.model = self.create_model()
        self.memory = deque(maxlen=2000)
        
    def create_model(self):
        # Neural Net for Deep Q Learning
        model = Sequential()
        # Input Layer of state size(3) and Hidden Layer with 24 nodes
        model.add(Dense(24, input_shape=(3,), activation='relu'))
        # Hidden layer with 24 nodes
        model.add(Dense(24, activation='relu'))
        # Output Layer with reward
        model.add(Dense(1, activation='linear'))
        # Create the model based on the information above
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        return model


    def chooseAction(self, position, speed):
        """Epsilon greedy"""

        # Random action
        if np.random.rand() <= self.epsilon:
            return random.choice(self.domain.actions)

        # Action maximising Q
        prediction_f = self.model.predict(np.array([position, speed, 4]).reshape(1,3))
        prediction_b = self.model.predict(np.array([position, speed, -4]).reshape(1,3))
        return 4 if prediction_f > prediction_b else -4   

    def replay(self, batch_size):
        # Sample minibatch
        minibatch = random.sample(self.memory, batch_size)
        for position, speed, action, reward, next_position, next_speed in minibatch:
            # predict the future discounted reward
            forward_sample = np.array([next_position, next_speed, 4]).reshape(1,3)
            backward_sample = np.array([next_position, next_speed, -4]).reshape(1,3)
            forward_reward = self.model.predict(forward_sample)[0]
            backward_reward = self.model.predict(backward_sample)[0]
            target = reward + self.domain.discount * max(forward_reward, backward_reward)

            # Train the Neural Net with the state and target
            self.model.fit(np.array([position, speed, action]).reshape(1,3), target, epochs=1, verbose=0)
        self.epsilon *= self.epsilon_decay


    def train(self):
        # 1000 episodes with starting state (p,s) = (-0.5, 0)
        for ep in range(self.n_episodes):
            position = -0.5
            speed = 0
            i = 0
            while not self.domain.isStuck() and i < 200:
                action = self.chooseAction(position, speed)
                reward, next_pos, next_speed = self.domain.rewardAtState(position, speed, action)
                self.memory.append((position, speed, action, reward, next_pos, next_speed))
                position = next_pos
                speed = next_speed
                i += 1
            self.domain.reset()
            print("episode: {}/{}".format(ep, self.n_episodes))
            
            if (ep + 1) %50 == 0:
                self.createRewardHeatmap(ep + 1, 'mlp')
                self.createPolicyHeatmap(ep + 1, 'mlp')
            
            # Refit model on 32 samples
            if len(self.memory) < self.batch_size:
                self.replay(len(self.memory))
            else:
                self.replay(self.batch_size)
                

    def computePolicy(self, position, speed):
        # Action maximising Q
        prediction_f = self.model.predict(np.array([position, speed, 4]).reshape(1,3))
        prediction_b = self.model.predict(np.array([position, speed, -4]).reshape(1,3))
        return 4 if prediction_f > prediction_b else -4 

    def plotTrajetory(self, modeltype):

        positions= []
        speeds = []

        position = -0.5
        speed = 0
        positions.append(position)
        speeds.append(speed)
        i = 0
        print("Plotting trajectory")
        # Computing trajectory with policy of Q_max_iter
        while not self.domain.isStuck() and i < 100:
            action = self.computePolicy(position, speed)
            position, speed = self.domain.nextState(position, speed, action)
            positions.append(position)
            speeds.append(speed)
            i += 1
        self.domain.reset()

        #Plotting
        plt.figure()
        plt.plot(positions, speeds)
        plt.xlabel("Position")
        plt.xlim(-1,1)
        plt.ylim(-3,3)
        plt.ylabel("Speed")
        plt.savefig("Figures/paramq/ParamQ_{}_trajectory.svg".format(modeltype))
        plt.clf()


    def createRewardHeatmap(self, episode, modeltype):
        print("Creating reward heatmap at episode ", episode)
        positions = np.linspace(-1, 1, 100)
        speeds = np.linspace(-3, 3 ,100)

        forward_rewards = []
        backward_rewards = []
        forward_rewards
        for p in positions:
            for s in speeds:
                forward_rewards.append(self.model.predict(np.array([p, s, 4]).reshape(1,3)))
                backward_rewards.append(self.model.predict(np.array([p, s, -4]).reshape(1,3)))
        
        forward_rewards = np.array(forward_rewards).reshape(len(positions), len(speeds))
        backward_rewards = np.array(backward_rewards).reshape(len(positions), len(speeds))
        
        
        # Heatmap of Q(.,4)
        cs = plt.contourf(positions, speeds, forward_rewards, cmap='Spectral')
        plt.colorbar(cs)
        plt.title("Q(p, s, +4) after {} episodes".format(episode))
        plt.xlabel("Position")
        plt.ylabel("Speed")
        plt.savefig("Figures/paramq/ParamQ_{}_forward_reward{}_ep.svg".format(modeltype, episode))
        plt.clf()

        # Heatmap of Q(.,-4)
        cs = plt.contourf(positions, speeds, backward_rewards, cmap='Spectral')
        plt.colorbar(cs)
        plt.title("Q(p, s, -4) after {} episodes".format(episode))
        plt.xlabel("Position")
        plt.ylabel("Speed")
        plt.savefig("Figures/paramq/ParamQ_{}_backward_reward{}_ep.svg".format(modeltype, episode))
        plt.clf()


    def createPolicyHeatmap(self, episode, modeltype):
        print("Creating policy heatmap at iteration ", episode)
        policy = []
        positions = np.linspace(-1, 1, 100)
        speeds = np.linspace(-3, 3 ,100)

        forward_rewards = []
        backward_rewards = []
        forward_rewards
        for p in positions:
            for s in speeds:
                forward_rewards.append(self.model.predict(np.array([p, s, 4]).reshape(1,3)))
                backward_rewards.append(self.model.predict(np.array([p, s, -4]).reshape(1,3)))
        
        for f, b in zip(forward_rewards, backward_rewards):
            policy.append(4 if f > b else -4)
        
        policy = np.array(policy).reshape(len(positions), len(speeds))

        # Heatmap of policy
        cs = plt.contourf(positions, speeds, policy, cmap='Spectral')
        plt.colorbar(cs)
        plt.title("Policy after {} episodes".format(episode))
        plt.xlabel("Position")
        plt.ylabel("Speed")
        plt.savefig("Figures/paramq/ParamQ_{}_policy_{}_ep.svg".format(modeltype, episode))
        plt.clf()



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
    # plt.savefig("Figures/evolution.png")
    # plt.show()


    """ ================= Fitted Q Iteration ============================"""
    NB_ITER = 50
    linear_model = LinearRegression(n_jobs=-2)
    Q_iteration_agentLINEAR = FittedQIterationAgent(Domain(DISCOUNT, ACTIONS, INTEGRATION_STEP, DISCRETE_STEP),linear_model, 1000)
    Q_iteration_agentLINEAR.qIterationAlgo(NB_ITER,"linear")
    Q_iteration_agentLINEAR.plotTrajetory("linear")

    extratrees_model = ExtraTreesRegressor(n_estimators=50, n_jobs=-2)
    Q_iteration_agentTREES = FittedQIterationAgent(Domain(DISCOUNT, ACTIONS, INTEGRATION_STEP, DISCRETE_STEP),extratrees_model, 1000)
    Q_iteration_agentTREES.qIterationAlgo(NB_ITER, "trees")
    Q_iteration_agentTREES.plotTrajetory("trees")
    
    mlp_model =  MLPRegressor(activation='tanh', solver='adam', max_iter=2000, hidden_layer_sizes= (5,5))
    Q_iteration_agentMLP = FittedQIterationAgent(Domain(DISCOUNT, ACTIONS, INTEGRATION_STEP, DISCRETE_STEP),mlp_model, 1000)
    Q_iteration_agentMLP.qIterationAlgo(NB_ITER,"mlp")
    Q_iteration_agentMLP.plotTrajetory("mlp")


    """ ================= Parametric Q ============================"""

    paramqMLP = ParametricQLearningAgent(Domain(DISCOUNT, ACTIONS, INTEGRATION_STEP, DISCRETE_STEP), 1000)
    paramqMLP.train()
    paramqMLP.plotTrajetory("mlp")