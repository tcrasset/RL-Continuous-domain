import math
import numpy as np
import random
# from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

from display_caronthehill import save_caronthehill_image
class Domain:

    def __init__(self, discount, actions, integration_step):
        self.discount = discount
        self.actions = actions
        self.integration_step = integration_step
        self.stuck = False

    def nextPosition(self, position, speed):
        next_position = position + self.integration_step * speed

        #Check if you reach a terminal state
        if abs(next_position) > 1:
            self.stuck = True
        return next_position


    def nextSpeed(self, position, speed, action):
        next_speed = speed + self.integration_step * self._speedDiff(position, speed, action)

        #Check if you reach a terminal state
        if abs(next_speed) > 3:
            self.stuck = True
        return next_speed


    def nextState(self, position, speed, action):
        return self.nextPosition(position, speed), \
                self.nextSpeed(position, speed, action)


    def _speedDiff(self, position, speed, action):
        return (action/(1 + self._hill_diff(position)**2) 
                - 9.81 * self._hill_diff(position)/(1 + self._hill_diff(position)**2)
                - ((self._hill_diff(position)**2) * (speed**2))/(1 + self._hill_diff(position)**2))


    def _hill_diff(self, position):
        if position < 0:
            return 2 * position + 1
        else:
            return (1/math.sqrt(1 + 5 * position ** 2) 
                    - 5 * position ** 2 * (1 + 5 * position **2)**-1.5)

    def expectedReward(self, starting_position, starting_speed, policy, N):
        total_reward = 0
        position = starting_position
        speed = starting_speed
        # action = policy[(starting_position, starting_speed)]
        action = policy
        for i in range(N):
            reward, new_pos, new_speed = self.rewardAtState(position, speed, action)
            total_reward += (self.discount ** i) * reward
            position = new_pos
            speed = new_speed

        return total_reward


    def rewardAtState(self, position, speed, action):
        """Returns the reward from a given cell in the 
        domains `reward_matrix` whose coordinates are given by a tuple
        Arguments:
        ----------
        - coords : tuple describing the cell from which to extract the reward

        Returns:
        ----------
        - integer reward
        """

        new_pos, new_speed = self.nextState(position, speed, action)

        if (new_pos < -1 or abs(new_speed) > 3 and not self.stuck):
            return -1, new_pos, new_speed
        elif (new_pos > 1 and abs(new_speed) <= 3 and not self.stuck):
            return 1, new_pos, new_speed
        else:
            return 0, new_pos, new_speed


class Agent:

    def __init__(self, domain):
        self.domain = domain


if __name__ == '__main__':

    # CONSTANTS
    PLUS4 = 4
    MINUS4 = -4
    ACTIONS = [PLUS4, MINUS4]
    DISCOUNT = 0.95
    TIME_STEP = 0.001

    det_domain = Domain(DISCOUNT, ACTIONS, TIME_STEP)
    agent_det = Agent(det_domain)

    position = -1
    positions = []
    speeds = []
    speed = 1


    n_iter = 1000
    for i in range (n_iter):
        # save_caronthehill_image(position, speed, "video/test{}.jpg".format(i))
        next_pos = det_domain.nextPosition(position, speed)
        next_speed = det_domain.nextSpeed(position, speed, PLUS4)
        positions.append(next_pos)
        speeds.append(next_speed)
        # print("Next speed", next_speed)
        position = next_pos
        speed = next_speed




    # plt.plot(list(range(n_iter)), positions)
    # plt.show()
    # plt.plot(list(range(n_iter)), speeds)
    # plt.show()

    # policy = {}
    # positions = np.arange(-1, 1, 0.1)
    # speeds = np.arange(-3, 3 ,0.1)

    # totals = []
    # p = []
    # s = []
    # for pos in positions:
    #     for sp in speeds:
    #         # policy[(pos, sp)] = PLUS4
    #         total = det_domain.expectedReward(pos, sp, PLUS4, 1000)
    #         p.append(pos)
    #         s.append(sp)
    #         totals.append(total)
    #         # if abs(total) > 0.0001:
    #         #     print("Reward is {} at position {} with speed {}".format(total, pos,sp))
    #         # print("Total reward: ", det_domain.expectedReward(pos, sp, PLUS4, 1000))


    # from mpl_toolkits.mplot3d import Axes3D
    # from matplotlib import cm
    # from matplotlib.ticker import LinearLocator, FormatStrFormatter


    # fig = plt.figure()
    # ax = fig.gca(projection='3d')

    # # Make data.
    # X, Y = np.meshgrid(p, s)
    # Z = np.asarray(totals).reshape(1200,1)
    # # Plot the surface.
    # surf = ax.plot_surface(X, Y, Z,  cmap=cm.coolwarm, linewidth=0, antialiased=False)

    # # Customize the z axis.
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # # Add a color bar which maps values to colors.
    # fig.colorbar(surf, shrink=0.5, aspect=5)

    # plt.show()


    print("Total reward: ", det_domain.expectedReward(0, 2, PLUS4, 1000))
