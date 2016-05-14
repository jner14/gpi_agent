import gym
import numpy as np
import random
import matplotlib.pyplot as plt
import time
import pickle
from os import path

env = gym.make('CartPole-v0')

print env.action_space

print env.observation_space.high
print env.observation_space.low


class Memory(object):

    memories = {}
    memCnt = 0
    rewardBar = 0
    BINCNT = 12
    obvMax = [.0001, .0001, .0001, .0001]
    obvMin = [-0.0001, -0.0001, -0.0001, -0.0001]
    OBVCNT = len(env.observation_space.high)
    bins = [np.linspace(obvMin[0], obvMax[0], BINCNT)] * OBVCNT

    def __init__(self, observation, reward, action):
        self.observation_bak = observation

        Memory.updateMaxMinBins(observation)

        self.observation = Memory.digitizeIt(observation)

        self.reward = reward
        self.action = action
        self.id = Memory.memCnt

        Memory.memories[Memory.memCnt] = self
        Memory.memCnt += 1

    @staticmethod
    def exists(obs, action):
        dobs = Memory.digitizeIt(obs)
        for mem in Memory.memories.values():
            if mem.observation == dobs and mem.action == action:
                return True
        return False

    @staticmethod
    def searchObservations(observation):
        best = []
        if len(Memory.memories) < 1: return -1

        Memory.updateMaxMinBins(observation)
        obs = Memory.digitizeIt(observation)

        for k, mem in Memory.memories.iteritems():

            if mem.observation == obs:
                if list(observation) != list(mem.observation_bak):
                    if mem.reward > Memory.rewardBar:
                        best.append(mem)

        if len(best) > 0:
            best = sorted(best, key=lambda x: x.reward, reverse=True)
            winner = best[0]

            # Trim rest from memory as they are unneeded
            if len(best) > 1:
                trash = best[1:]
                for bag in trash:
                    del Memory.memories[bag.id]

            # Return the match
            return winner
        else:
            return -1

    @staticmethod
    def saveToFile(filename):
        pickle.dump((Memory.memories,
                     Memory.memCnt,
                     Memory.rewardBar,
                     Memory.BINCNT,
                     Memory.obvMax,
                     Memory.obvMin,
                     Memory.OBVCNT,
                     Memory.bins), open(filename, "wb"))

    @staticmethod
    def loadFromFile(filename):
        Memory.memories,      \
            Memory.memCnt,    \
            Memory.rewardBar, \
            Memory.BINCNT,    \
            Memory.obvMax,    \
            Memory.obvMin,    \
            Memory.OBVCNT,    \
            Memory.bins = pickle.load(open(filename, "rb"))

    @staticmethod
    def updateMaxMinBins(observation):
        # print(Memory.bins)
        updated = False
        for i in range(Memory.OBVCNT):
            if observation[i] > Memory.obvMax[i]:
                updated = True
                Memory.obvMax[i] = observation[i]
            if observation[i] < Memory.obvMin[i]:
                updated = True
                Memory.obvMin[i] = observation[i]
            Memory.bins[i] = np.linspace(Memory.obvMin[i], Memory.obvMax[i], Memory.BINCNT)

        if updated:
            Memory.reDigitizeAll()

    @staticmethod
    def reDigitizeAll():
        Memory.bins = []
        for i in xrange(Memory.OBVCNT):
            Memory.bins.append(np.linspace(Memory.obvMin[i], Memory.obvMax[i], Memory.BINCNT))

        for mem in Memory.memories.itervalues():
            mem.observation = Memory.digitizeIt(mem.observation_bak)

    @staticmethod
    def digitizeIt(observation):
        obs = [int(np.digitize(observation[i], Memory.bins[i])) for i in range(Memory.OBVCNT)]
        return obs


class RewardCenter(object):

    furthestStep = 1
    BINCNT = 20
    bins = np.linspace(0, furthestStep, BINCNT)
    REWARD_BASE = 1
    timestep_history = []

    @staticmethod
    def check_bins(t):
        if t > RewardCenter.furthestStep:
            RewardCenter.furthestStep = t
            RewardCenter.bins = RewardCenter._update_bins()
            return 1
        else:
            return 0

    @staticmethod
    def _update_bins():
        # TODO Handle cases where timestep_history is less than BINCNT
        ts_hist = RewardCenter.timestep_history
        bin_cnt = RewardCenter.BINCNT

        bin_size = float(len(ts_hist)) / bin_cnt

        ts_hist_sorted = sorted(ts_hist)

        bin_indexes = [(i+1) * bin_size for i in xrange(bin_cnt)]

        bins = [ts_hist_sorted[int(i)] for i in bin_indexes]

        return bins

        # RewardCenter.bins = np.linspace(0, RewardCenter.furthestStep, bin_cnt)

    @staticmethod
    def update_rewards(memories, step):
        # Subtracts BINCNT so that half are negative, creating a negative rewards for lower bins
        reward = (RewardCenter._digitize_it(step) - 0.6 * RewardCenter.BINCNT) * RewardCenter.REWARD_BASE
        reward = reward
        for mem in memories:
            mem.reward += reward

    @staticmethod
    def _digitize_it(step):
        dstep = np.digitize(step, RewardCenter.bins)
        return dstep

    @staticmethod
    def saveToFile(filename):
        pickle.dump((RewardCenter.furthestStep,
                     RewardCenter.BINCNT,
                     RewardCenter.bins,
                     RewardCenter.REWARD_BASE), open(filename, "wb"))

    @staticmethod
    def loadFromFile(filename):
        RewardCenter.furthestStep, \
        RewardCenter.BINCNT,       \
        RewardCenter.bins,         \
        RewardCenter.REWARD_BASE = pickle.load(open(filename, "rb"))


def run(episode_cnt=100, max_timestep=200):
    tstep_results = []

    for j_episode in xrange(episode_cnt):

        # Memories for this episode
        episode_memories = []

        # The first observation of this episode
        obs = env.reset()

        # The steps
        for t in xrange(max_timestep):
            env.render()

            # act = env.action_space.sample()

            match = Memory.searchObservations(obs)

            if match != -1:
                act = match.action
            else:
                act = env.action_space.sample()

            # act = 1 if act == 0 else 0

            obs, rwd, done, info = env.step(act)

            #print("Reward %s: %s" % (t, rwd))

            if match == -1 and not Memory.exists(obs, act):
                # print("Random Action %s: %s" % (t, act))
                episode_memories.append(Memory(obs, 1, act))
            elif match != -1:
                episode_memories.append(match)
                # print("Memory Action %s: %s" % (t, act))


            if done:
                # After failing, update reward system

                # Add the current timestep history
                RewardCenter.timestep_history += tstep_results

                # First check bins to see if they need updating
                RewardCenter.check_bins(t)

                # Update rewards based on how many steps were performed
                RewardCenter.update_rewards(episode_memories, t)

                tstep_results.append(t)

                print "Episode {} finished after {} timesteps".format(j_episode, t + 1)
                break

    return tstep_results
# END RUN() FUNCTION


monitorIt = False
monitorName = '/tmp/cartpole-experiment-3'
memory_path = 'ancestral_memory'
runs = 100
episodes_per_run = 20

# Keep track of time steps taken
all_results = []

# Get monitor ready
if monitorIt:
    env.monitor.start(monitorName)

# Load ancestral_memory
if path.exists(memory_path + ".pkl"):
    Memory.loadFromFile(memory_path + ".pkl")
    RewardCenter.loadFromFile(memory_path + "_rewards.pkl")
    # Memory.BINCNT = 12
    # Memory.reDigitizeAll()

for r in range(1, runs + 1):

    timestep_results = run(episodes_per_run)
    all_results += timestep_results
    mean = int(np.mean(timestep_results))

    print("Run {} finished with a mean of {}".format(r, mean))

    # Plot Steps Taken per Episode
    plt.plot(all_results, 'bo', )
    plt.title('Run {} Mean = {}'.format(r, mean))
    plt.ylabel('Steps Taken')
    plt.xlabel('Episode')
    plt.axis([0, len(all_results), 0, max(all_results)])

    # Save memory to ancestral file
    Memory.saveToFile(memory_path + ".pkl")
    RewardCenter.saveToFile(memory_path + "_rewards.pkl")

    if r < runs:
        plt.show(block=False)
    else:
        plt.show(block=True)

print "Done"

if monitorIt:
    env.monitor.close()
    gym.upload(monitorName, api_key='sk_Ph6CeyS8SVK2plcajcWhw')