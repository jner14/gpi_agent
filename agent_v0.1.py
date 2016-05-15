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
    BINCNT = 20
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
        match_list = []
        if len(Memory.memories) > 0:

            Memory.updateMaxMinBins(observation)
            obs = Memory.digitizeIt(observation)

            for k, mem in Memory.memories.iteritems():

                if mem.observation == obs:
                    if list(observation) != list(mem.observation_bak):
                        match_list.append(mem)

            if len(match_list) > 0:
                match_list = sorted(match_list, key=lambda x: x.reward, reverse=True)

        return match_list

    @staticmethod
    def clean_house():


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

    BINCNT = 20
    bins = np.linspace(0, 1, BINCNT)
    REWARD_BASE = 1
    timestep_history = []
    MAX_STEP_HISTORY = 200

    @staticmethod
    def update_bins(min_steps):
        ts_hist = RewardCenter.timestep_history
        bin_cnt = RewardCenter.BINCNT

        if len(ts_hist) < min_steps:
            furthest_step = max(ts_hist)
            shortest_step = min(ts_hist) if min(ts_hist) != max(ts_hist) else 0
            bins = np.linspace(shortest_step, furthest_step, bin_cnt)
            return bins

        else:
            bin_size = float(len(ts_hist)) / bin_cnt
            ts_hist_sorted = sorted(ts_hist)
            bin_indexes = [(i+1) * bin_size for i in xrange(bin_cnt)]
            bins = [ts_hist_sorted[int(i)-1 if int(i)-1 >=0 else 0] for i in bin_indexes]
            return bins

    tempVal = 0
    @staticmethod
    def update_rewards(memories, step):
        # Subtracts BINCNT so that half are negative, creating a negative rewards for lower bins
        reward = (RewardCenter._digitize_it(step) - 0.5 * RewardCenter.BINCNT - 1) * RewardCenter.REWARD_BASE
        reward = reward
        # print("Reward Changed by {}".format(reward))
        # RewardCenter.tempVal += reward
        # print("Reward Cumulative Value: {}".format(RewardCenter.tempVal))
        for mem in memories:
            mem.reward += reward

    @staticmethod
    def _digitize_it(step):
        dstep = np.digitize(step, RewardCenter.bins)
        return dstep

    @staticmethod
    def add_timestep(step):
        RewardCenter.timestep_history += [step]
        if len(RewardCenter.timestep_history) > RewardCenter.MAX_STEP_HISTORY:
            diff = len(RewardCenter.timestep_history) - RewardCenter.MAX_STEP_HISTORY
            RewardCenter.timestep_history = RewardCenter.timestep_history[diff:]

    @staticmethod
    def saveToFile(filename):
        pickle.dump((RewardCenter.BINCNT,
                     RewardCenter.bins,
                     RewardCenter.REWARD_BASE,
                     RewardCenter.timestep_history,
                     RewardCenter.MAX_STEP_HISTORY), open(filename, "wb"))

    @staticmethod
    def loadFromFile(filename):
        RewardCenter.BINCNT,           \
        RewardCenter.bins,             \
        RewardCenter.REWARD_BASE,      \
        RewardCenter.timestep_history, \
        RewardCenter.MAX_STEP_HISTORY = pickle.load(open(filename, "rb"))


def run(episode_cnt=100, max_timestep=200):
    tstep_results = []

    for j_episode in xrange(episode_cnt):

        # Memories for this episode
        episode_memories = []

        # The first observation of this episode
        obs = env.reset()

        # The steps
        for t in xrange(max_timestep):
            act = None
            is_new_memory = False
            env.render()

            # Look for similar observations in memory
            matches = Memory.searchObservations(obs)
            if not len(matches) <= env.action_space.n:
                tmp = 1
            # TODO Use assert to check for too many matches
            # assert len(matches) <= env.action_space.n

            # If no matches are found, generate a random action
            if matches == []:
                act = env.action_space.sample()
                is_new_memory = True

            # If the top match has a positive reward, use its action
            elif matches[0].reward > 0:
                act = matches[0].action

            # If the action space is covered, still use the best match action
            elif len(matches) >= env.action_space.n:
                act = matches[0].action

            # Otherwise, generate a new action never used with this observation
            else:
                # Generate possible actions
                assert type(env.action_space) == type(gym.spaces.Discrete(2))
                poss_actions = xrange(env.action_space.n)

                # Verify if the actions exist in matches
                for p in poss_actions:
                    found = False
                    for m in matches:
                        if m.action == p:
                            found = True
                            break

                    # If the action is not found, break and use it
                    if not found:
                        act = p
                        is_new_memory = True
                        break

            if is_new_memory:
                if len(matches) > 2:
                    tempvar = 1
                memory1 = Memory(obs, 1, act)
            else:
                memory1 = matches[0]

            episode_memories.append(memory1)

            assert act != None
            obs, rwd, done, info = env.step(act)

            if done or t >= max_timestep - 1:
                # After failing, update reward system

                # Add the current timestep history
                RewardCenter.add_timestep(t)

                # Update bins
                RewardCenter.bins = RewardCenter.update_bins(episode_cnt)

                # Update rewards based on how many steps were performed
                RewardCenter.update_rewards(episode_memories, t)

                tstep_results.append(t)

                print "Episode {} finished after {} timesteps".format(j_episode, t + 1)
                break

    return tstep_results
# END RUN() FUNCTION

# TODO Add stop learning variable
monitorIt = False
monitorName = '/tmp/cartpole-experiment-5'
memory_path = 'ancestral_memory'
runs = 1
episodes_per_run = 100
max_timesteps = 300

# Keep track of time steps taken
all_results = []

# Get monitor ready
if monitorIt:
    env.monitor.start(monitorName)

# Load ancestral_memory
if path.exists(memory_path + ".pkl"):
    Memory.loadFromFile(memory_path + ".pkl")
    RewardCenter.loadFromFile(memory_path + "_rewards.pkl")
    # Memory.BINCNT = 4
    # Memory.reDigitizeAll()

for r in range(1, runs + 1):

    timestep_results = run(episodes_per_run, max_timesteps)
    all_results = RewardCenter.timestep_history
    mean = int(np.mean(timestep_results))

    print("Run {} finished with a mean of {}".format(r, mean))

    # Plot Steps Taken per Episode
    plt.plot(all_results, 'bo', )
    plt.title('Run {} Mean = {}'.format(r, mean))
    plt.ylabel('Steps Taken')
    plt.xlabel('Episode')
    plt.axis([0, len(all_results), 0, max(all_results)])

    #Clean memory of redundant entries
    # TODO clean memory of redundant entries
    Memory.clean_house()

    # Save memory to ancestral file
    Memory.saveToFile(memory_path + ".pkl")
    RewardCenter.saveToFile(memory_path + "_rewards.pkl")

    if r < runs:
        pass
        plt.show(block=True)
        # plt.show(block=False)
    else:
        plt.show(block=True)

print "Done"

if monitorIt:
    env.monitor.close()
    gym.upload(monitorName, api_key='sk_Ph6CeyS8SVK2plcajcWhw')