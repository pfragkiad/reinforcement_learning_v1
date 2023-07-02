import gym
import numpy as np
import matplotlib.pyplot as plt


# pygame

binsCount = 30

learningRate = 0.1
discount = 0.95  # how important are future actions
episodes = 4_000
showEvery = 500
epsilon = 0.5
startEpsilonDecaying = 1
endEpsilonDecaying = episodes // 2
epsilonDecayValue = epsilon / (endEpsilonDecaying - startEpsilonDecaying)

episodeRewards = []
aggregatedEpisodeRewards = {"ep": [], "avg": [], "min": [], "max": []}

scenario = "MountainCar-v0"


env = gym.make(scenario)
print(f"Actions: {env.action_space.n}")
print(f"High: {env.observation_space.high}")  # [0.6 0.07]
print(f"Low: {env.observation_space.low}")  # [-1.2 -0.07]
dimensions = env.observation_space.shape[0]  # len(env.observation_space.high)

# initialize number of bins/buckets per dimension
osSize = [binsCount] * len(env.observation_space.high)

# np.savez('test.npz',osSize)
# os2 = np.load('test.npz')[0]
# print(os2)

binSize = (env.observation_space.high - env.observation_space.low) / osSize
print(f"Size: {binSize}")

# initialize q table
q = np.random.uniform(low=-2, high=0, size=(osSize + [env.action_space.n]))


def getDiscreteState(state):
    discreteState = (state - env.observation_space.low) / binSize
    return tuple(discreteState.astype(np.int_))


# 0 - left
# 1 - still
# 2 - right

for episode in range(episodes):
    episodeReward = 0

    if episode % showEvery == 0 or episode == episodes - 1:  # also show the last one
        env = gym.make(scenario, render_mode="human")
        print(f"EPISODE: {episode}")
    else:
        env = gym.make(scenario)

    newDiscreteState = getDiscreteState(env.reset()[0])
    # print(f'Discrete state: {newDiscreteState}, Q: {q[newDiscreteState]}')
    # env.reset()

    terminated = False
    truncated = False
    oldDiscreteState = newDiscreteState
    while not terminated and not truncated:
        # action with the maximum Q
        if np.random.random() > epsilon:
            action = np.argmax(q[newDiscreteState])
        else:  # random action!
            action = np.random.randint(0, env.action_space.n)

        # new_state,reward,done,truncated,info =  env.step(action)

        newState, reward, terminated, truncated, _ = env.step(action)
        episodeReward += reward
        newDiscreteState = getDiscreteState(newState)

        # if previousDiscreteState != discreteState:
        #     print(f'Discrete state: {discreteState}, Q: {qTable[discreteState]}')

        # print(reward, new_state)

        oldIndex = oldDiscreteState + (action,)
        if not terminated:
            maxFutureQ = np.max(q[newDiscreteState])

            currentQ = q[oldIndex]
            newQ = (1 - learningRate) * currentQ + learningRate * (
                reward + discount * maxFutureQ
            )
            q[oldIndex] = newQ
        else:
            q[oldIndex] = 0
        # elif newState[0] >=env.goal_position:
        #     q[oldDiscreteState+(action,)] = 0

        oldDiscreteState = newDiscreteState

        # if terminated:
        # print(f"Terminated at Episode: {episode}")

        # not needed
        # env.render
    if truncated and episode % 10 == 0:
        print(f"Truncated at {episode}")
    elif terminated and episode % 10 == 0:
        print(f"Terminated at {episode}")

    if endEpsilonDecaying >= episode >= startEpsilonDecaying:
        epsilon -= epsilonDecayValue

    episodeRewards.append(episodeReward)

    #if episode % showEvery == 0:
    if episode % showEvery == 0 and episode>0:
        lastRewards = episodeRewards[-showEvery:]
        averageReward = sum(lastRewards)/len(lastRewards)
        aggregatedEpisodeRewards['ep'].append(episode)
        aggregatedEpisodeRewards['avg'].append(averageReward)
        aggregatedEpisodeRewards['min'].append(min(lastRewards))
        aggregatedEpisodeRewards['max'].append(max(lastRewards))

        print(f'Episode: {episode}, avg: {averageReward}, min: {min(lastRewards)}, max: {max(lastRewards)}')

        np.save(f'data/{episode:05}',q)

env.close()

plt.plot(aggregatedEpisodeRewards['ep'],aggregatedEpisodeRewards['avg'], label='avg')
plt.plot(aggregatedEpisodeRewards['ep'],aggregatedEpisodeRewards['min'], label='min')
plt.plot(aggregatedEpisodeRewards['ep'],aggregatedEpisodeRewards['max'], label='max')
plt.legend(loc='upper left')
plt.show()