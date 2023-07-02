import gym
import numpy as np


# pygame

env = gym.make("MountainCar-v0", render_mode="human")


print(f"Actions: {env.action_space.n}")
print(f"High: {env.observation_space.high}")  # [0.6 0.07]
print(f"Low: {env.observation_space.low}")  # [-1.2 -0.07]

dimensions = env.observation_space.shape[0]  # len(env.observation_space.high)

# initialize number of bins/buckets per dimension
osSize = [20] * len(env.observation_space.high)

# np.savez('test.npz',osSize)
# os2 = np.load('test.npz')[0]
# print(os2)


binSize = (env.observation_space.high - env.observation_space.low) / osSize
print(f"Size: {binSize}")

q = np.random.uniform(low=-2, high=0, size=(osSize + [env.action_space.n]))

learningRate = 0.1
discount = 0.95  # how important are future actions
episodes = 10_000
epsilon = 0.5
startEpsilonDecaying = 1
endEpsilonDecaying = episodes // 2
epsilonDecayValue = epsilon / (endEpsilonDecaying - startEpsilonDecaying)


def getDiscreteState(state):
    discreteState = (state - env.observation_space.low) / binSize
    return tuple(discreteState.astype(np.int_))

# 0 - left
# 1 - still
# 2 - right

scenario = "MountainCar-v0"

for episode in range(episodes):
    if episode % 2000 == 0 or episode == episodes - 1:
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
        print(f"Truncated @{episode}")
    elif terminated and episode % 10 == 0:
        print(f"Terminated @{episode}")
    
    if endEpsilonDecaying >= episode >= startEpsilonDecaying:
        epsilon -= epsilonDecayValue


env.close()
