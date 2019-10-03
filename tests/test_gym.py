import gym
from PIL import Image

env = gym.make("Breakout-v0")
observation = env.reset()
for _ in range(1000):
    env.render()
    action = env.action_space.sample()  # your agent here (this takes random actions)
    observation, reward, done, info = env.step(action)

    # print(observation.shape)
    # for i in range(3):
    #     img = Image.fromarray(observation[:, :, i])
    #     img.save('img_'+str(i)+'.png')
    if done:
        observation = env.reset()
    # break
env.close()
