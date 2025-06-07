import gymnasium as gym
import numpy as np
from td3_torch import Agent
from utils import plot_learning_curve
from airhockey_gym.airhockey_env import AirHockeyEnv

if __name__ == "__main__":
    env = gym.make("AirHockey-v0")
    agent = Agent(
        alpha=0.001,
        beta=0.001,
        input_dims=env.observation_space.shape,
        tau=0.005,
        env=env,
        batch_size=100,
        layer1_size=400,
        layer2_size=300,
        n_actions=env.action_space.shape[0],
    )

    n_games = 1500
    filename = "AirHockey_" + str(n_games) + "_2.png"
    figure_file = "plots/" + filename

    best_score = -np.inf
    score_history = []
    print("Starting training...")
    for i in range(n_games):
        observation, _ = env.reset()
        done = False
        truncated = False
        score = 0
        
        while not (done or truncated):
            action = agent.choose_action(observation)
            observation_, reward, done, truncated, info = env.step(action)
            agent.remember(observation, action, reward, observation_, done or truncated)
            agent.learn()
            score += reward
            observation = observation_

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print(
            "episode ", i,
            " | episode len ", info["steps"],
            " | reward %.2f" % score,
            " | avg last 100 %.3f" % avg_score,
        )

    x = [i + 1 for i in range(n_games)]
    plot_learning_curve(x, score_history, figure_file)
