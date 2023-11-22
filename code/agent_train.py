'''
University of Colorado at Colorado Springs
PhD in Computer Science

Class: CS 4080-5080 - Reinforcement Learning - Fall 2021
Professor: Jugal Kalita
Student: Carlos Eugenio Lopes Pires Xavier Torres
Student ID: 110320128
E-mail: clopespi@uccs.edu
Date: November 22, 2021

Homework 3
Parking Lot Agent Train - Deep Q-learning (using Neural Network)
'''

from gridworld_env import GridworldEnv
import time
from dqn_agent import Agent
import tensorflow as tf
import numpy as np
import sys
from plot_results import plot, plot3

'''
----------------------------------
Parking lot grid map
----------------------------------
0 - black: empty space
1 - gray: barrier
2 - blue: empty parking spot
3 - green: target parking spot
4 - red: car agent start position
----------------------------------
'''
GRID_MAP = [
  [0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0],
  [0, 3, 2, 2, 2, 0],
  [0, 1, 1, 1, 1, 0],
  [0, 2, 2, 2, 2, 0],
  [0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 4]
]

def main():
    tf.compat.v1.disable_eager_execution()

    # Initialize the environment
    env = GridworldEnv(grid_map=GRID_MAP, window_title='Homework 3 - Simplified Parking Lot - Training agent')
    env.restart_once_done = True
    env.reward_negative = -0.01
    env.reward_positive = 1

    # Number of discrete states per state dimension
    ENV_SIZE = env.grid_map_shape

    # Number of discrete actions
    NUM_ACTIONS = env.action_space.n

    # Hyperparameters
    REPLAY_MEMORY = 1_000_000
    GAMMA = 0.99
    ALPHA = 0.001
    EPISILON = 0.8
    EPSILON_DECAY = np.prod(ENV_SIZE, dtype=float) / 10.0
    EPSILON_MIN = 0.001
    BATCH_SIZE = 64

    # Training constants
    MAX_EPISODES = 1_000
    MAX_STEPS_PER_EPISODE = np.prod(ENV_SIZE, dtype=int) * 10
    SOLVINGS_TO_END = 30
    SOLVED_STEPS = np.prod(ENV_SIZE, dtype=int)
    RENDER_ENV = False
    SHOW_PLOT = False
    SHOW_PLOT_END = True
    SHOW_DEBUG_INFO = False
    SIM_SPEED = 0.002
    NETWORK_TYPE = 3

    # Read command
    command = sys.argv[1] if len(sys.argv) > 1 else ''
    if command == '--render=1':
        RENDER_ENV = True

    # Create DQN agent
    dqn_agent_1 = Agent(gamma=GAMMA, epsilon=EPISILON, alpha=ALPHA,
                input_dims=2, n_actions=NUM_ACTIONS, net_type=1,
                mem_size=REPLAY_MEMORY, batch_size=BATCH_SIZE,
                epsilon_dec=EPSILON_DECAY, epsilon_end=EPSILON_MIN)
    dqn_agent_2 = Agent(gamma=GAMMA, epsilon=EPISILON, alpha=ALPHA,
                input_dims=2, n_actions=NUM_ACTIONS, net_type=2,
                mem_size=REPLAY_MEMORY, batch_size=BATCH_SIZE,
                epsilon_dec=EPSILON_DECAY, epsilon_end=EPSILON_MIN)
    dqn_agent_3 = Agent(gamma=GAMMA, epsilon=EPISILON, alpha=ALPHA,
                input_dims=2, n_actions=NUM_ACTIONS, net_type=3,
                mem_size=REPLAY_MEMORY, batch_size=BATCH_SIZE,
                epsilon_dec=EPSILON_DECAY, epsilon_end=EPSILON_MIN)
    dqn_agent_1.model_summary()
    dqn_agent_2.model_summary()
    dqn_agent_3.model_summary()

    # Create Deep Q-learning agent
    agent_dql_1 = DeepQLearning(dqn_agent=dqn_agent_1, env=env, max_episodes=MAX_EPISODES,
        max_steps_per_episode=MAX_STEPS_PER_EPISODE, show_plot=SHOW_PLOT, show_plot_end=SHOW_PLOT_END,
        solved_steps=SOLVED_STEPS, solvings_to_end=SOLVINGS_TO_END,
        render_env=RENDER_ENV, sim_speed=SIM_SPEED, show_debug_info=SHOW_DEBUG_INFO)
    agent_dql_2 = DeepQLearning(dqn_agent=dqn_agent_2, env=env, max_episodes=MAX_EPISODES,
        max_steps_per_episode=MAX_STEPS_PER_EPISODE, show_plot=SHOW_PLOT, show_plot_end=SHOW_PLOT_END,
        solved_steps=SOLVED_STEPS, solvings_to_end=SOLVINGS_TO_END,
        render_env=RENDER_ENV, sim_speed=SIM_SPEED, show_debug_info=SHOW_DEBUG_INFO)
    agent_dql_3 = DeepQLearning(dqn_agent=dqn_agent_3, env=env, max_episodes=MAX_EPISODES,
        max_steps_per_episode=MAX_STEPS_PER_EPISODE, show_plot=SHOW_PLOT, show_plot_end=SHOW_PLOT_END,
        solved_steps=SOLVED_STEPS, solvings_to_end=SOLVINGS_TO_END,
        render_env=RENDER_ENV, sim_speed=SIM_SPEED, show_debug_info=SHOW_DEBUG_INFO)
    
    plot_results_1, _ = agent_dql_1.start_training()
    plot_results_2, _ = agent_dql_2.start_training()
    plot_results_3, _ = agent_dql_3.start_training()
    # plot3(plot_results_1,plot_results_2,plot_results_3)

    plot(plot_results_1)
    input("Press Enter to continue...")
    plot(plot_results_2)
    input("Press Enter to continue...")
    plot(plot_results_3)
    input("Press Enter to continue...")


class DeepQLearning():
    def __init__(self, dqn_agent, env, max_episodes, max_steps_per_episode,
                 show_plot_end, show_debug_info, show_plot,
                 solvings_to_end, solved_steps, render_env, sim_speed):
        self.dqn_agent = dqn_agent
        self.env = env
        self.max_episodes = max_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.solvings_to_end = solvings_to_end
        self.solved_steps = solved_steps
        self.render_env = render_env
        self.sim_speed = sim_speed
        self.show_plot = show_plot
        self.show_plot_end = show_plot_end
        self.show_debug_info = show_debug_info

    def start_training(self):
        num_consecutive_solvings = 0
        plot_rewards = []
        plot_loss = []

        # Render tha maze
        if self.render_env:
            self.env.verbose = True

        for episode in range(self.max_episodes):

            # Reset the environment
            obs = self.env.reset()

            # Set the initial state
            previous_state = tuple(obs)
            total_reward = 0
            total_loss = 0

            for step in range(self.max_steps_per_episode):
                # Select an action: random or with highest q
                action = self.dqn_agent.choose_action(previous_state)

                # Execute the action
                obs, reward, done, _ = self.env.step(action)
                
                # Observe the result
                next_state = tuple(obs)
                total_reward += reward
                
                # Store transition
                self.dqn_agent.store_transition(previous_state, action, reward, next_state, done)
                
                # Set previous state to next state
                previous_state = next_state
                
                # Set the DQN to learn
                loss = self.dqn_agent.learn()
                total_loss = loss[0] if loss != None else 0

                # Print debug info
                if self.show_debug_info:
                    print("\nEpisode: %d" % (episode+1))
                    print("Steps: %d" % (step+1))
                    print("Action: %d" % action)
                    print("State: %s" % str(previous_state))
                    print("Reward: %f" % reward)
                    print("Total reward: %f" % total_reward)
                    print("Explore rate: %f" % self.dqn_agent.epsilon)
                    print("Consecutive solvings: %d" % num_consecutive_solvings)
                    print("")

                # Render tha maze
                if self.render_env:
                    self.env.verbose = True
                    time.sleep(self.sim_speed)
                
                if done:
                    print("Episode %d finished after %d steps with total_reward = %f and loss = %f (consecutive solvings %d)."
                        % (episode+1, step+1, total_reward, total_loss, num_consecutive_solvings))

                    # Update plot
                    plot_rewards.append(total_reward)
                    plot_loss.append(total_loss)
                    if self.show_plot:
                        plot(plot_rewards)

                    if step <= self.solved_steps:
                        num_consecutive_solvings += 1
                    else:
                        num_consecutive_solvings = 0

                    break

                elif step >= self.max_steps_per_episode - 1:
                    print("Episode %d timed out at %d with total reward = %f." % (episode+1, step+1, total_reward))

                    # Update plot
                    plot_rewards.append(total_reward)
                    plot_loss.append(total_loss)
                    if self.show_plot:
                        plot(plot_rewards)

            # The best policy is considered achieved when solved over SOLVINGS_TO_END times consecutively
            if num_consecutive_solvings > self.solvings_to_end:
                self.dqn_agent.save_model()
                return plot_rewards, plot_loss

            # Decay epsilon
            self.dqn_agent.decay_epsilon(episode)


if __name__ == "__main__":
    main()