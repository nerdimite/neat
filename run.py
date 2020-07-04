import argparse
import os
import neat
import numpy as np
import gym
import pickle

def run_agent(genome, config, episodes=3, render=False, max_episode_length=None):
    env = gym.make("LunarLander-v2")
    
    agent = neat.nn.FeedForwardNetwork.create(genome, config)
    
    total_rewards = []
    
    for ep in range(episodes):
        observation = env.reset()
        if max_episode_length is not None:
            env._max_episode_steps = max_episode_length
        
        episodic_reward = 0
        
        while(1):
            if render:
                env.render()
            
            action = np.array(agent.activate(observation)).argmax()
            observation, reward, done, info = env.step(action)
            
            episodic_reward += reward
            if done:
                break
        
        
        total_rewards.append(episodic_reward)
    
    env.close()
    print('Mean Rewards across all episodes', np.array(total_rewards).mean())
    print('Best Reward in any single episode', max(total_rewards))
    
    
if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description='Run a trained NEAT Agent')
    parser.add_argument('--episodes', type=int, default=3, 
                        help='Number of episodes to run the agent for')
    parser.add_argument('--genome', type=str, default='best.genome', help='Path of the genome of the trained agent')
    parser.add_argument('-r', '--render', action='store_true', default=False, 
                        help='Flag to whether render the environment or not')
    args = parser.parse_args()
    
    config_file = 'config-feedforward'
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    
    # Load the Best Genome
    genome = pickle.load(open(args.genome, "rb"))
    
    # Run the agent
    run_agent(genome, config, args.episodes, args.render)