import argparse
import os
import neat
import visualize
import numpy as np
import gym
import pickle
from time import time
import multiprocessing

START = time()

def eval_single_genome(genome, config, episodes=10, max_episode_length=None):
    '''Evaluate a single genome on a gym environment'''
    env = gym.make("LunarLander-v2")
    
    agent = neat.nn.FeedForwardNetwork.create(genome, config)
    
    total_rewards = []
    
    for ep in range(episodes):
        # Get initial observation
        observation = env.reset()
        
        # Modify the maximum steps that can be taken in a single episode
        if max_episode_length is not None:
            env._max_episode_steps = max_episode_length
        
        episodic_reward = 0
        # Start episode
        while(1):
            
            action = np.array(agent.activate(observation)).argmax()
            observation, reward, done, info = env.step(action)
            
            episodic_reward += reward
            if done:
                break
                
        total_rewards.append(episodic_reward)
                
    return np.array(total_rewards).mean()


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_single_genome(genome, config)
    print(f'\nElapsed Time: {time()-START}\n')
        
        
def main(gen, save_ckpt, load_ckpt, save_path):
    config_file = 'config-feedforward'
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    # Create population
    if load_ckpt is None:
        p = neat.Population(config)
    else:
        p = neat.Checkpointer.restore_checkpoint(load_ckpt)
        save_ckpt = load_ckpt.split('_G')[0]
    
    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(generation_interval=10, filename_prefix=f"{save_ckpt}_G"))
    
    # Evolve
    winner = p.run(eval_genomes, gen)
    
    print('\nBest genome:\n{}'.format(winner))
    print('\nTime Elapsed:', time() - START)
    
    visualize.draw_net(config, winner, False)
    visualize.plot_stats(stats, ylog=False, view=False)
    visualize.plot_species(stats, view=False)
    
    # Save Genome
    pickle.dump(winner, open(save_path, "wb"))
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Train a NEAT Algorithm on a Gym Environment')
    parser.add_argument('--save', type=str, default='best.genome', help='Filename for best genome after the training')
    parser.add_argument('--gen', type=int, default=None, 
                        help='Max Number of Generations. Default is None i.e. continue till fitness threshold is reached')
    parser.add_argument('--load_ckpt', type=str, default=None, help='Restore Checkpoint to continue training')
    parser.add_argument('--save_ckpt', type=str, default='checkpoint', help='Filename prefix for checkpoints')
    args = parser.parse_args()
    
    # Run the main program
    main(args.gen, args.save_ckpt, args.load_ckpt, args.save)