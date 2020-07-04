# NEAT

NEAT (NeuroEvolution of Augmenting Topologies) is an algorithm which applies genetic algorithm to evolve both, the weights and topology of a neural network.

This repository contains the code for the third part of the webinar series I am doing on Evolutionary AI. In this session, I discuss the initial [NEAT paper](http://nn.cs.utexas.edu/downloads/papers/stanley.cec02.pdf) and demonstrate NEAT applied on Gym Environments, such as the LunarLander.


The webinar series can be accessed here:
1. [Genetic Algorithms](https://www.meetup.com/Disrupt-4-0/events/271033356/)
2. [Intro to NeuroEvolution](https://www.meetup.com/Disrupt-4-0/events/zfsxrrybcjbbc/)
3. [NeuroEvolution of Augmenting Topologies (NEAT)](https://www.meetup.com/Disrupt-4-0/events/271212059/)

## Usage
### Training

* Run the [train.py](train.py) script with the necessary arguments as described in the help menu of the CLI by running `python train.py --help`. 
* To train on default settings, run `python train.py`
* You can also stop the training in between and continue it using the `--load_ckpt` argument and passing the path of the checkpoint. For example, `python train.py --load_ckpt checkpoint_G49`
* By default, the best trained genome is saved as a `best.genome` file.

### Testing

* Test the trained genome by executing the [run.py](run.py) file with the necessary arguments described in the help menu, `python run.py --help`.
* On default settings, to run and watch the simulation on the trained genome (`best.genome`), `python run.py --render`.
* To run the evaluation consequently for large episodes, `python run.py --episodes 100`.