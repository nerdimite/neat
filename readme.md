# NEAT

NEAT (NeuroEvolution of Augmenting Topologies) is an algorithm which applies genetic algorithm to evolve both, the weights and topology of a neural network. This code in this repository implements NEAT for Gym Environments like the LunarLander.

## Requirements

* neat-python
* graphviz
* gym (including box2d environments)

## Usage

You can run this repository either on colab or locally.

### Colab

Run the [Colab Notebook](https://colab.research.google.com/drive/1-Fc57grgRzHnIG7eW3WVne3EjI0RL7KS?usp=sharing
) to run this repository directly

### Local
#### Training

* Run the [train.py](train.py) script with the necessary arguments as described in the help menu of the CLI by running `python train.py --help`. 
* To train on default settings, run `python train.py`
* You can also stop the training in between and continue it using the `--load_ckpt` argument and passing the path of the checkpoint. For example, `python train.py --load_ckpt checkpoint_G49`
* By default, the best trained genome is saved as a `best.genome` file.

#### Testing

* Test the trained genome by executing the [run.py](run.py) file with the necessary arguments described in the help menu, `python run.py --help`.
* On default settings, to run and watch the simulation on the trained genome (`best.genome`), `python run.py --render`.
* To run the evaluation consequently for large episodes, `python run.py --episodes 100`.

* You can accesss the trained model and results under [trained_files/](trained_files/) including the resulting neural network and plots of fitness and speciation.

## References

- [Efficient Evolution of Neural Network Topologies](http://nn.cs.utexas.edu/downloads/papers/stanley.cec02.pdf) (short version of NEAT)
- [Evolving Neural Networks through Augmenting Topologies](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf) (full version of NEAT)
- [NEAT-Python Documentation](http://neat-python.readthedocs.io/en/latest/)
