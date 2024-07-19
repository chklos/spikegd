# Smooth exact gradient descent learning in spiking neural networks

[![arXiv](https://img.shields.io/badge/arXiv-2309.14523-b31b1b.svg)](https://arxiv.org/abs/2309.14523)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![JAX](https://img.shields.io/badge/Framework-JAX-important)](https://github.com/google/jax)

This repository contains the code underlying the paper [*Smooth Exact Gradient Descent Learning in Spiking Neural Networks*](https://arxiv.org/abs/2309.14523). It shows how to perform gradient descent learning that is both exact and smooth (or at least continuous) in spiking neural networks. The scheme relies on neuron models whose spikes can only appear or vanish at the trial end, such as quadratic integrate-and-fire neurons. These neuron models further allow adding or removing spikes in a principled, gradient-based manner.

We use event-based spiking neural network simulations, iterating over input and network spikes. The code is written in Python using [JAX](https://github.com/google/jax) and makes use of its automatic differentiation and JIT-compilation features.

## Setup

1. Create a virtual environment and install Python 3.10 as well as JAX, [preferably with GPU support](https://jax.readthedocs.io/en/latest/installation.html).
2. Clone or download this repository.
3. Install the package and the dependencies necessary to use it or to run most of the experiments via

      ```bash
      pip install -e .
      ```

4. For the MNIST experiments, you need to additionally [install](https://pytorch.org/get-started/locally/) PyTorch (cpu-only is sufficient) and TorchVision to load the dataset.

## Structure

The neuron models including methods to simulate networks of them and to compute pseudospike times are located in `spikegd`. Abstract base classes, i.e. templates for neuron models, are given in `models.py`. Specific implementations of a few neuron models are provided in the other files in the same directory.

The `experiments` folder contains scripts and notebooks to generate the main results of the paper. They can also serve as a starting point to implement new experiments.

## Citation

If you use this code in your research, please cite our arXiv paper:

```bibtex
@misc{klos2023smooth,
      title={Smooth Exact Gradient Descent Learning in Spiking Neural Networks}, 
      author={Christian Klos and Raoul-Martin Memmesheimer},
      year={2023},
      eprint={2309.14523},
      archivePrefix={arXiv},
      primaryClass={q-bio.NC}
}
```
