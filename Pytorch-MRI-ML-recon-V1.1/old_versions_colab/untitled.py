import sys

sys.path.append('../')
sys.path.append("../binary_stochastic_neurons")
sys.path.append("../binary_stochastic_neurons/distributions")

from binary_stochastic_neurons.utils import Hardsigmoid
from binary_stochastic_neurons.activations import DeterministicBinaryActivation, StochasticBinaryActivation

print("yes")