import numpy as np


class SimpleNeuralNetwork:
    """Red neuronal de dos capas implementada con numpy."""

    def __init__(self, input_size, hidden_size, output_size, genotype=None):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        total_params = (input_size * hidden_size) + (hidden_size * output_size)
        if genotype is None:
            self.genotype = np.random.randn(total_params)
        else:
            self.genotype = np.array(genotype, dtype=float)
            if self.genotype.size != total_params:
                raise ValueError("Genotype size does not match network parameters")

        idx = input_size * hidden_size
        self.w1 = self.genotype[:idx].reshape(input_size, hidden_size)
        self.w2 = self.genotype[idx:].reshape(hidden_size, output_size)

    def forward(self, x):
        x = np.asarray(x, dtype=float)
        h = np.tanh(x.dot(self.w1))
        return h.dot(self.w2)
