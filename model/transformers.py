import math
from torch import Tensor
from torch import exp, sin, cos

### NOTE: Try this to initialize the positional encoding.

def sinusoidal_encoding(tensor: Tensor, scaling_factor: int = 10000):
    sequence_lenght_limit, model_dimension = tensor.size()
    for dimension in range(model_dimension):
        tensor[:,dimension] = dimension // 2 + 1
        tensor[:,dimension] = exp(-2*tensor[:,dimension] * math.log(scaling_factor) / model_dimension)
        for sequence in range(sequence_lenght_limit):
            if dimension % 2 == 0:
                tensor[sequence,dimension] = sin(sequence * tensor[sequence,dimension])
            else:
                tensor[sequence,dimension] = cos(sequence * tensor[sequence,dimension])