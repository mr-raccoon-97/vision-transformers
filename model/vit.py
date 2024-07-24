from typing import Tuple, Optional
from typing import List
from einops.layers.torch import Rearrange

from torch import Tensor
from torch import cat, randn, flatten
from torch.nn import Parameter
from torch.nn import Module, Sequential, ModuleList
from torch.nn import LayerNorm
from torch.nn import Linear, GELU, Dropout
from torch.nn import MultiheadAttention
from torch.nn import Conv2d

    
class ConvolutionalImagePatchEmbedding(Module):
    def __init__(self, model_dimension: int, patch_shape: Tuple[int, int], number_of_channels: int):
        super().__init__()
        self.projector = Conv2d(number_of_channels, model_dimension, kernel_size=patch_shape, stride=patch_shape)

    def forward(self, input: Tensor) -> Tensor:
        output = self.projector(input)
        return flatten(output, 2).transpose(1, 2)
    

class LinearImagePatchEmbedding(Module):
    def __init__(self, model_dimension: int, patch_shape: Tuple[int, int], number_of_channels: int):
        super().__init__()
        patch_height, patch_width = patch_shape
        patch_dimension = number_of_channels * patch_height * patch_width
        self.projector = Sequential(
            Rearrange('b c (h ph) (w pw) -> b (h w) (ph pw c)', ph = patch_height, pw = patch_width),
            Linear(patch_dimension, model_dimension),
        )

    def forward(self, image: Tensor) -> Tensor:
        output = self.projector(image)
        return output


class Encoder(Module):
    def __init__(self, model_dimension: int, hidden_dimension: int, number_of_heads: int, dropout: float = 0.):
        super().__init__()
        self.normalization = LayerNorm(model_dimension)
        self.attention = MultiheadAttention(model_dimension, number_of_heads, dropout = dropout, batch_first=True)
        self.feed_forward = Sequential(
            LayerNorm(model_dimension),
            Linear(model_dimension, hidden_dimension),
            GELU(),
            Dropout(dropout),
            Linear(hidden_dimension, model_dimension),
            Dropout(dropout)
        )
        
    def forward(self, input: Tensor, need_weights: bool = False) -> Tuple[Tensor, Optional[Tensor]]:
        output = self.normalization(input)
        output, weights = self.attention(output, output, output, need_weights=need_weights)
        output = output + input
        output = self.feed_forward(output) + output
        return output, weights
    


class Transformer(Module):
    def __init__(self, model_dimension: int, hidden_dimension: int, number_of_layers: int, number_of_heads: int, dropout = 0.):
        super().__init__()
        self.norm = LayerNorm(model_dimension)
        self.layers = ModuleList([
            Encoder(model_dimension, hidden_dimension, number_of_heads, dropout) for layer in range(number_of_layers)
        ])

    def forward(self, sequence: Tensor, need_weights: bool = False) -> Tuple[Tensor, List[Optional[Tensor]]]:
        attention_weights = []
        for layer in self.layers:
            sequence, weights = layer(sequence)
            attention_weights.append(weights)
        return self.norm(sequence), attention_weights 



class CLSToken(Module):
    def __init__(self, model_dimension: int):
        super().__init__()
        self.token = Parameter(randn(1, 1, model_dimension))

    def forward(self, input: Tensor) -> Tensor:
        batch_size = input.shape[0]
        token = self.token.expand(batch_size, -1, -1)
        return cat([token, input], dim=1)


def number_of_patches(image_shape: Tuple[int, int], patch_shape: Tuple[int, int]) -> int:
    image_height, image_width = image_shape
    patch_height, patch_width = patch_shape
    return (image_height // patch_height) * (image_width // patch_width)



class LearnablePositionalEncoding(Module):
    def __init__(self, model_dimension: int, sequence_lenght_limit: int = 196):
        super().__init__()
        self.sequence_lenght_limit = sequence_lenght_limit
        self.position_embeddings = Parameter(randn(1, sequence_lenght_limit + 1, model_dimension))

    def forward(self, input: Tensor) -> Tensor:
        assert input.size(1) <= self.sequence_lenght_limit + 1, 'input sequence is too long'
        input = input + self.position_embeddings[:, :input.size(1)]
        return input


class ClassificationHead(Module):
    def __init__(self, model_dimension: int, number_of_classes: int):
        super().__init__()
        self.head = Linear(model_dimension, number_of_classes)

    def forward(self, input: Tensor) -> Tensor:
        return self.head(input[:, 0])


class ViT(Module):
    def __init__(
        self, 
        patch_shape: Tuple[int, int], 
        model_dimension: int, 
        number_of_layers: int, 
        number_of_heads: int, 
        hidden_dimension: int, 
        number_of_channels: int, 
        number_of_classes: int,
        max_image_shape: Tuple[int, int] = (28, 28),
        dropout = 0., 
    ):
        super().__init__()
        self.image_to_embeddings = Sequential(
            ConvolutionalImagePatchEmbedding(model_dimension, patch_shape, number_of_channels),
            CLSToken(model_dimension),
            LearnablePositionalEncoding(model_dimension, number_of_patches(max_image_shape, patch_shape)),
            Dropout(dropout),
        )

        self.transformer = Transformer(model_dimension, hidden_dimension, number_of_layers, number_of_heads, dropout)
        self.head = ClassificationHead(model_dimension, number_of_classes)

    def forward(self, image: Tensor) -> Tensor:
        output = self.image_to_embeddings(image)
        output, weights = self.transformer(output)
        return self.head(output)