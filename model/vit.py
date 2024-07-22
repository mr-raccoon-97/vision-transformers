from typing import Optional, Tuple
from dataclasses import dataclass
from typing import List
from torch import flatten, randn, cat
from torch import Tensor
from torch.nn import Parameter
from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import GELU
from torch.nn import Dropout
from torch.nn import Sequential
from torch.nn import LayerNorm
from torch.nn import MultiheadAttention
from torch.nn import ModuleList


class ImageEmbeddings(Module):
    def __init__(self, image_width: int, image_height: int, input_channels: int, patch_size: int, model_dimension: int):
        super().__init__()
        self.number_of_patches = (image_width *  image_height // patch_size) ** 2        
        self.projector = Conv2d(input_channels, model_dimension, kernel_size=patch_size, stride=patch_size)

    def forward(self, input: Tensor) -> Tensor:
        output = self.projector(input)
        return flatten(output, 2).transpose(1, 2)


class CLSToken(Module):
    def __init__(self, model_dimension: int):
        super().__init__()
        self.token = Parameter(randn(1, 1, model_dimension))

    def forward(self, input: Tensor) -> Tensor:
        batch_size = input.shape[0]
        token = self.token.expand(batch_size, -1, -1)
        return cat([token, input], dim=1)


class LearnablePositionalEncoding(Module):
    def __init__(self, model_dimension: int, number_of_patches: int):
        super().__init__()
        self.position_embeddings = Parameter(randn(1, number_of_patches + 1, model_dimension))

    def forward(self, input: Tensor) -> Tensor:
        input = input + self.position_embeddings
        return input
    

class Encoder(Module):
    def __init__(self, model_dimension: int, hidden_dimension: int, number_of_heads: int, dropout: float):
        super().__init__()
        self.attention = MultiheadAttention(model_dimension, number_of_heads, dropout=dropout)
        self.first_layer_normalization = LayerNorm(model_dimension)
        self.second_layer_normalization = LayerNorm(model_dimension)
        self.mlp = Sequential(
            Linear(model_dimension, hidden_dimension),
            GELU(),
            Dropout(dropout),
            Linear(hidden_dimension, model_dimension),
            Dropout(dropout)
        )

    def forward(self, input: Tensor, need_weights: bool = False) -> Tuple[Tensor, Optional[Tensor]]:
        output = self.first_layer_normalization(input)
        attention, attention_weights = self.attention(output, output, output, need_weights=need_weights)
        output = output + attention
        output = self.second_layer_normalization(output)
        output = output + self.mlp(output)
        return output, attention_weights
    
@dataclass
class Settings:
    image_width: int
    image_height: int
    patch_size: int
    input_channels: int
    model_dimension: int
    hidden_dimension: int
    number_of_heads: int
    number_of_layers: int
    dropout: float

class ViTClassifier(Module):
    def __init__(self, settings: Settings, output_classes: int):
        super().__init__()
        self.image_embeddings = ImageEmbeddings(settings.image_width, settings.image_height, settings.input_channels, settings.patch_size, settings.model_dimension)
        self.cls_token = CLSToken(settings.model_dimension)
        self.positional_encoding = LearnablePositionalEncoding(settings.model_dimension, self.image_embeddings.number_of_patches)
        self.encoders = ModuleList([Encoder(settings.model_dimension, settings.hidden_dimension, settings.number_of_heads, settings.dropout) for layer in range(settings.number_of_layers)])
        self.classification_head = Linear(settings.model_dimension, output_classes)

    def forward(self, input: Tensor, need_weights: bool = False) -> Tuple[Tensor, Optional[List[Tensor]]]:
        output = self.image_embeddings(input)
        output = self.cls_token(output)
        output = self.positional_encoding(output)
        attention_weights = []
        for encoder in self.encoders:
            output, weight = encoder(output, need_weights=need_weights)
            if need_weights:
                attention_weights.append(weight)

        logits = self.classification_head(output[:, 0])
        return logits, attention_weights