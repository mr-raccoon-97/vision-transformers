# vision-transformers
Pytorch implementation of vision transformers


### Feature Extractor

For the transformer model to learn from images, it has to learn the representation of the image in the embedding space. This representation is obtained by applying a convolutional layer to the image.

Let:
- $d\in\mathbb{N}$ the embedding dimension.
- $X\in\mathbb{R}^{c \times h \times w}$ the input image, with $c$ channels, and height and width $h$ and $w$ respectively.

The idea is to divide the input image into blocks of size $k \times k$ or "patches". This can be don with convolutions applied to the image, with $d$ output channels, using filters of size $c \times k \times k$, with a stride of $k$, a filter will be applied to each block of the image. The result will be, given the filter $K\in\mathbb{R}^{c \times k \times k}$:

$$Y = \text{Conv2D}(X, K, \text{out channels}=d,\text{stride}=k)$$

With $Y\in\mathbb{R}^{d \times \frac{h}{k} \times \frac{w}{k}}$. Finally, if the tensor rank is reduced, and it is transposed, a tensor $Y\in\mathbb{R}^{\frac{hw}{k^2} \times d}$ is obtained, which will be the representation of the image in the embedding space.

This can be implemented as follows:

```python
class ImageEmbeddings(Module):
    def __init__(self, image_width: int, image_height: int, input_channels: int, patch_size: int, model_dimension: int):
        super().__init__()
        self.number_of_patches = (image_width *  image_height // patch_size) ** 2        
        self.projector = Conv2d(input_channels, model_dimension, kernel_size=patch_size, stride=patch_size)

    def forward(self, input: Tensor) -> Tensor:
        output = self.projector(input)
        return flatten(output, 2).transpose(1, 2)
```