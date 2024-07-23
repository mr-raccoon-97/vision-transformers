from typing import Iterator, Tuple, Protocol
import torch
from torch import argmax
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer


class Criterion(Protocol):
    def __call__(self, input: Tensor, target: Tensor) -> Tensor:
        ...


class Data(Protocol):
    def __iter__(self) -> Iterator[Tuple[Tensor, Tensor]]:
        ...


class Metrics(Protocol):
    def start(self, phase: str) -> None:
        ...

    def record(self, batch: int, loss: float, output: Tensor, target: Tensor) -> None:
        ...

    def stop(self) -> None:
        ...


def train(model: Module, criterion: Criterion, optimizer: Optimizer, data: Data, metrics: Metrics, device: str):
    model.train()
    metrics.start('train')
    for batch, (input, target) in enumerate(data, start=1):
        input, target = input.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        metrics.record(batch, loss.item(), output, target)
    metrics.stop()

def test(model: Module, criterion: Criterion, data: Data, metrics: Metrics, device: str):
    with torch.no_grad():
        model.eval()
        metrics.start('test')
        for batch, (input, target) in enumerate(data, start=1):
            input, target = input.to(device), target.to(device)
            output = model(input)
            loss = criterion(output, target)
            metrics.record(batch, loss.item(), output, target)
        metrics.stop()