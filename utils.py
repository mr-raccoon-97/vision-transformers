from typing import Iterator, Tuple, Protocol
from abc import ABC, abstractmethod
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


class Metrics(ABC):
    @abstractmethod
    def start(self, phase: str):
        ...

    @abstractmethod
    def update(self, batch: int, loss: float, accuracy: float):
        ...

    @abstractmethod
    def stop(self):
        ...


def accuracy(predictions: Tensor, target: Tensor) -> float:
    return (predictions == target).float().mean().item()

def predictions(output: Tensor):
    return argmax(output, dim=1)

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
        metrics.update(batch, loss.item(), accuracy(predictions(output), target))
    metrics.stop()

def test(model: Module, criterion: Criterion, data: Data, metrics: Metrics, device: str):
    with torch.no_grad():
        model.eval()
        metrics.start('test')
        for batch, (input, target) in enumerate(data, start=1):
            input, target = input.to(device), target.to(device)
            output = model(input)
            loss = criterion(output, target)
            metrics.update(batch, loss.item(), accuracy(predictions(output), target))
        metrics.stop()