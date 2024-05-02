import numpy as np
import random
from abc import ABC, abstractmethod
from typing import List


class Parameter(ABC):
    def __init__(self, minVal: float, maxVal: float, n: int) -> None:
        self.min: float = minVal
        self.max: float = maxVal
        self.n: int = n
        self.currentIndex: int = 0
        self.data: List[float] = None
    
    def __iter__(self):
        if self.data is None:
            self._generate_data()
        self.currentIndex = 0
        return self
    
    def __next__(self):
        if self.currentIndex >= self.n:
            raise StopIteration
        value: float = self.data[self.currentIndex]
        self.currentIndex += 1
        return value
    
    def to_numpy(self) -> np.ndarray:
        if self.data is None:
            self._generate_data()
        return np.array(list(self.data))
    
    @abstractmethod
    def __str__(self) -> str:
        pass
    
    @abstractmethod
    def _generate_data(self) -> None:
        pass


class UniformRange(Parameter):
    def __str__(self) -> str:
        return f"UniformRange [{self.min}, {self.max}], with {self.n} points"
    
    def _generate_data(self) -> None:
        self.data = [self.min + i*((self.max - self.min) / (self.n-1)) for i in range(self.n)]


class RandomRange(Parameter):
    def __str__(self) -> str:
        return f"RandomRange [{self.min}, {self.max}], with {self.n} points"
    
    def _generate_data(self) -> None:
        self.data = [random.uniform(self.min, self.max) for i in range(self.n)]



if __name__ == "__main__":
    # testing
    u = RandomRange(0, 10, 5)
    
    print(list(u))
    v = list(u)
    print(u.to_numpy())
        