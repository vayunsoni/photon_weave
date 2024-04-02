from enum import Enum, auto
from .generic_operation import GenericOperation
import numpy as np
import scipy.linalg

class PolarizationOperationType(Enum):
    Identity = auto()

class PolarizationOperation(GenericOperation):
    def __init__(self, operation: PolarizationOperationType,
                 apply_count: int = 1, **kwargs):
        self.kwargs = kwargs
        self.operation = operation

    def compute_operator(self):
        match self.operation:
            case PolarizationOperationType.Identity:
                self.operator = np.eye(2)
