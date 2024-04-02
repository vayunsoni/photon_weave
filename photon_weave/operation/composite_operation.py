"""
Composite Operation:
operates on multiple spaces
"""
from enum import Enum, auto
from typing import Union
from photon_weave.state.composite_envelope import CompositeEnvelope
from photon_weave.state.envelope import Envelope
from photon_weave.operation.fock_operation import (
    FockOperation, FockOperationType
)

class WrongNumberOfStatesException(Exception):
    pass

class CompositeOperationType(Enum):
    NonPolarizingBeamSplit = auto()

def ensure_equal_expansion(func):
    """
    Ensures that all given states, have the same expansion level
    """
    def wrapper(self, *states, **kwargs):
        expansion_levels = [s.expansion_level for s in states]
        if max(expansion_levels) == 0:
            desired_expansion = 1
        else:
            desired_expansion = max(expansion_levels)
        for s in states:
            while s.expansion_level < desired_expansion:
                s.expand()
        return func(self, *states, **kwargs)
    return wrapper

def ensure_composite(func):
    def wrapper(self, *states, **kwargs):
        composite_envelopes = [s.composite_envelope for s in states if s.composite_envelope is not None]
        if len(composite_envelopes) == 0:
            new_envelope = CompositeEnvelope(*states)
        elif len(composite_envelopes) == 1:
            existing_envelope = composite_envelopes[0]
            for state in states:
                existing_envelope.add_envelope(state)
        else:
            new_envelope = CompositeEnvelope(*states)
        return func(self, *states, **kwargs)
    return wrapper


class CompositeOperation():
    def __init__(self, operation: CompositeOperationType,
                 apply_num: int = 1, **kwargs):
        self.kwargs = kwargs
        self.operation = operation
        self.operator = None
        self.apply_num = apply_num

    def operate(self, *args) -> Union[CompositeEnvelope, Envelope]:
        match self.operation:
            case CompositeOperationType.NonPolarizingBeamSplit:
                self._operate_non_polarizing_beam_split(*args)

    @ensure_composite
    def _operate_non_polarizing_beam_split(self, *args, **kwargs):
        if len(args) != 2:
            raise WrongNumberOfStatesException(
                "Beam split can operate only on two states")
        ce = args[0].composite_envelope
        ce.combine(args[0].fock, args[1].fock)
        ce.rearange(args[1].fock, args[0].fock)
        dim1 = args[0].fock.dimensions
        dim2 = args[1].fock.dimensions
        self.compute_operator(dim1, dim2)
        

    def compute_operator(self, *args, **kwargs):
        match self.operation:
            case CompositeOperationType.NonPolarizingBeamSplit:
                pass

