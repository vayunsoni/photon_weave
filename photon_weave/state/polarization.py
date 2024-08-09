"""
Polarization State
"""

from enum import Enum
from __future__ import annotations

import numpy as np

from .expansion_levels import ExpansionLevel


class PolarizationLabel(Enum):
    """
    Labels for the polarization basis states
    """
    H = "H"
    V = "V"
    R = "R"
    L = "L"


class Polarization:
    """
    Polarization class

    This class handles the polarization state or points to the
    Envelope or Composite envelope, which holds the state

    Attributes
    ----------
    index: Union[int, Tuple[int]]
        If polarization is part of a product space index
        holds information about the space and subspace index
        of this state
    dimension: int
        The dimensions of the Hilbert space (2)
    label: PolarizationLabel
        If expansion level is Label then label holds the state
    state_vector: np.array
        If expansion level is Vector then state_vector holds
        the state
    density_matrix: np.array
        If expansion level is Matrix then density_matrix holds
        the state
    envelope: Envelope
        If the state is part of a envelope, the envelope attribute
        holds a reference to the Envelope instance
    expansion_level: ExpansionLevel
        Holds information about the expansion level of this system
    """
    def __init__(
        self,
        polarization: PolarizationLabel = PolarizationLabel.H,
        envelope: "Envelope" = None,
    ):
        self.index = None
        self.label = polarization
        self.dimensions = 2
        self.state_vector = None
        self.density_matrix = None
        self.envelope = envelope
        self.expansion_level = ExpansionLevel.Label

    def __repr__(self):
        if self.label is not None:
            return f"|{self.label.value}⟩"
        elif self.state_vector is not None:
            formatted_vector = "\n".join(
                [
                    f"{complex_num.real:.2f} {'+' if complex_num.imag >= 0 else '-'} {abs(complex_num.imag):.2f}j"
                    for complex_num in self.state_vector.flatten()
                ]
            )

            return f"{formatted_vector}"
        elif self.density_matrix is not None:
            formatted_matrix = "\n".join(
                [
                    "\t".join(
                        [
                            f"({num.real:.2f} {'+' if num.imag >= 0 else '-'} {abs(num.imag):.2f}j)"
                            for num in row
                        ]
                    )
                    for row in self.density_matrix
                ]
            )
            return f"{formatted_matrix}"
        elif self.index is not None:
            return "System is part of the Envelope"
        else:
            return "Invalid Fock object"

    def expand(self):
        """
        Expands the representation
        """
        if self.label is not None:
            match self.label:
                case PolarizationLabel.H:
                    vector = [1, 0]
                case PolarizationLabel.V:
                    vector = [0, 1]
                case PolarizationLabel.R:
                    # Right circular polarization = (1/sqrt(2)) * (|H⟩ + i|V⟩)
                    vector = [1 / np.sqrt(2), 1j / np.sqrt(2)]
                case PolarizationLabel.L:
                    # Left circular polarization = (1/sqrt(2)) * (|H⟩ - i|V⟩)
                    vector = [1 / np.sqrt(2), -1j / np.sqrt(2)]

            self.state_vector = np.array(vector)[:, np.newaxis]
            self.label = None
            self.expansion_level = ExpansionLevel.Vector
        elif self.state_vector is not None:
            self.density_matrix = np.outer(
                self.state_vector.flatten(), np.conj(self.state_vector.flatten())
            )
            self.state_vector = None
            self.expansion_level = ExpansionLevel.Matrix

    def extract(self, index: int):
        """
        Is this method necessary
        """
        self.index = index
        self.label = None
        self.density_matrix = None
        self.state_vector = None

    def set_index(self, minor:int, major:int=-1):
        """
        Sets the index, when product space is created, or
        manipulated

        Parameters
        ----------
        minor: int
            Minor index show the order of tensoring in the space
        major: int
            Major index points to the product space when it is in
            CompositeEnvelope
        """
        if major >= 0:
            self.index = (major, minor)
        else:
            self.index = minor

    def apply_operation(self, operation: PolarizationOperation):
        """
        Applies a specific operation to the state

        Todo
        ----
        If the state is in the product space the operation should be
        Routed to the correct space

        Parameters
        ----------
        operation: PolarizationOperation
            Operation which should be carried out on this state
        """
        from photon_weave.operation.polarization_operations import (
            PolarizationOperationType,
        )

        match operation.operation:
            case PolarizationOperationType.I:
                return
            case PolarizationOperationType.X:
                if self.label is not None:
                    match self.label:
                        case PolarizationLabel.H:
                            self.label = PolarizationLabel.V
                        case PolarizationLabel.V:
                            self.label = PolarizationLabel.H
                        case PolarizationLabel.R:
                            self.label = PolarizationLabel.L
                        case PolarizationLabel.L:
                            self.label = PolarizationLabel.R
        min_expansion_level = operation.expansion_level_required(self)
        while self.expansion_level < min_expansion_level:
            self.expand()
        operation.compute_operator()
        self._execute_apply(operation)

    def _execute_apply(self, operation: PolarizationOperation):
        """
        Actually executes the operation

        Todo
        ----
        Consider using gpu for this operation
        """
        if self.expansion_level == 1:
            self.state_vector = operation.operator @ self.state_vector
        elif self.expansion_level == 2:
            self.density_matrix = operation.operator @ self.density_matrix
            self.density_matrix @= operation.operator.conj().T

    def _set_measured(self, **kwargs):
        """
        Destroys the state
        """
        self.measured = True
        self.label = None
        self.expansion_level = None
        self.state_vector = None
        self.density_matrix = None

    def measure(self, **kwargs):
        """
        Measures this state

        Todo
        ----
        Needs to be Implemented still 
        """
        self._set_measured()
        return None
