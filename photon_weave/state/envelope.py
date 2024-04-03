"""
Envelope
"""
from __future__ import annotations
from typing import Optional
from photon_weave.operation.generic_operation import GenericOperation
import numpy as np
class Envelope:
    def __init__(self, fock: Optional['Fock'] = None,
                 polarization: Optional['Polarization'] = None):
        if fock is None:
            from .fock import Fock
            self.fock = Fock(envelope=self)
        else:
            self.fock = fock

        if polarization is None:
            from .polarization import Polarization
            self.polarization = Polarization(envelope=self)
        else:
            self.polarization = polarization

        self.composite_vector = None
        self.composite_matrix = None
        self.composite_envelope = None
        
    def __repr__(self):
        if (self.composite_matrix is None and
            self.composite_vector is None):
            if (self.fock.expansion_level == 0 and
                self.polarization.expansion_level == 0):
                return f"{repr(self.fock)} ⊗ {repr(self.polarization)}"
            else:
                return f"{repr(self.fock)}\n   ⊗\n {repr(self.polarization)}"
        elif self.composite_vector is not None:
            formatted_vector = "\n".join(
                [f"{complex_num.real:.2f} {'+' if complex_num.imag >= 0 else '-'} {abs(complex_num.imag):.2f}j" for complex_num in self.composite_vector.flatten()])
            return f"{formatted_vector}"
        elif self.composite_matrix is not None:
            formatted_matrix = "\n".join(["\t".join([f"({num.real:.2f} {'+' if num.imag >= 0 else '-'} {abs(num.imag):.2f}j)" for num in row]) for row in self.composite_matrix])
            return f"{formatted_matrix}"
            
    def combine(self):
        """
        Combines the fock and polarization into one matrix
        """
        if self.fock.expansion_level == 0:
            self.fock.expand()
        if self.polarization.expansion_level == 0:
            self.polarization.expand()

        while self.fock.expansion_level < self.polarization.expansion_level:
            self.fock.expand()

        while self.fock.expansion_level > self.polarization.expansion_level:
            self.polarization.expand()

        if (self.fock.expansion_level == 1 and
             self.polarization.expansion_level == 1):
            self.composite_vector = np.kron(self.fock.state_vector,
                                            self.polarization.state_vector)
            self.fock.extract(0)
            self.polarization.extract(1)

        if self.fock.expansion_level == 2 and self.polarization.expansion_level == 2:
            self.composite_matrix = np.kron(self.fock.density_matrix,
                                            self.polarization.density_matrix)
            self.fock.extract(0)
            self.polarization.extract(1)

    def extract(self, state):
        pass

    @property
    def expansion_level(self):
        if self.composite_vector is not None:
            return 1
        elif self.composite_matrix is not None:
            return 2
        else:
            return -1

    def separate(self):
        pass

    def apply_operation(self, operation: GenericOperation):
        from photon_weave.operation.fock_operation import (
            FockOperation, FockOperationType)
        from photon_weave.operation.polarization_operations import (
            PolarizationOperationType, PolarizationOperation)
        if isinstance(operation, FockOperation):
            if (self.composite_vector is None and
                self.composite_matrix is None):
                self.fock.apply_operation(operation)
            else:
                fock_index = self.fock.index
                polarization_index = self.polarization.index
                operation.compute_operator(self.fock.dimensions)
                operators = np.ones(2)
                operators[fock_index] = operation.operator
                polarization_identity = PolarizationOperation(
                    operation=PolarizationOperationType.Identity)
                operators[polarization_index] = polarization_identity.operator
                operator = np.kron(*operators)
                if self.composite_vector is not None:
                    self.composite_vector = operator @ self.composite_vector
                if self.composite_matrix is not None:
                    self.composite_matrix = operator @ self.composite_matrix
                    op_dagger = operator.conj().T
                    self.composite_matrix = self.composite_matrix @ op_dagger
        if isinstance(operation, PolarizationOperation):
            if (self.composite_vector is None and
                self.composite_matrix is None):
                self.polarization.apply_operation(operation)
            else:
                fock_index = self.fock.index
                polarization_index = self.polarization.index
                operators = [1, 1]
                fock_identity = FockOperation(
                    operation=FockOperationType.Identity
                )
                fock_identity.compute_operator(self.fock.dimensions)
                operators[polarization_index] = operation.operator
                operators[fock_index] = fock_identity.operator
                operator = np.kron(*operators)
                if self.composite_vector is not None:
                    self.composite_vector = operator @ self.composite_vector
                if self.composite_matrix is not None:
                    self.composite_matrix = operator @ self.composite_matrix
                    op_dagger = operator.conj().T
                    self.composite_matrix = self.composite_matrix @ op_dagger



class EnvelopeAssignedException(Exception):
    pass
