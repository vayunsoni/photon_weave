"""
Fock state 
"""
from __future__ import annotations

import numpy as np

from photon_weave.operation.fock_operation import FockOperation, FockOperationType

from .envelope import EnvelopeAssignedException
from .expansion_levels import ExpansionLevel


class Fock:
    """
    Fock class

    This class handles the Fock state or points to the
    Envelope or Composite envelope, which holds the state

    Attributes
    ----------
    index: Union[int, Tuple[int]]
        If Fock space is part of a product space index
        holds information about the space and subspace index
        of this state
    dimension: int
        The dimensions of the Hilbert space, can be set or is
        computed on the fly when expanding the state
    label: int
        If expansion level is Label then label holds the state
        (number basis state)
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
    def __init__(self, envelope: Envelope = None):
        """
        Creates Fock object in a vacuum state
        """
        self.index = None
        self.dimensions = -1
        self.label = 0
        self.state_vector = None
        self.density_matrix = None
        self.envelope = envelope
        self.expansion_level = ExpansionLevel.Label
        self.measured = False

    def __repr__(self):
        """
        Simple string representation, when printing the Fock state
        """
        if self.label is not None:
            return f"|{self.label}⟩"
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

    def __eq__(self, other: Fock):
        """
        Comparison operator for the states, returns True if
        states are expanded to the same level and are not part
        of the product space
        Todo
        ----
        Method should work for a spaces if they do not have equal
        expansion level
        """
        if not isinstance(other, Fock):
            return False
        if self.label is not None and other.label is not None:
            if self.label == other.label:
                return True
            return False
        if self.state_vector is not None and other.state_vector is not None:
            if np.array_equal(self.state_vector, other.state_vector):
                return True
            return False
        if self.density_matrix is not None and other.density_matrix is not None:
            if np.array_equal(self.density_matrix, other.density_matrix):
                return True
            return False
        return False

    def assign_envelope(self, envelope: Envelope):
        from .envelope import Envelope

        assert isinstance(envelope, Envelope)
        if self.envelope is not None:
            raise EnvelopeAssignedException("Envelope can't be reassigned")
        self.envelope = envelope

    def expand(self):
        """
        Expands the representation. If the state is stored in
        label then it is expanded to state_vector and if the
        state is in state_vector, then the state is expanded
        to the state_matrix
        """
        if self.dimensions < 0:
            self.dimensions = self.label + 3
        if self.expansion_level is ExpansionLevel.Label:
            state_vector = np.zeros(int(self.dimensions))
            state_vector[self.label] = 1
            self.state_vector = state_vector[:, np.newaxis]
            self.label = None
            self.expansion_level = ExpansionLevel.Vector
        elif self.expansion_level is ExpansionLevel.Vector:
            self.density_matrix = np.outer(
                self.state_vector.flatten(), np.conj(self.state_vector.flatten())
            )
            self.state_vector = None
            self.expansion_level = ExpansionLevel.Matrix


    def extract(self, index: int):
        """
        This method is called, when the state is
        joined into a product space. Then the
        index is set and the label, density_matrix and
        state_vector is set to None
        """
        self.index = index
        self.label = None
        self.density_matrix = None
        self.state_vector = None

    def apply_operation(self, operation: FockOperation) -> None:
        """
        Applies a specific operation to the state

        Todo
        ----
        If the state is in the product space the operation should be
        Routed to the correct space

        Parameters
        ----------
        operation: FockOperation
            Operation which should be carried out on this state
        """
        match operation.operation:
            case FockOperationType.Creation:
                if self.label is not None:
                    self.label += operation.apply_count
                    return
            case FockOperationType.Annihilation:
                if self.label is not None:
                    self.label -= operation.apply_count
                    if self.label < 0:
                        self.label = 0
                    return
        min_expansion_level = operation.expansion_level_required()
        while self.expansion_level < min_expansion_level:
            self.expand()

        cutoff_required = operation.cutoff_required(self._num_quanta)
        if cutoff_required > self.dimensions:
            self.resize(cutoff_required)

        match operation.operation:
            case FockOperationType.Creation:
                if self._num_quanta + operation.apply_count + 1 > self.dimensions:
                    self.resize(self._num_quanta + operation.apply_count + 1)
        operation.compute_operator(self.dimensions)

        self._execute_apply(operation)

    @property
    def _num_quanta(self):
        """
        The highest possible measurement outcome.
        returns highest basis with non_zero probability
        """
        if self.state_vector is not None:
            non_zero_indices = np.nonzero(self.state_vector)[
                0
            ]  # Get indices of non-zero elements
            highest_non_zero_index_vector = non_zero_indices[-1]
            return highest_non_zero_index_vector
        if self.density_matrix is not None:
            non_zero_rows = np.any(self.density_matrix != 0, axis=1)
            non_zero_cols = np.any(self.density_matrix != 0, axis=0)

            highest_non_zero_index_row = (
                np.where(non_zero_rows)[0][-1] if np.any(non_zero_rows) else None
            )
            highest_non_zero_index_col = (
                np.where(non_zero_cols)[0][-1] if np.any(non_zero_cols) else None
            )

            # Determine the overall highest index
            highest_non_zero_index_matrix = max(
                highest_non_zero_index_row, highest_non_zero_index_col
            )
            return highest_non_zero_index_matrix

    def _execute_apply(self, operation: FockOperation):
        """
        Actually executes the operation

        Todo
        ----
        Consider using gpu for this operation
        """
        if self.state_vector is not None:
            self.state_vector = operation.operator @ self.state_vector
        if self.density_matrix is not None:
            self.density_matrix = operation.operator @ self.density_matrix
            self.density_matrix @= operation.operator.conj().T

        if operation.renormalize:
            self.normalize()

    def normalize(self):
        """
        Normalizes the state.
        """
        if self.density_matrix is not None:
            trace_rho = np.trace(self.density_matrix)
            self.density_matrix = self.density_matrix / trace_rho
        if self.state_vector is not None:
            norm_psi = np.linalg.norm(self.state_vector)
            self.state_vector = self.state_vector / norm_psi

    def resize(self, new_dimensions:int):
        """
        Resizes the state to the new_dimensions

        Parameters
        ----------
        new_dimensions: int
            New size to change to
        """
        if self.label is not None:
            self.dimensions = new_dimensions
            return
        # Pad
        if self.dimensions < new_dimensions:
            pad_size = new_dimensions - self.dimensions
            if self.state_vector is not None:
                self.state_vector = np.pad(
                    self.state_vector,
                    ((0, pad_size), (0, 0)),
                    "constant",
                    constant_values=(0,),
                )
            if self.density_matrix is not None:
                self.density_matrix = np.pad(
                    self.density_matrix,
                    ((0, pad_size), (0, pad_size)),
                    "constant",
                    constant_values=0,
                )
        # Truncate
        elif self.dimensions > new_dimensions:
            pad_size = new_dimensions - self.dimensions
            if self.state_vector is not None:
                if np.all(self.state_vector[new_dimensions:] == 0):
                    self.state_vector = self.state_vector[:new_dimensions]
            if self.density_matrix is not None:
                bottom_rows_zero = np.all(self.density_matrix[new_dimensions:, :] == 0)
                right_columns_zero = np.all(
                    self.density_matrix[:, new_dimensions:] == 0
                )
                if bottom_rows_zero and right_columns_zero:
                    self.density_matrix = self.density_matrix[
                        :new_dimensions, :new_dimensions
                    ]
        self.dimensions = new_dimensions

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

    def measure(self, non_destructive=False, remove_composite=True, partial=False) -> int:
        """
        Measures the state in the number basis. This Method can be used if the
        state resides in the Envelope or Composite Envelope

        Parameters
        ----------
        non_destructive: bool
            If True the state won't be destroyed post measurement
        remove_composite: bool
            If True and the state is part of the composite envelope
            the state won't be removed from the composite 
        partial: bool
            If true then accompanying Polarization space in the envelop
            won't be measured

        Returns
        -------
        outcome: int
            Outcome of the measurement
        """
        if self.measured:
            raise FockAlreadyMeasuredException()
        outcome = None
        if isinstance(self.index, int):
            return self.envelope.measure(remove_composite=remove_composite)
        else:
            match self.expansion_level:
                case ExpansionLevel.Label:
                    outcome = self.label
                case ExpansionLevel.Vector:
                    probabilities = np.abs(self.state_vector.flatten()) ** 2
                    outcome = np.random.choice(len(probabilities), p=probabilities)
                case ExpansionLevel.Matrix:
                    probabilities = np.real(np.diag(self.density_matrix))
                    outcome = np.random.choice(len(probabilities), p=probabilities)
        if not partial:
            self.envelope._set_measured(remove_composite=remove_composite)
        self._set_measured()
        return outcome

    def _set_measured(self, **kwargs):
        """
        Destroys the state
        """
        self.measured = True
        self.label = None
        self.expansion_level = None
        self.state_vector = None
        self.density_matrix = None
        self.index = None

    def get_subspace(self) -> np.array:
        """
        Returns the space subspace. If the state is in label representation
        then it is expanded once. If the state is in product space,
        then the space will be traced from the product space

        Returns
        -------
        state: np.array
            The state in the numpy array
        """
        if self.index is None:
            if not self.label is None:
                self.expand()

            if not self.state_vector is None:
                return self.state_vector
            elif not self.density_matrix is None:
                return self.density_matrix
        elif len(self.index) == 1:
            # State is in the Envelope
            pass

        elif len(self.index) == 2:
            state = self.envelope.composite_envelope._trace_out(self, destructive=False)
            return state


class FockAlreadyMeasuredException(Exception):
    pass
