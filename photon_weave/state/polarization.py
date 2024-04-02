"""
Polarization State
"""
from enum import Enum
import numpy as np
from .expansion_levels import ExpansionLevel


class PolarizationLabel(Enum):
    # Horizontal Polarization
    H = "H"
    V = "V"
    R = "R"
    L = "L"


class Polarization:
    def __init__(self,
                 polarization: PolarizationLabel = PolarizationLabel.H,
                 envelope: 'Envelope' = None):
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
                [f"{complex_num.real:.2f} {'+' if complex_num.imag >= 0 else '-'} {abs(complex_num.imag):.2f}j" for complex_num in self.state_vector.flatten()])

            return f"{formatted_vector}"
        elif self.density_matrix is not None:
            formatted_matrix = "\n".join(["\t".join([f"({num.real:.2f} {'+' if num.imag >= 0 else '-'} {abs(num.imag):.2f}j)" for num in row]) for row in self.density_matrix])
            return f"{formatted_matrix}"
        elif self.index is not None:
            return "System is part of the Envelope"
        else:
            return "Invalid Fock object"


    def expand(self):
        if self.label is not None:
            match self.label:
                case PolarizationLabel.H:
                    vector = [1, 0]
                case PolarizationLabel.V:
                    vector = [0, 1]
                case PolarizationLabel.R:
                    # Right circular polarization = (1/sqrt(2)) * (|H⟩ + i|V⟩)
                    vector = [1/np.sqrt(2), 1j/np.sqrt(2)]
                case PolarizationLabel.L:
                    # Left circular polarization = (1/sqrt(2)) * (|H⟩ - i|V⟩)
                    vector = [1/np.sqrt(2), -1j/np.sqrt(2)]

            self.state_vector = np.array(vector)[:, np.newaxis]
            self.label = None
            self.expansion_level = ExpansionLevel.Vector
        elif self.state_vector is not None:
            self.density_matrix = np.outer(
                self.state_vector.flatten(),
                np.conj(self.state_vector.flatten()))
            self.state_vector = None
            self.expansion_level = ExpansionLevel.Matrix

    def extract(self, index: int):
        self.index = index
        self.label = None
        self.density_matrix = None
        self.state_vector = None
        

    def set_index(self, minor, major=-1):
        if major >= 0:
            self.index = (major, minor)
        else:
            self.index = minor
