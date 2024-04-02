"""
Test the Composite Operations
"""
import random
import numpy as np
import unittest
from photon_weave.operation.fock_operation import (
    FockOperation, FockOperationType
)
from photon_weave.operation.composite_operation import (
    CompositeOperation, CompositeOperationType)
from photon_weave.state.fock import Fock
from photon_weave.state.polarization import Polarization
from photon_weave.state.envelope import Envelope


class TestFockOperation(unittest.TestCase):
    def test_non_polarizing_beam_split(self):
        env1 = Envelope()
        env2 = Envelope()
        env1.fock.resize(5)
        env2.fock.resize(5)
        op_create = FockOperation(operation=FockOperationType.Creation,
                                  apply_count=2)
        env1.fock.apply_operation(op_create)
        op = CompositeOperation(
            operation=CompositeOperationType.NonPolarizingBeamSplit)
        op.operate(env2, env1)
        self.assertTrue(env1.composite_envelope is env2.composite_envelope)
