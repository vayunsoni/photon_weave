"""
Tests for the Fock class
"""
import numpy as np
import unittest
from photon_weave.state.envelope import Envelope
from photon_weave.state.composite_envelope import (
    CompositeEnvelope,
    StateNotInThisCompositeEnvelopeException,
    FockOrPolarizationExpectedException
)
from photon_weave.state.polarization import (
    Polarization,
    PolarizationLabel
)
from photon_weave.operation.fock_operation import (
    FockOperation, FockOperationType
)


class TestFock(unittest.TestCase):
    def test_initiation_from_envelopes(self):
        env1 = Envelope()
        env2 = Envelope()
        composite_envelope = CompositeEnvelope(env1, env2)
        self.assertEqual(env1.composite_envelope, composite_envelope)
        self.assertEqual(env2.composite_envelope, composite_envelope)
        self.assertTrue(env1 in composite_envelope.envelopes)
        self.assertTrue(env2 in composite_envelope.envelopes)

    def test_envelope_addition(self):
        env1 = Envelope()
        env2 = Envelope()
        composite_envelope = CompositeEnvelope(env1, env2)
        env3 = Envelope()
        composite_envelope.add_envelope(env3)
        self.assertEqual(env3.composite_envelope, composite_envelope)
        self.assertTrue(env3 in composite_envelope.envelopes)


    def test_composite_envelope_merge(self):
        env1 = Envelope()
        env2 = Envelope()
        ce1 = CompositeEnvelope(env1, env2)

        env3 = Envelope()
        env4 = Envelope()
        ce2 = CompositeEnvelope(env3, env4)

        ce3 = CompositeEnvelope(ce1, ce2)

        self.assertEqual(len(ce3.envelopes), 4)
        for e in [env1, env2, env3, env4]:
            self.assertTrue(e in ce3.envelopes, msg="Added envelope should be in envelopes")
            self.assertEqual(e.composite_envelope, ce3)

    def test_composite_envelope_combine(self):
        env1 = Envelope()
        env2 = Envelope()
        ce1 = CompositeEnvelope(env1, env2)
        ce1.combine(env1.fock, env2.fock)
        ce1.combine(env1.polarization, env2.polarization)

        self.assertEqual(env1.fock.index, (0, 0))
        self.assertEqual(env2.fock.index, (0, 1))
        self.assertEqual(env1.polarization.index, (1, 0))
        self.assertEqual(env2.polarization.index, (1, 1))
        self.assertEqual(env1.composite_envelope, ce1)
        self.assertEqual(env2.composite_envelope, ce1)
        self.assertTrue(np.array_equal(ce1.states[1][0], [[1],[0],[0],[0]]))

        env3 = Envelope()
        ce1.add_envelope(env3)
        ce1.combine(env1.polarization, env3.polarization)
        self.assertTrue(np.array_equal(
            ce1.states[1][0],
            [[1],[0],[0],[0],[0],[0],[0],[0]]))
        self.assertEqual(env3.polarization.index, (1,2))

    def test_composite_combine(self):
        env1 = Envelope()
        env2 = Envelope()
        ce1 = CompositeEnvelope(env1, env2)
        ce1.combine(env1.fock, env2.fock)
        ce1.combine(env1.polarization, env2.polarization)

        env3 = Envelope()
        env4 = Envelope()
        ce2 = CompositeEnvelope(env3, env4)
        ce2.combine(env3.fock, env4.fock)
        ce2.combine(env3.polarization, env4.polarization)
        ce3 = CompositeEnvelope(ce1,ce2)
        self.assertTrue(np.array_equal(
            ce3.states[1][0],
            [[1], [0], [0], [0]]
            
        ))
        self.assertTrue(np.array_equal(
            ce3.states[3][0],
            [[1], [0], [0], [0]]
        ))
        ce3.combine(env1.polarization, env3.polarization)
        self.assertTrue(np.array_equal(
            ce3.states[1][0],
            [[1], [0], [0], [0], [0], [0], [0], [0],
             [0], [0], [0], [0], [0], [0], [0], [0]]
        ))
        self.assertEqual(len(ce3.states), 3, msg="Is the residual vector deleted?")

    def test_matrix_combine(self):
        env1 = Envelope()
        env2 = Envelope()
        env1.polarization.expand()
        env1.polarization.expand()
        ce1 = CompositeEnvelope(env1, env2)
        ce1.combine(env1.polarization, env2.polarization)
        self.assertTrue(np.array_equal(
            ce1.states[0][0],
            [[1,0,0,0],
             [0,0,0,0],
             [0,0,0,0],
             [0,0,0,0]]
        ))

    def test_precombined_envelope(self):
        env1 = Envelope()
        env1.fock.expand()
        env1.combine()
        env2 = Envelope()
        ce = CompositeEnvelope(env1, env2)
        ce.combine(env1.polarization, env2.polarization)
        self.assertTrue(np.array_equal(
            ce.states[0][0],
            [[1], [0], [0], [0],
             [0], [0], [0], [0],
             [0], [0], [0], [0]]
        ))

        env3 = Envelope()
        env3.fock.expand()
        env3.fock.expand()
        env3.combine()
        env4 = Envelope()
        ce1 = CompositeEnvelope(env3, env4)
        ce2 = CompositeEnvelope(ce1, ce)
        ce2.combine(env3.fock, env1.polarization)
        expected_vector = np.zeros(72)
        expected_vector[0] = 1
        expected_density_matrix = np.outer(
            expected_vector.flatten(),
            np.conj(expected_vector.flatten())
        )
        self.assertTrue(np.array_equal(
            ce2.states[0][0],
            expected_density_matrix
        ))


    def test_exception(self):
        with self.assertRaises(StateNotInThisCompositeEnvelopeException):
            env1 = Envelope()
            env2 = Envelope()
            ce = CompositeEnvelope(env1, env2)
            env3 = Envelope()
            ce.combine(env1.fock, env3.fock)
        with self.assertRaises(FockOrPolarizationExpectedException):
            env1 = Envelope()
            env2 = Envelope()
            ce = CompositeEnvelope(env1, env2)
            ce.combine(env1, env2)


    def test_composite_expansion(self):
        env1 = Envelope()
        env2 = Envelope()
        ce = CompositeEnvelope(env1, env2)
        ce.combine(env1.polarization, env2.polarization)
        ce.expand(env1.polarization)
        self.assertEqual(ce.states[0][0].shape[0], 4)
        self.assertEqual(ce.states[0][0].shape[1], 4)
        ce.expand(env1.polarization)

    def test_composite_reorder(self):
        p1 = Polarization(polarization=PolarizationLabel.V)
        p2 = Polarization(polarization=PolarizationLabel.H)
        env1 = Envelope(polarization=p1)
        env2 = Envelope(polarization=p2)
        env1.polarization.expand()
        env2.polarization.expand()
        state1 = env1.polarization.state_vector.copy()
        state2 = env2.polarization.state_vector.copy()

        ce = CompositeEnvelope(env1, env2)
        ce.combine(env1.polarization, env2.polarization)
        self.assertTrue(np.array_equal(
            ce.states[0][0],
            np.kron(state1, state2)
            ))
        self.assertEqual(ce.states[0][1],
                         [env1.polarization, env2.polarization])
        ce.rearange(env2.polarization, env1.polarization)
        self.assertTrue(np.array_equal(
            ce.states[0][0],
            np.kron(state2, state1)
            ))
        self.assertEqual(ce.states[0][1],
                         [env2.polarization, env1.polarization])
        ce.rearange(env1.polarization, env2.polarization)

    def test_composite_reorder_matrix(self):
        p1 = Polarization(polarization=PolarizationLabel.R)
        p1.expand()
        p1.expand()
        pol = p1.density_matrix.copy()
        env1 = Envelope(polarization=p1)
        env2 = Envelope()
        env1.fock.dimensions = 5
        env1.fock.expand()
        env1.fock.expand()
        f1 = env1.fock.density_matrix.copy()
        env2.fock.dimensions = 4
        env2.fock.expand()
        env2.fock.expand()
        f2 = env2.fock.density_matrix.copy()
        ce = CompositeEnvelope(env1, env2)
        ce.combine(env1.fock, env2.fock, env1.polarization)

        expected_matrix = np.kron(np.kron(f1, f2), pol)
        self.assertTrue(np.array_equal(
            expected_matrix,
            ce.states[0][0]
        ))
        self.assertEqual(
            ce.states[0][1],
            [env1.fock, env2.fock, env1.polarization]
        )
        ce.rearange(env1.polarization, env1.fock)

        expected_matrix = np.kron(np.kron(pol, f1), f2)
        self.assertTrue(np.array_equal(
            expected_matrix,
            ce.states[0][0]
        ))
        self.assertEqual(
            ce.states[0][1],
            [env1.polarization, env1.fock, env2.fock]
        )

    def test_fock_operations_on_composite_envelope(self):
        env1 = Envelope()
        env2 = Envelope()
        ce = CompositeEnvelope(env1, env2)
        op = FockOperation(FockOperationType.Creation)
        
        ce.apply_operation(op, env1)
