from photon_weave.state.envelope import Envelope
from photon_weave.state.fock import Fock
from photon_weave.state.polarization import (
    Polarization, PolarizationLabel)
from photon_weave.operation.fock_operation import (
    FockOperation, FockOperationType)
import numpy as np
import random
import unittest

class TestFock(unittest.TestCase):
    def test_envelope_initiation(self):
        envelope = Envelope()
        self.assertIsNone(envelope.composite_vector)
        self.assertIsNone(envelope.composite_matrix)
        self.assertEqual(envelope.fock, Fock())

        fock = Fock()
        fock.label=1
        polarization = Polarization(PolarizationLabel.R)
        custom_envelope = Envelope(fock=fock, polarization=polarization)
        self.assertEqual(custom_envelope.polarization, polarization)
        self.assertEqual(custom_envelope.fock, fock)

    def test_envelope_combine(self):
        envelope = Envelope()
        envelope.combine()
        aux_fock = Fock()
        aux_fock.expand()
        aux_pol = Polarization()
        aux_pol.expand()

        expected_vector = np.kron(aux_fock.state_vector, aux_pol.state_vector)
        self.assertTrue(np.array_equal(
            envelope.composite_vector,
            expected_vector
        ))
        self.assertEqual(envelope.fock.index, 0)
        self.assertEqual(envelope.polarization.index, 1)
        self.assertIsNone(envelope.fock.label)
        self.assertIsNone(envelope.fock.state_vector)
        self.assertIsNone(envelope.fock.density_matrix)
        self.assertIsNone(envelope.polarization.label)
        self.assertIsNone(envelope.polarization.state_vector)
        self.assertIsNone(envelope.polarization.density_matrix)

        envelope = Envelope()
        envelope.fock.expand()
        envelope.fock.expand()
        envelope.polarization.expand()


    def test_measure_envelope_fock(self):
        """
        Test measurement, where the fock state is not
        combined into an envelope or composite envelope.
        """
        r = random.randint(1, 20)
        env = Envelope()
        op = FockOperation(FockOperationType.Creation, apply_count=r)
        env.apply_operation(op)
        outcome = env.measure()
        self.assertEqual(outcome, r)
        self.assertEqual(env.measured, True)
        self.assertEqual(env.fock.measured, True)
        self.assertEqual(env.fock.label, None)

        env = Envelope()
        env.fock.expand()
        env.apply_operation(op)
        outcome = env.measure()
        self.assertEqual(outcome, r)
        self.assertEqual(env.measured, True)
        self.assertEqual(env.fock.measured, True)
        self.assertEqual(env.fock.state_vector, None)

        env = Envelope()
        env.fock.expand()
        env.fock.expand()
        env.apply_operation(op)
        outcome = env.measure()
        self.assertEqual(outcome, r)
        self.assertEqual(env.measured, True)
        self.assertEqual(env.fock.measured, True)
        self.assertEqual(env.fock.density_matrix, None)

    def test_envelope_measurement(self):
        """
        Test measurement, when state is combined into an envelope
        """
        r = random.randint(1,10)
        env = Envelope()
        op = FockOperation(FockOperationType.Creation, apply_count=r)
        env.apply_operation(op)
        env.combine()
        outcome = env.measure()
        self.assertEqual(outcome, r)
        self.assertEqual(env.measured, True)
        self.assertEqual(env.composite_vector, None)
        self.assertEqual(env.composite_matrix, None)
        self.assertEqual(env.polarization.measured, True)
        self.assertEqual(env.polarization.label, None)
        self.assertEqual(env.polarization.state_vector, None)
        self.assertEqual(env.polarization.density_matrix, None)
        self.assertEqual(env.fock.measured, True)
        self.assertEqual(env.fock.label, None)
        self.assertEqual(env.fock.state_vector, None)
        self.assertEqual(env.fock.density_matrix, None)


if __name__ == '__main__':
    unittest.main()
