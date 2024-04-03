import numpy as np
from .expansion_levels import ExpansionLevel

class FockOrPolarizationExpectedException(Exception):
    pass

class StateNotInThisCompositeEnvelopeException(Exception):
    pass


class CompositeEnvelope:
    def __init__(self, *envelopes):
        from photon_weave.state.envelope import Envelope
        self.envelopes = []
        self.states = []
        for e in envelopes:
            if isinstance(e, CompositeEnvelope):
                self.envelopes.extend(e.envelopes)
                for env in e.envelopes:
                    env.composite_envelope = self
                for state in e.states:
                    self.states.append(state)
                del(e)
            elif isinstance(e, Envelope):
                self.envelopes.append(e)
        self.update_indices() 
        for e in envelopes:
            e.composite_envelope = self

    def combine(self, *states):
        """
        Combines states into a product space
        """
        from photon_weave.state.fock import Fock
        from photon_weave.state.polarization import Polarization
        for s in states:
            if not (isinstance(s, Fock) or isinstance(s, Polarization)):
                raise FockOrPolarizationExpectedException()
            included_states = []
            for env in self.envelopes:
                included_states.append(env.fock)
                included_states.append(env.polarization)
            if not any(s is state for state in included_states):
                raise StateNotInThisCompositeEnvelopeException()
                
        existing_product_states = []
        tmp = []
        for state in states:
            for i,s in enumerate(self.states):
                if state in s[1]:
                    existing_product_states.append(i)
                    tmp.append(state)

        new_product_states = [s for s in states if s not in tmp]

        # Combine
        if len(existing_product_states) > 0:
            target_ps = existing_product_states[0]
            existing_product_states.pop(0)
        else:
            target_ps = len(self.states)

        ## First combine existing spaces
        expected_expansion = max([s.expansion_level for s in states])
        if expected_expansion < ExpansionLevel.Vector:
            expected_expansion = ExpansionLevel.Vector

        ## Correct expansion in existing product spaces
        if target_ps != len(self.states):
            if self.states[target_ps][1][0].expansion_level < expected_expansion:
                self.states[target_ps][0] = np.outer(
                    self.states[target_ps][0].flatten(),
                    np.conj(self.states[target_ps][0].flatten()),
                )
        else:
            self.states.append([1, []])
                
        for i in existing_product_states:
            if self.states[i][1][0].expansion_level < expected_expansion:
                self.states[i][0] = np.outer(
                    self.states[i][0].flatten(),
                    np.conj(self.states[i][0].flatten()),
                )

        for eps in existing_product_states:
            existing = self.states[eps]
            self.states[target_ps][0] = np.kron(
                self.states[target_ps][0],
                existing[0])
            self.states[target_ps][1].extend(existing[1])
        ## Second combine new spaces
        for nps in new_product_states:
            while nps.expansion_level < expected_expansion:
                nps.expand()
            if nps.expansion_level == ExpansionLevel.Vector:
                if nps.state_vector is not None:
                    self.states[target_ps][0] = np.kron(
                        self.states[target_ps][0],
                        nps.state_vector)
                    nps.state_vector = None
                    self.states[target_ps][1].append(nps)
                else:
                    self.states[target_ps][0] = np.kron(
                        self.states[target_ps][0],
                        nps.envelope.composite_vector
                    )
                    indices = [None, None]
                    indices[nps.envelope.fock.index] = nps.envelope.fock
                    indices[nps.envelope.polarization.index] = nps.envelope.polarization
                    nps.envelope.composite_vector = None
                    self.states[target_ps][1].append(nps)
            elif nps.expansion_level == ExpansionLevel.Matrix:
                if nps.density_matrix is not None:
                    self.states[target_ps][0] = np.kron(
                        self.states[target_ps][0],
                        nps.density_matrix)
                    nps.density_matrix = None
                    self.states[target_ps][1].append(nps)
                else:
                    self.states[target_ps][0] = np.kron(
                        self.states[target_ps][0],
                        nps.envelope.composite_matrix
                    )
                    indices = [None, None]
                    indices[nps.envelope.fock.index] = nps.envelope.fock
                    indices[nps.envelope.polarization.index] = nps.envelope.polarization
                    self.states[target_ps][1].append(nps)
        # Delete old product spaces
        for index in sorted(existing_product_states, reverse=True):
            del self.states[index]
        self.update_indices()

    def update_indices(self):
        for major, _ in enumerate(self.states):
            for minor, state in enumerate(self.states[major][1]):
                state.set_index(minor, major)

    def add_envelope(self, envelope):
        self.envelopes.append(envelope)
        envelope.composite_envelope = self

    def expand(self, state):
        if state.envelope.expansion_level >= ExpansionLevel.Matrix:
            return
        state_index = None
        for i,s in enumerate(self.states):
            if state in s[1]:
                state_index = i
                break
        self.states[state_index][0] = np.outer( 
            self.states[state_index][0].flatten(), 
            np.conj(self.states[state_index][0].flatten()), 
        ) 
        for s in self.states[state_index][1]:
            s.expansion_level = ExpansionLevel.Matrix

    def _find_composite_state_index(self, *states):
        composite_state_index = None
        for i, (_, states_group) in enumerate(self.states):
            if all(s in states_group for s in states):
                composite_state_index = i
                return composite_state_index

        if composite_state_index is None:
            raise ValueError("Specified states do not match any composite state.")

        

    def rearange(self, *ordered_states):
        """
        Uses the swap operation to rearange the states, according to the given order
        The ordered states must already be in the same product
        """
        composite_state_index = None
        for i, (_, states_group) in enumerate(self.states):
            if all(s in states_group for s in ordered_states):
                composite_state_index = i
                break

        if composite_state_index is None:
            raise ValueError("Specified states do not match any composite state.")

        # Check if the states are already in the desired order
        current_order = self.states[composite_state_index][1]
        if all(ordered_states[i] == current_order[i]
               for i in range(min(len(ordered_states), len(current_order)))):
            return

        dimensions = [state.dimensions for state in self.states[composite_state_index][1]]

        new_order = self.states[composite_state_index][1].copy()
        for idx, ordered_state in enumerate(ordered_states):
            if new_order.index(ordered_state) != idx:
                tmp = new_order[idx]
                old_idx = new_order.index(ordered_state)
                new_order[idx] = ordered_state
                new_order[old_idx] = tmp
        self._reorder_states(new_order, composite_state_index)

    def _reorder_states(self, order, state_index):
        # Calculate the total dimension of the composite system
        total_dim = np.prod([s.dimensions for s in self.states[state_index][1]])
        
        # Calculate the current (flattened) index for each subsystem
        current_dimensions = [s.dimensions for s in self.states[state_index][1]]
        target_dimensions = [s.dimensions for s in order]  # The new order's dimensions
        
        # Initialize the permutation matrix
        permutation_matrix = np.zeros((total_dim, total_dim))
        
        # Calculate new index for each element in the flattened composite state
        for idx in range(total_dim):
            # Determine the multi-dimensional index in the current order
            multi_idx = np.unravel_index(idx, current_dimensions)
            
            # Map the multi-dimensional index to the new order
            new_order_multi_idx = [multi_idx[self.states[state_index][1].index(s)] for s in order]
            
            # Calculate the linear index in the new order
            new_idx = np.ravel_multi_index(new_order_multi_idx, target_dimensions)
            
            # Update the permutation matrix
            permutation_matrix[new_idx, idx] = 1

        if order[0].expansion_level == ExpansionLevel.Vector:
            self.states[state_index][0] = permutation_matrix @ self.states[state_index][0]
            self.states[state_index][1] = order
        elif order[0].expansion_level == ExpansionLevel.Matrix:
            self.states[state_index][0] = permutation_matrix @ self.states[state_index][0] @ permutation_matrix.conj().T
            self.states[state_index][1] = order

    def apply_operation(self, operation, *states):
        from photon_weave.operation.fock_operation import FockOperation

    def apply_operator(self, operation, *states):
        """
        Assumes the spaces are correctly ordered
        """
        from photon_weave.state.fock import Fock
        from photon_weave.state.polarization import Polarization
        from photon_weave.operation.fock_operation import FockOperation, FockOperationType
        from photon_weave.operation.polarization_operations import PolarizationOperation, PolarizationOperationType
        csi = self._find_composite_state_index(*states)
        composite_operator = 1
        skip_count = 0
        for i, state in enumerate(self.states[csi][1]):
            if skip_count > 0:
                skip_count -= 1
                continue
            if all(state is self.states[csi][1][i+j] for j, state in enumerate(states)):
                if operation.operator is None:
                    operation.compute_operator(state.dimensions)
                composite_operator = np.kron(
                    composite_operator, operation.operator)
                skip_count += len(states)-1
            else:
                identity = None
                if isinstance(state, Fock):
                    identity = FockOperation(FockOperationType.Identity)
                    identity.compute_operator(state.dimensions)
                    identity = identity.operator
                elif isinstance(state, Polarization()):
                    identity = PolarizationOperation(PolarizationOperationType.Identity)
                    identity.compute_operator(state.dimensions)
                    identity = identity.operator
                composite_operator = np.kron(
                    composite_operator,
                    identity)

        if self.states[csi][1][0].expansion_level == ExpansionLevel.Vector:
            self.states[csi][0] = composite_operator @ self.states[csi][0]
        elif self.states[csi][1][0].expansion_level == ExpansionLevel.Matrix:
            self.states[csi][0] = composite_operator @ self.states[csi][0]
            self.states[csi][0] = self.states[csi][0] @ composite_operator.conj().T
                    
