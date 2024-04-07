import numpy as np
from .expansion_levels import ExpansionLevel

class FockOrPolarizationExpectedException(Exception):
    pass

class StateNotInThisCompositeEnvelopeException(Exception):
    pass


def redirect_if_consumed(method):
    def wrapper(self, *args, **kwargs):
        # Check if the object has been consumed by another CompositeEnvelope
        if hasattr(self, '_consumed_by') and self._consumed_by:
            # Redirect the method call to the new CompositeEnvelope
            return getattr(self._consumed_by, method.__name__)(*args, **kwargs)
        else:
            return method(self, *args, **kwargs)
    return wrapper


class CompositeEnvelope:
    def __init__(self, *envelopes):
        from photon_weave.state.envelope import Envelope
        self.envelopes = []
        self.states = []
        # If the state is consumed by another composite state, the reference is stored here
        self._consumed_by = None
        seen_composite_envelopes = set()
        for e in envelopes:
            if isinstance(e, CompositeEnvelope):
                if e not in seen_composite_envelopes:
                    self.envelopes.extend(e.envelopes)
                    self.states.extend(e.states)
                    seen_composite_envelopes.add(e)
            elif isinstance(e, Envelope):
                if (not e.composite_envelope is None and
                    e.composite_envelope not in seen_composite_envelopes):
                    self.states.extend(e.composite_envelope.states)
                    seen_composite_envelopes.add(e.composite_envelope)
                    self.envelopes.extend(e.composite_envelope.envelopes)
                else:
                    self.envelopes.append(e)
        for ce in seen_composite_envelopes:
            ce._consumed_by = self
            ce.states = []

        self.envelopes = list(set(self.envelopes))
        self.update_indices() 
        for e in self.envelopes:
            e.composite_envelope = self

    @redirect_if_consumed
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

    @redirect_if_consumed
    def update_indices(self):
        for major, _ in enumerate(self.states):
            for minor, state in enumerate(self.states[major][1]):
                state.set_index(minor, major)

    @redirect_if_consumed
    def add_envelope(self, envelope):
        self.envelopes.append(envelope)
        envelope.composite_envelope = self

    @redirect_if_consumed
    def expand(self, state):
        if state.envelope.expansion_level >= ExpansionLevel.Matrix:
            return
        state_index = None
        for i, s in enumerate(self.states):
            if state in s[1]:
                state_index = i
                break
        self.states[state_index][0] = np.outer(
            self.states[state_index][0].flatten(),
            np.conj(self.states[state_index][0].flatten()),
        )
        for s in self.states[state_index][1]:
            s.expansion_level = ExpansionLevel.Matrix

    @redirect_if_consumed
    def _find_composite_state_index(self, *states):
        composite_state_index = None
        for i, (_, states_group) in enumerate(self.states):
            if all(s in states_group for s in states):
                composite_state_index = i
                return composite_state_index
        return None

    @redirect_if_consumed
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

    @redirect_if_consumed
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

    @redirect_if_consumed
    def apply_operation(self, operation, *states):
        from photon_weave.operation.fock_operation import FockOperation
        from photon_weave.operation.polarization_operations import PolarizationOperation
        from photon_weave.operation.composite_operation import CompositeOperation
        from photon_weave.state.envelope import Envelope
        from photon_weave.state.composite_envelope import CompositeEnvelope
        csi = self._find_composite_state_index(states[0])
        if (isinstance(operation, FockOperation) or
            isinstance(operation, PolarizationOperation)):
            if csi is None:
                states[0].apply_operation(operation)
            else:
                self._apply_operator(operation, *states)
        elif isinstance(operation, CompositeOperation):
            operation.operate(*states)

    @redirect_if_consumed
    def _apply_operator(self, operation, *states):
        """
        Assumes the spaces are correctly ordered
        """
        from photon_weave.state.fock import Fock
        from photon_weave.state.polarization import Polarization
        from photon_weave.operation.fock_operation import (
            FockOperation, FockOperationType)
        from photon_weave.operation.polarization_operations import (
            PolarizationOperation, PolarizationOperationType)
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
                elif isinstance(state, Polarization):
                    identity = PolarizationOperation(
                        PolarizationOperationType.I)
                    identity.compute_operator()
                    identity = identity.operator
                composite_operator = np.kron(
                    composite_operator,
                    identity)

        if self.states[csi][1][0].expansion_level == ExpansionLevel.Vector:
            self.states[csi][0] = composite_operator @ self.states[csi][0]
        elif self.states[csi][1][0].expansion_level == ExpansionLevel.Matrix:
            self.states[csi][0] = composite_operator @ self.states[csi][0]
            self.states[csi][0] = self.states[csi][0] @ composite_operator.conj().T

    @redirect_if_consumed
    def measure(self, *states) -> int:
        from photon_weave.state.envelope import Envelope
        from photon_weave.state.polarization import Polarization
        from photon_weave.state.fock import Fock
        outcome = None
        nstates = []
        for s in states:
            if isinstance(s, Envelope):
                nstates.append(s)
            elif isinstance(s, Polarization) or isinstance(s, Fock):
                nstates.append(s.envelope)
        states = list(set(nstates))
        for s in states:
            if isinstance(s, Envelope):
                if s not in self.envelopes:
                    raise StateNotInThisCompositeEnvelopeException()
                if (self._find_composite_state_index(s.fock) is None and
                    self._find_composite_state_index(s.polarization) is None):
                    outcome = s.measure()
                    self.envelopes.remove(s)
                    s.composite_envelope = None
                else:
                    pol = s.polarization.index
                    fock = s.fock.index
                    # measure fock state
                    cutoffs = [s.dimensions for s in self.states[fock[0]][1]]
                    probabilities = []
                    projection_states = []
                    if s.fock.expansion_level == ExpansionLevel.Vector:
                        before = np.eye(int(np.prod(cutoffs[:fock[1]])))
                        after = np.eye(int(np.prod(cutoffs[fock[1]+1:])))
                        for num_particles in range(s.fock.dimensions):
                            projection = np.zeros(
                                (
                                    s.fock.dimensions,
                                    s.fock.dimensions
                                )
                            )
                            projection[num_particles, num_particles] = 1
                            full_projection = np.kron(
                                np.kron(before, projection),
                                after)
                            projected_state = full_projection @ self.states[fock[0]][0]
                            prob = np.linalg.norm(projected_state)**2
                            probabilities.append(prob)
                            projection_states.append(projected_state)
                    outcome = np.random.choice(
                        range(s.fock.dimensions),
                        p=probabilities)
                    self.states[fock[0]][0] = projection_states[outcome]
                    self.states[fock[0]][0] /= np.linalg.norm(projection_states[outcome])
                    # Removing the space from the product space
                    self._trace_out(s.fock)
                    s._set_measured()
        self.update_indices()
        return outcome

    def _trace_out(self, state):
        space_index, subsystem_index = state.index
        dims = [s.dimensions for s in self.states[space_index][1]]
        if len(self.states[space_index][1]) < 2:
            return
        if state.expansion_level < ExpansionLevel.Matrix:
            self.expand(state)
        n = len(dims)
        letters = "bcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        rho = self.states[space_index][0]

        input_str = ""
        output_str = ""
        reshape_dims = (*dims, *dims)
        c_id = 0
        for i in range(len(reshape_dims)):
            if i%len(dims) == subsystem_index:
                input_str+="a"
            else:
                char = letters[c_id]
                input_str+=char
                output_str+=char
                c_id += 1
                

        einsum_str = f"{input_str}->{output_str}"

        rho = rho.reshape(reshape_dims)
        rho = np.einsum(einsum_str, rho)
        new_dims = np.prod(dims) // state.dimensions
        rho = rho.reshape(new_dims, new_dims)
        self.states[space_index][0]=rho

        self.contract(state)
        # Update system information post trace-out
        del self.states[space_index][1][subsystem_index]
        

    @redirect_if_consumed
    def contract(self, state):
        if state.expansion_level < ExpansionLevel.Matrix:
            return
        space_index = state.index[0]
        # Assuming self.states[space_index] correctly references the density matrix
        rho = self.states[space_index][0]
        # Square the density matrix before taking the trace
        tr_rho_squared = np.trace(np.dot(rho, rho))
        if not np.isclose(tr_rho_squared, 1.0):
            return
        eigenvalues, eigenvectors = np.linalg.eigh(rho)
        # Assuming you want to keep the eigenvector corresponding to the largest eigenvalue
        # And assuming the structure of self.states allows direct replacement
        vector = eigenvectors[:, np.argmax(eigenvalues)].reshape(-1, 1)
        # Update the state with the column vector
        self.states[space_index][0] = vector
        
        # This line seems to attempt to update expansion_level for multiple states,
        # Ensure the structure of self.states[space_index][1] (if it exists) supports iteration like this
        for s in self.states[space_index][1]:
            s.expansion_level = ExpansionLevel.Vector
