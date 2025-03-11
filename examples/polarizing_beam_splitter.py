from photon_weave.state.composite_envelope import CompositeEnvelope
from photon_weave.state.envelope import Envelope
from photon_weave.state.polarization import PolarizationLabel
from photon_weave.operation.operation import Operation
from photon_weave.operation.composite_operation import CompositeOperationType
from photon_weave.state.fock import Fock, FockOperationType
from photon_weave.state.polarization import Polarization  
from photon_weave.state.polarization import PolarizationOperationType
import random
import jax.numpy as jnp
from typing import Optional
from photon_weave.state.expansion_levels import ExpansionLevel
from photon_weave.state.base_state import BaseState

def  PBS(env1:Optional[Envelope] = None, env2: Optional[Envelope] = None) -> dict[str, Envelope]:
    for env in [env1, env2]:
        if env:
            assert env1.polarization.index == None, "Polarization must not be in a product state"
        else:
            env1 = Envelope()
    env3 = Envelope()
    env3.polarization.state = PolarizationLabel.V
    env4 = Envelope()
    env4.polarization.state = PolarizationLabel.V

    #Operations
    H = jnp.array([[1],[0]])
    V = jnp.array([0,1])
    P_HH = jnp.array([[1,0], [0,0]])  #H -> H
    P_VV = jnp.array([[0,0], [0,1]])  #V -> V 
    P_HV = jnp.array([[0,1], [0,0]])  #V -> H
    P_VH = jnp.array([[0,0], [1,0]])  #H -> V
    op_p_HH = Operation(PolarizationOperationType.Custom, operator = P_HH)
    op_p_VV = Operation(PolarizationOperationType.Custom, operator = P_VV)
    op_p_VH = Operation(PolarizationOperationType.Custom, operator = P_VH)
    op_p_HV = Operation(PolarizationOperationType.Custom, operator = P_HV)

    #Computing Beam Splitter Angle Port 1
    if env1.polarization.expansion_level == ExpansionLevel.Label: 
        env1.polarization.expand()
    if env1.polarization.expansion_level == ExpansionLevel.Vector:
        H_inner_1 = abs(jnp.vdot(H, env1.polarization.state)) ** 2
        env1.polarization.contract()
    elif env1.polarization.expansion_level == ExpansionLevel.Matrix:
        H_inner_1 = H.conj().T @ (env1.polarization.state @ H)
        H_inner_1 = float(abs(H_inner_1[0][0]))
        
    env1.polarization.apply_kraus([P_HH, P_HV])   
    eta_1 = (1-H_inner_1) * jnp.pi/2
  
    #Computing Beam Splitter Angle Port 2
    if env2.polarization.expansion_level == ExpansionLevel.Label: 
        env2.polarization.expand()
    if env2.polarization.expansion_level == ExpansionLevel.Vector:
        H_inner_2 = abs(jnp.vdot(H, env2.polarization.state)) ** 2
        env2.polarization.contract()
    elif env2.polarization.expansion_level == ExpansionLevel.Matrix:
        H_inner_2 = H.conj().T @ (env2.polarization.state @ H)
        
    env2.polarization.apply_kraus([P_HH, P_HV])
    eta_2 = (1-H_inner_2) * jnp.pi/2

    #Creating Beam Splitting Operations
    op_1 = Operation(CompositeOperationType.NonPolarizingBeamSplitter, eta = eta_1)
    op_2 = Operation(CompositeOperationType.NonPolarizingBeamSplitter, eta = eta_2)

    #Applying Beam Splitting Operations
    ce = CompositeEnvelope(env1, env2, env3, env4)
    ce.apply_operation(op_1, env1.fock, env3.fock)
    ce.apply_operation(op_2, env2.fock, env4.fock)


    return {
       '1H': env1, '1V': env4,
       '2H': env2, '2V': env3
    }
        
if __name__ == "__main__":
   env1 = Envelope()
   env1.fock.state = 1
   env1.polarization.state = PolarizationLabel.R
   env1.polarization.expand()
   env2 = Envelope()
   out = PBS(env1, env2)
   for port, env in out.items():
        print(port, env.polarization)
        state = env.fock.trace_out()
        one=jnp.abs(state[1][0])**2
        print(one)
