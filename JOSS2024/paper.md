---
title: "Photon-Weave"
tags:
  - Python
  - Quantum Optics Simulation
  - Quantum
authors:
  - name: Simon Sekavčnik
    orcid: 0000-0002-1370-9751
    affiliation: 1
  - name: Kareem El-Safty
    orcid: 0000-0001-8740-0637
    affiliation: 1
  - name: Janis Nötzel
    orcid: 0000-0003-0091-3072
    affiliation: 1
affiliation:
  - name: Techincal University of Munich, TQSD, Arcißstraße 21
    index: 1
date: 22.09.2024
bibliography: paper.bib
---

# Summary
Photon Weave is a quantum systems simulator designed to offer intuitive abstractions for simulating photonic quantum systems and their interactions in Fock space along with the any custom Hilbert space. The simulator focuses on simplifying complex quantum state representations, such as photon pulses (envelopes) with polarization, making it more approachable for specialized quantum simulations. While general-purpose quantum simulation libraries such as QuTiP [@johansson2012qutip]provide robust tools for quantum state manipulations, they often require metiulous organization of operations for larger simulations, introucing complexity that can be automated. Photon Weave addresses this by abstracting such details, streamlining the simulation process, and allowing quantum systems to interact naturally as the simulation progresses.

Unlike frameworks such as Qiskit [@wille2019ibm], whic hare typically tailored for qubit-based computations, Photon Weave exels in simulating sontinuous-variable quanntu, systems, particularly photons, as well as custom quantum states thac can interact dynamically. Furthermore, Photon Weave offers a balance of flexibility and automation by deferring the joining of quantum spaces until it is necessary, enhancing computational efficiency. The simulator supports both CPU and GPU execution, ensuring scalability and performance for large-scale simulations.

# Statement of Need
The field of quantum optics and quantum systems simulation has evolved rapidly, with tools such QuTiP, Qiskit, and Strawberry Fields [killoran2019strawberry] offering frameworks to model quantum phenomena. However many existing simulator either provide too general an approach, as seen in QuTiP, where extensive user control is required to manage state transformations, or impose rigid structures like circuits, as in Strawberry Fields. Researachers working on quantum optics, quantum informaion, or other related fields need a tool that simplifies the simulation of photonic systems, allows for the dynamic interaction of custom quantum systems, and does not require a circuit model. Photon Weave fulfills this need by offering high-level abstractions like `Fock`, `Polarization`, `Envelope`, `CustomState` and `CompositeEnvelope`. An example of usecase of this tool is a model of quantum device, which can easily be then used in another setup.


# Background

# Photon Weave Overview
Photon Weave is a quantum simulation library, allowing for simulation of any system, provided that underlying hardware resource requirements are met. With this simulator one can create, operate on, and perform different measurements on the quantum systems. 

## Photon Weave Implementation Details

### State Containers
Photon Weave logic is built around quantum state containers. State can be represented in three different ways: `Label`, `Vector` or `Matrix`, which progressively require more memory. The representations are handled automatically by the Photon Weave and are by default "shrunk" if appilcable. The framework ships with the following state containers `Fock`, `Polarization`, `Envelope`, `CompositeEnvelope` and `CustomState`. `Fock`, `Polarization` and `CustomState` are rudimentary state containers, holding the state (in any representation) as long as the state is not joined. When the state is joined, these containers hold a reference to the `Envelope`, `CompositeEnvelope` or both. In essence, each rudimentary container understands where its system is and how it is tensored in a product space.

### Envelopes
Photon Weave places a special focus on so called `Envelope`. An `Envelope` represents a pulse of light, where all photons are indistinguishable and have the same polarization, representing $`\mathcal{F}\otimes\mathcal{P}`$ space. At the creation time, when the spaces are separable their states are contained in the respective `Fock` and `Polarization` instances. Along with the states, the `Envelope` contains additional information in the form of wavelength as well as temporal profile. 

### CompositeEnvelopes



# Conclusion
We offer Photon Weave to an audience with a need for easy to use open source quantum systems simulator under Apache-2.0 license. On of the intended outcomes is to build interoperable library of interoperable quantum device descriptions enabled by the Photon Weave framework.

# References
