# Photon Weave Usage Examples

## Mach-Zehnder Interferometer
[mach_zehnder_interferometer.py](./mach_zehnder_interferometer.py)

This example demonstrates the simulation of a Mach-Zehnder Interferometer (MZI), a key optical component used in quantum computing and photonics research. The MZI splits an input photon beam into two paths, introduces phase shifts, and then recombines the paths to produce interference patterns.

The diagram below illustrates the simulated output from the Mach-Zehnder Interferometer:

![Mach-Zehnder Interferometer](plots/mzi.png)

## Time-Bin Encoding
[time_bin_encoding.py](./time_bin_encoding.py)

This example contains a PhotonWeave implementation of Time-Bin Encoding scheme for qubits using tho MZIs. The qubit is encoded into optical pulses that travel through paths of different lengths within the interferometers, creating a time-bin encoding for quantum information. The pulses in each interferometer undergo a beam splitter operation and a phase shift, affecting their final probability distribution.

The simulation models the evolution of the photon states as they pass through the interferometers. The first MZI introduces a controlled superposition of early and late time bins, while the second MZI can be used to recombine the pulses, depending on the phase shifts applied.


![Time-Bin Encoding](plots/time_bin_encoding.png)

## Polarizing Beam Splitter

[polarizing_beam_splitter.py](./polarizing_beam_splitter.py)

In quantum optics, a Polarizing Beam Splitter (PBS) is a device that routes single photons based on their polarization state. It transmits photons with horizontal polarization and reflects those with vertical polarization. When a photon is in a quantum superposition of horizontal and vertical polarizations, the photon is detected at either the transmitted or reflected port with probabilities determined by its polarization amplitudes.

In this example, three diagonally polarized photons are input into Port 1 of the PBS. The figure below shows the probability distribution of detecting different combinations of horizontally and vertically polarized photons at the output ports.

![Polarizing Beam Splitter](plots/pbs.png)
