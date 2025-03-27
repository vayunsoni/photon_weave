# Photon Weave Usage Examples

## Mach-Zehnder Interferometer
[mach_zehnder_interferometer.py](./mach_zehnder_interferometer.py)

This example demonstrates the simulation of a Mach-Zehnder Interferometer (MZI), a key optical component used in quantum computing and photonics research. The MZI splits an input photon beam into two paths, introduces phase shifts, and then recombines the paths to produce interference patterns.

The diagram below illustrates the simulated output from the Mach-Zehnder Interferometer:

![Mach-Zehnder Interferometer](plots/mzi.png)

## Polarizing Beam Splitter

[polarizing_beam_splitter.py](./polarizing_beam_splitter.py)

A Polarizing Beam Splitter (PBS) is a type of beam splitter that splits a beam of light into two beams with orthogonal polarizations. It is widely used in optics and photonics to manipulate and analyze polarized light. A Horizontally polarized photon is transmitted while a vertically polarized photon is reflected. If a photon is in a superposition state of both horizontal and vertical polarizations, the photon is obsevered at either the horizontal or vertical port with a certain probability, depending on its polarization.

In this example, three diagonally polarized photons are input into Port 1 of the PBS. The figure below shows the probability distribution of detecting different combinations of horizontally and vertically polarized photons at the output ports.

![Polarizing Beam Splitter](plots/pbs.png)
