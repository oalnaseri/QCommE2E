# Channel Models

This document summarizes the mathematical form and the intended interpretation of
the benchmark channel implementations in `quantum_comm_sim.channels`.

The current benchmark suite is designed for end-to-end simulator interpretability,
not for full electromagnetic or wave-optics propagation. In practice that means:

- each channel acts on one density matrix at a time,
- most channels are memoryless across symbols,
- the discrete-variable baseline uses low-dimensional surrogate state spaces,
- optical channels are reduced models that preserve the key impairment mechanism
	without reproducing every propagation effect.

## Common Interface

All benchmark channels expose a single method:

`rho_out = channel.apply(rho_in)`

The input is a density matrix. Most channels preserve dimension, but the erasure
channel expands the state space by adding an orthogonal flag state.

## Depolarizing Channel

`DepolarizingChannel` implements the standard qubit depolarizing map

`E(rho) = (1 - p) rho + p I / 2`

Interpretation:

- `p` is the depolarizing probability.
- the Bloch vector is contracted isotropically by a factor of `1 - p`.
- this is a memoryless qubit benchmark channel.

Assumptions and limits:

- valid only for `2 x 2` inputs,
- does not model any preferred axis or structured noise spectrum,
- appropriate as a textbook discrete-variable baseline, not as a fiber or free-space physical model.

## Dephasing Channel

`DephasingChannel` implements phase damping rather than a Pauli-`Z` phase-flip
mixture. For a qubit density matrix,

- diagonal terms are unchanged,
- off-diagonal terms are multiplied by `1 - p`.

Equivalently,

`rho_01 -> (1 - p) rho_01` and `rho_10 -> (1 - p) rho_10`

Interpretation:

- `p` controls coherence loss while preserving populations,
- this is the correct reduced model when the intended impairment is decoherence
	without energy exchange.

Assumptions and limits:

- qubit-only,
- memoryless across uses,
- not a frequency-selective or time-correlated phase-noise model.

## Erasure Channel

`ErasureChannel` implements a flagged quantum erasure map. For a `d`-dimensional
input state `rho`, the output lives in dimension `d + 1`:

`E(rho) = (1 - p) rho (+) p |e><e|`

where `|e>` is an erasure flag orthogonal to the original code space.

Interpretation:

- with probability `1 - p`, the state remains in the original subspace,
- with probability `p`, all information is moved to an orthogonal flag state,
- the benchmark receiver exposes this as label `-1`.

Assumptions and limits:

- this is a true flagged erasure model, not a same-space depolarizing surrogate,
- the erased branch does not retain residual information about the original symbol,
- BER and SER in the current benchmark count erasures as hard failures.

## Bosonic Thermal-Loss Channel

`BosonicChannel` implements a single-mode thermal-loss model in a truncated Fock
space. The system mode is mixed with a thermal environment on a beamsplitter of
transmissivity

`eta = 10^(-loss_db / 10)`

and the environment is then traced out.

Implementation details:

- the environment state is a truncated thermal state parameterized by `noise_factor`,
- the channel is realized through a finite-dimensional Stinespring picture,
- a pure-loss helper is also exposed and reused by the turbulence model.

Interpretation:

- `loss_db` controls attenuation,
- `noise_factor` is the mean thermal occupancy used to build the environment state,
- in `dim = 2`, the model reduces to the vacuum / one-photon subspace, which is a
	compact surrogate rather than a full continuous-variable simulation.

Assumptions and limits:

- single mode only,
- finite Fock truncation on both system and environment,
- truncation error grows when the relevant photon-number support exceeds the chosen dimension,
- does not model multimode coupling, detector saturation, or propagation memory.

## FSO Turbulence Channel

`TurbulenceChannel` models atmospheric turbulence as random transmissivity followed
by bosonic pure loss. The transmissivity sample combines:

- a Gamma-Gamma or Malaga-inspired scintillation surrogate,
- a pointing-loss factor derived from the beam waist and pointing error,
- clipping to the physically meaningful interval `[0, 1]`.

After sampling transmissivity `eta`, the channel applies the bosonic pure-loss map
for that `eta`.

Interpretation:

- weak turbulence corresponds to transmissivity fluctuating around larger values,
- stronger turbulence broadens the fading distribution and increases deep fades,
- the model captures intensity fading in a form suitable for Monte Carlo end-to-end benchmarks.

Assumptions and limits:

- no explicit phase-screen propagation,
- no spatial-mode distortion or modal crosstalk,
- no temporal correlation across symbols,
- `link_distance_m` is currently descriptive metadata rather than a parameter that directly changes the sampled channel law.

## PMD Channel

`PMDChannel` is a spectrum-averaged polarization-qubit surrogate for polarization
mode dispersion. Each section of the fiber model:

1. draws a random `2 x 2` Haar unitary to represent a principal-state basis,
2. attenuates coherence in that basis by a factor
	 `exp(-0.5 * (sigma_omega * tau_section)^2)`,
3. rotates back and repeats for `num_sections` sections.

Here `tau_section` is the section DGD implied by `dgd_mean_ps`, and `sigma_omega`
is determined from `signal_bandwidth_ghz`.

Interpretation:

- larger DGD or larger signal bandwidth reduces visibility more strongly,
- multiple sections create random polarization-basis scrambling rather than a single deterministic rotation,
- this reproduces bandwidth-dependent coherence loss in a reduced qubit setting.

Assumptions and limits:

- this is not a full frequency-dependent Jones-matrix simulation,
- it does not model pulse broadening, chromatic dispersion, or symbol-to-symbol temporal memory,
- it is best interpreted as a polarization decoherence surrogate for end-to-end benchmarking.

## Composite Channel

`CompositeChannel` chains multiple channel instances in sequence. It is useful when
an experiment should combine benchmark impairments, for example a polarization or
optical loss stage followed by an additional quantum-noise stage.

Assumptions and limits:

- the interpretation depends entirely on the chosen ordering of subchannels,
- if a subchannel changes state dimension, downstream blocks must accept that output space,
- physical realism depends on whether the composed subchannels are valid at the same abstraction level.
