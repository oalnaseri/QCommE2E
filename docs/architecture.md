# Architecture

## End-to-End Quantum Communication Simulator

### Block Diagram

```
[Symbol Source] --> [TX: Modulator / State Preparation] --> [Quantum Channel] --> [RX: Measurement / Detection] --> [Recovered Symbols / Bits]
```

### Measurement Path Used By The Benchmark

The benchmark notebooks and the `Simulator` class use a matched transmit and
receive chain so that BER and SER reflect channel impairment rather than a
receiver-model mismatch.

1. A symbol source draws labels uniformly from `{0, 1, 2, 3}`.
2. `QPSKModulator` maps each label to a pure-state codeword and stores it as a
	 density matrix.
3. The active channel applies `BaseChannel.apply(rho)` independently to each
	 transmitted state.
4. `PrettyGoodMeasurementDetector` builds a POVM from the same reference-state
	 codebook used by the modulator.
5. The receiver makes a hard symbol decision by maximizing
	 `Tr(E_i rho_out)` over the detector POVM elements.
6. BER is computed from the full two-bit labels associated with the detected
	 symbols. The benchmark does not collapse QPSK decisions to a single bit.

This matched path is important for scientific interpretation. Earlier surrogate
setups can look noisy simply because the detector is incompatible with the
transmitted alphabet. The current baseline avoids that failure mode by using the
same codebook at both ends of the link.

### QPSK Symbol And Bit Mapping

The discrete-variable baseline currently uses the following qubit-like codebook:

- Symbol `0` -> `|0>` -> bits `[0, 0]`
- Symbol `1` -> `|1>` -> bits `[0, 1]`
- Symbol `2` -> `(|0> + |1>) / sqrt(2)` -> bits `[1, 0]`
- Symbol `3` -> `(|0> - |1>) / sqrt(2)` -> bits `[1, 1]`

The modulation alphabet is therefore a four-state ensemble of `2 x 2` density
matrices. This keeps the channel benchmark analytically simple while still
exercising state preparation, channel action, and nontrivial POVM detection.

### Erasures In The End-To-End Path

The erasure channel is the only benchmark channel that expands the Hilbert space.
It maps a `d`-dimensional input to a `(d + 1)`-dimensional output by appending an
orthogonal erasure flag state.

To keep the end-to-end benchmark coherent:

- The pretty-good measurement detector embeds its original POVM into the larger
	output space.
- An additional erasure POVM element is added for the orthogonal flag state.
- The detector returns symbol label `-1` when that erasure branch wins.
- `QPSKModulator.symbols_to_bits` maps invalid labels such as `-1` to `[-1, -1]`.

This means erasures are counted as hard failures in BER and SER rather than being
wrapped back into one of the valid QPSK labels.

### Metrics Interpretation

- BER is computed from the full flattened two-bit labels of the transmitted and
	detected symbols.
- SER is computed directly on the hard symbol labels.
- GMI is estimated from hard symbol decisions, so it should be interpreted as a
	coarse discrete benchmark rather than a soft-information achievable-rate bound.

### Design Principles

- Density-matrix channel interface: each benchmark channel exposes a single
	`apply(rho)` map acting on one density matrix at a time.
- Matched benchmark measurement path: the default QPSK benchmark uses a detector
	constructed from the transmit codebook.
- Explicit surrogate assumptions: some channels are literature-faithful reduced
	models, but they are still benchmark surrogates rather than full field solvers.
- Modular transceiver components: transmitters, receivers, detectors, and channels
	are swappable via configuration.
- Gradient-based E2E training: learned transmitter and receiver components can be
	optimized jointly when differentiable blocks are used.
