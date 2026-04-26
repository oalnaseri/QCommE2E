# Architecture

## End-to-End Quantum Communication Simulator

### Block Diagram

```
[Source Bits] --> [TX: Modulator + Encoder] --> [Quantum Channel] --> [RX: Decoder + Demodulator] --> [Recovered Bits]
```

### Design Principles
- Kraus operator formalism for all quantum channels.
- Modular transceiver components swappable via configuration.
- Gradient-based E2E training supported for learned TX/RX.
