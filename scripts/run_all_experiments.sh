#!/bin/bash
set -e
python experiments/fiber_pmd_qae/run.py
python experiments/fso_turbulence_e2e/run.py
python experiments/quantum_baseline_comparison/run.py
