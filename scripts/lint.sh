#!/bin/bash
pylint src/quantum_comm_sim --fail-under=7.0
black src/ tests/ experiments/ --check
