#!/usr/bin/env python3
"""
Quick test script - runs minimal experiment configuration.
Useful for verifying installation and testing changes.

Usage:
    python scripts/run_quick.py
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from run_experiments import main as run_main

if __name__ == "__main__":
    # Override sys.argv to run quick mode
    sys.argv = ['run_experiments.py', '--quick']
    run_main()
