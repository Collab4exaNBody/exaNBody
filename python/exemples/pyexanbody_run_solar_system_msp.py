"""Run the solar-system gravitational N-body example.

Usage (after sourcing setup-env.sh):
    python run_solar_system.py [path/to/solar_system.msp]

If no path is given the script looks for solar_system.msp next to itself,
which is where the CMake install puts it.
"""

import os
import sys
import pyexanbody as xnb

default_msp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../contribs/microCosmos/samples/simple_example/solar_system.msp")
msp = sys.argv[1] if len(sys.argv) > 1 else default_msp

if not os.path.exists(msp):
    sys.exit(f"error: .msp file not found: {msp}")

print(f"Running: {msp}")
rc = xnb.run_file(msp, argv0=sys.argv[0])
sys.exit(rc)
