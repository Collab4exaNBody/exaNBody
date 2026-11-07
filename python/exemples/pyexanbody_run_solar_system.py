"""Run the solar-system gravitational N-body example.

Usage (after sourcing setup-env.sh):
    python pyexanbody_run_solar_system_msp.py [path/to/solar_system.msp]

If no path is given the script finds solar_system.msp from the installed
share/examples directory using xnb.find_example().
"""

import sys
import pyexanbody as xnb

if len(sys.argv) > 1:
    msp = sys.argv[1]
else:
    msp = xnb.find_example("solar_system.msp")

print(f"Running: {msp}")
rc = xnb.run_file(msp, argv0=sys.argv[0])
sys.exit(rc)
