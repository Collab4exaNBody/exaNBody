"""Run the user-provided example.

Usage (after sourcing setup-env.sh):
    python pyexanbody_run_example.py example.msp
"""

import sys
import pyexanbody as xnb

list_examples = xnb.list_examples()
for ex in list_examples:
    print(ex)
if len(sys.argv) > 1:
    msp = xnb.find_example(sys.argv[1])
else:
    print("Please provide a .msp name")
    sys.exit(0)

print(f"Running: {msp}")
rc = xnb.run_file(msp, argv0=sys.argv[0])
sys.exit(rc)
