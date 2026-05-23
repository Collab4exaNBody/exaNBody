"""Minimal smoke test — verifies that pyexanbody loads and that the exaNBody
operators are visible through the plugin system."""

import os
import sys
import pyexanbody as xnb

main_config = os.path.join(os.environ["ONIKA_CONFIG_PATH"], "main-config.msp")
ctx = xnb.init([sys.argv[0], main_config])
if ctx.error_code >= 0:
    sys.exit(ctx.error_code)

ops = xnb.available_operators()
for op in ops:
    print(op)
    
print(f"\npyexanbody: {len(ops)} operators loaded")

# Check that at least one exaNBody-specific operator is present.
xnb_ops = [o for o in ops if any(kw in o for kw in ("domain", "nbh_dist", "lattice", "gravitational"))]
if not xnb_ops:
    print("WARNING: no exaNBody operators found — check ONIKA_PLUGIN_PATH")
else:
    print(f"pyexanbody: exaNBody operators visible (e.g. {xnb_ops[:3]})")

xnb.run(ctx)
xnb.end(ctx)
print("pyexanbody: OK")
