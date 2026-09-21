"""Small, auditable, extensible quantitative research platform."""
__version__ = "2.0.0"

# Optional NumExpr 2.14.1 in the local Anaconda environment intermittently
# returned all-zero/-one results for large DataFrame division. Use pandas'
# NumPy reference path consistently in research, UI, and notebooks. This is a
# process-wide pandas option, deliberately documented as a numerical policy.
import pandas as _pd
_pd.set_option("compute.use_numexpr", False)
