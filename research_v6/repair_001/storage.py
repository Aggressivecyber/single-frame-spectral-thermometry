"""Process-local routing to a fresh E-drive run; old files remain immutable."""
from pathlib import Path
from .. import common as base

PRIOR_HEAD='6f6c4f19e08e83d1ac0cd49e0c85c1c67dc56b51'
ROOT=base.ROOT
CODE=ROOT/'research_v6/repair_001'
REPORTS=ROOT/'reports/research_v6/repair_001'
DATA=ROOT/'outputs/research_v6/repair_001'
OLD_REPORTS=ROOT/'reports/research_v6/run_001'
OLD_DATA=ROOT/'outputs/research_v6/run_001'
# Frozen helpers read these globals at call time. No file is modified by the
# rebinding; their caches/temporary files now stay in the new run exclusively.
base.REPORTS=REPORTS
base.DATA=DATA
base.configure_runtime()


def sources():return {base.relative(p):base.sha(p) for p in sorted(CODE.glob('*.py'))}
