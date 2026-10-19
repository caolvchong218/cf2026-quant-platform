ARCHIVED PREPARATION IMPLEMENTATION — NOT THE CURRENT FORMAL ENTRY POINT

This directory preserves the 2026-10-16 preparation implementation and its
original notes below. The supplied eight-page Project 2 assignment has since
been received. The statement below about Blackboard visibility describes only
the earlier check; it is not the current assignment status.

Current formal study, run from the repository root:
  python scripts/run_project2_rsrs.py --help
  python scripts/run_project2_rsrs.py --data data/private/project2_rsrs

Current protocol: configs/project2_rsrs.json
Current public evidence: evidence/project2_rsrs/
Current materials: reports/project2_rsrs/
Current instructions: docs/PROJECT2_REPRODUCIBILITY.md
Course requirements: docs/PROJECT2_REQUIREMENTS_MATRIX.md

The formal matrix has 198 main runs (3 indices x 3 periods x 2 fee cases x
11 variants), including a 15% volatility target and fixed 75% sizing control.
This older module remains an independent oracle for 162 baseline comparisons.
Its old headline numbers, 162-run matrix and report builder are historical;
do not substitute them for the formal evidence or change them to hide history.
CogAlpha is a separate paper study with its own data, evidence and materials.
Neither package alone proves a Blackboard submission or a particular grade.

--- ORIGINAL PREPARATION README FOLLOWS ---

RSRS brokerage-report reproduction (research preparation for CF2026 Project 2)

Status: The authenticated Blackboard Project folder and grade columns checked
on 2026-10-16 contained Project 1 but no visible Project 2 assignment.
The user confirmed proceeding with research preparation before publication.
No Blackboard submission has been made. Do not claim rubric compliance.

Report source:
Everbright Securities, Liu Junwei / Zhou Xiaoxiao, 2017-05-01.
Resistance Support Relative Strength (RSRS) market timing.
Public report location: https://github.com/hugo2046/QuantsPlaybook
Pinned reference commit: 3f83a77f4c7996ed8185b8979a42c4372c1d811f
Directory: C-择时类/RSRS择时指标

Method:
- Standard score N=18 M=600; right-skew score N=16 M=300; +/-0.7 thresholds.
- Rolling OLS with intercept. Standardization uses previous M slopes, ddof=0.
- Complete warmup, no zero filling. MA20 entry filter only, no forced MA exits.
- At t close calculate decision; t+1 open execute; mark at close.
- Long/cash, no leverage, zero interest, fractional research units.
- Same fees and dates for benchmark. Price index is not itself tradable.
- Fixed main candidate right_ma; show all results, no ex-post winner replacement.
- Seven report variants plus buy/hold and MA20 controls.
- Three indices, three periods, two fee cases: 162 principal runs.
- Parameter sensitivity, implementation ambiguity checks, block bootstrap,
  cross-source price checks and an ETF proxy experiment are supplementary.

Reproduce (from this directory; Python 3.11+):
python -m pip install -r requirements.txt
python -m pytest test_rsrs.py -q
python fetch_data.py --output DATA_DIRECTORY
python run_study.py --data DATA_DIRECTORY --output OUTPUT_DIRECTORY

The delivered desktop bundle includes a fixed offline snapshot. Prefer it over
refetching, since providers can revise historical quotes. The main manifest
verifies data and configuration SHA-256. The --supplemental directory optionally
contains existing Tushare benchmark.csv and 510330 ETF etf.csv; no token needed.
build_report.py creates the dated report source and browser page using this
frozen study's results; its editorial narrative is specific to v0.1, not a
generic automatic report for changed experiments. Team names come from a local
--team JSON file and are not embedded in the public source.

Headline result (CSI 300, 2017-05-02..2026-10-15, 10 bp each side):
right_ma cumulative +14.5489%, CAGR 1.5020%, Sharpe 0.1787, MDD 35.0102%.
buy_hold cumulative +21.4103%; standard_ma cumulative +23.9476%.
These are retrospective historical results, not untouched prospective tests.
The main candidate did NOT beat CSI 300. Do not replace the platform incumbent.

Dependencies are reused numerical, scientific and plotting libraries. Formulas
and ledger code are independently implemented. The upstream notebook was read
for comparison, not copied or redistributed as our code. Data endpoint contract
is documented by AKShare: https://akshare.akfamily.xyz/data/index/index.html
