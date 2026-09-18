"""Where the audience-arm scripts find their files.

Every script in this folder imports these constants instead of hard-coding paths.
    ROOT     the repository root
    DATA     ROOT/data           (transcripts, external survey files, derived corpus tables)
    HANDOFF  ROOT/handoff        (frozen pre-registration files, alias tables, the reference scale)
    HERE     this folder (step7_audience/; the scripts sit one level down in numbered sub-folders)
    INPUTS   HERE/inputs         (inputs of record and shared intermediates: kett.pkl, directive_final.csv, ...)
    OUTPUTS  HERE/outputs        (result tables and run logs)
"""

from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"
HANDOFF = ROOT / "handoff"
INPUTS = HERE / "inputs"
OUTPUTS = HERE / "outputs"
