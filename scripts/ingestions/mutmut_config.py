"""mutmut configuration for the ingestion parsers.

Phase 4 of the robustness programme. Mutmut applies AST-level mutations
(swap operators, negate constants, modify return values, etc.) to the
listed files and asserts that the test suite catches each. Surviving
mutants reveal where the test assertions are too loose — same lesson
the eegdash-viewer learned during its Stryker iterations.

To run::

    cd scripts/ingestions
    mutmut run

To see surviving mutants::

    mutmut results
    mutmut show <id>

To stress one specific file (recommended starting point — establish
floor before expanding scope, per  § Phase 4)::

    mutmut run --paths-to-mutate _vhdr_parser.py

Target: ≥ 60% kill ratio on _vhdr_parser.py before expanding scope.
"""

# Mutmut 3.x reads its configuration from setup.cfg or pyproject.toml,
# but ALSO honours a mutmut_config.py module via init_mutmut_config().
# We keep this file as documentation; the actual config lives in
# pyproject.toml [tool.mutmut].
