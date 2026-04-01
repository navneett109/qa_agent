"""Bridge package so qa_agent.agent.* resolves to modules under /app."""

import os
import sys

# Route qa_agent.agent submodule lookups to the project root (/app).
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

__path__ = [_project_root]
