"""Import-isolation guard for the serving path (ticket [BRIDGES-XX], agent feedback #858).

`models.yolo26s_obb_bridges.serving` is the module the distroless inference container actually runs
(the STAC item's `mlm:entrypoint` and the knative MODEL_MODULE both point at it): it must import with
ultralytics, torch and zenml completely unavailable, since none of those are installed in that image.

`models.yolo26s_obb_bridges.pipeline` is different: it needs `zenml` importable at module load for its
`@step`/`@pipeline` decorators (fair/utils/model_validator.py's AST check requires those decorators to
live in that exact file), so it can never be zenml-free - that's documented, not a bug. It still must
import fine with ultralytics/torch blocked, since those stay inside function bodies.

Every subprocess here sets `FAIR_SKIP_ZENML_PATCH=1` (same as the inference image's Dockerfile ENV):
without it, fair-py-ops's own `fair_zenml_patch.pth` imports zenml at interpreter startup regardless of
what any script imports (see the Dockerfile's `inference` stage comment), which would make the zenml
checks below fail for a reason that has nothing to do with `serving.py` itself.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

_SCRIPT = """
import sys, importlib.abc

class _Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ImportError(f"blocked: {{name}}")
        return None

BLOCKED = {blocked!r}
sys.meta_path.insert(0, _Blocker())
import {module}
print("import ok")
"""


def _run_blocked(module: str, blocked: set[str]) -> subprocess.CompletedProcess:
    script = _SCRIPT.format(blocked=blocked, module=module)
    env = {**os.environ, "FAIR_SKIP_ZENML_PATCH": "1"}
    return subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, cwd=str(REPO_ROOT), env=env
    )


@pytest.mark.parametrize("module", ["models.yolo26s_obb_bridges.serving", "models.yolo26s_obb_bridges.pipeline"])
def test_module_imports_without_ultralytics_or_torch(module):
    result = _run_blocked(module, {"ultralytics", "torch"})
    assert result.returncode == 0, result.stderr
    assert "import ok" in result.stdout


def test_serving_module_imports_without_zenml():
    result = _run_blocked("models.yolo26s_obb_bridges.serving", {"zenml"})
    assert result.returncode == 0, result.stderr
    assert "import ok" in result.stdout


@pytest.mark.xfail(
    reason=(
        "pipeline.py needs `from zenml import log_metadata, pipeline, step` at module level for its "
        "decorators; fair/utils/model_validator.py requires those decorators to live in this exact "
        "file, so it can't be made zenml-free without breaking the model-pack contract. Known gap, "
        "see README.md 'Known limitations' and the hotosm/fAIr-models follow-up issue."
    ),
    strict=True,
)
def test_pipeline_module_imports_without_zenml():
    result = _run_blocked("models.yolo26s_obb_bridges.pipeline", {"zenml"})
    assert result.returncode == 0, result.stderr
