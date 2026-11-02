"""Static guard for the inference-image COPY bug (ticket [BRIDGES-XX], agent feedback #858):
`pipeline.py` imports `.obb_utils` and `.serving`, so a Dockerfile `inference-builder` stage that only
copies `pipeline.py` leaves the distroless container unable to import it at start-up. This walks
`pipeline.py`'s own local-module imports (relative, or `models.yolo26s_obb_bridges.*`) transitively via
`ast`, and asserts every file found is covered by a `COPY` source in the Dockerfile's
`inference-builder`/`inference` stages - plus the two `__init__.py` files the package needs to be
importable at all.
"""

import ast
from pathlib import Path

PACK_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = PACK_DIR.parent.parent
DOCKERFILE = PACK_DIR / "Dockerfile"


def _local_import_targets(tree: ast.Module, own_dir: Path) -> list[Path]:
    targets: list[Path] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.module is None:
            continue
        if node.level > 0:  # from .obb_utils import ...
            targets.append(own_dir / f"{node.module.replace('.', '/')}.py")
        elif node.module.startswith("models.yolo26s_obb_bridges"):
            rel = node.module.removeprefix("models.yolo26s_obb_bridges").lstrip(".")
            if rel:
                targets.append(own_dir / f"{rel.replace('.', '/')}.py")
    return targets


def _local_module_files(entry: Path, pack_dir: Path, seen: set[Path] | None = None) -> set[Path]:
    seen = set() if seen is None else seen
    if entry in seen or not entry.exists():
        return seen
    seen.add(entry)
    tree = ast.parse(entry.read_text())
    for target in _local_import_targets(tree, pack_dir):
        _local_module_files(target, pack_dir, seen)
    return seen


def _copy_sources_in_inference_stages(dockerfile_text: str) -> set[str]:
    """Source paths of every local (non `--from=`) `COPY` between a `FROM ... AS inference-builder` or
    `FROM ... AS inference` line and the next `FROM`."""
    sources: set[str] = set()
    in_scope = False
    for line in dockerfile_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("FROM"):
            in_scope = stripped.rstrip().endswith(("AS inference-builder", "AS inference"))
            continue
        if in_scope and stripped.startswith("COPY") and "--from=" not in stripped:
            sources.add(stripped.split()[1].replace("\\", "/"))
    return sources


def test_pipeline_local_imports_are_all_copied_into_inference_image():
    required = _local_module_files(PACK_DIR / "pipeline.py", PACK_DIR)
    required |= {PACK_DIR / "__init__.py", REPO_ROOT / "models" / "__init__.py"}

    copied = _copy_sources_in_inference_stages(DOCKERFILE.read_text())

    missing = sorted(
        str(f.relative_to(REPO_ROOT)).replace("\\", "/")
        for f in required
        if str(f.relative_to(REPO_ROOT)).replace("\\", "/") not in copied
    )
    assert not missing, f"Dockerfile inference-builder stage doesn't COPY: {missing}"
