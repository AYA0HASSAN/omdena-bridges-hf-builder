"""Static guard for the inference-image COPY bug (ticket [BRIDGES-XX], agent feedback #858):
`pipeline.py` imports `.obb_utils` and `.serving`, so a Dockerfile `inference-builder` stage that only
copies `pipeline.py` leaves the distroless container unable to import it at start-up. This walks
`pipeline.py`'s own local-module imports (relative, or `models.yolo26s_obb_bridges.*`) transitively via
`ast`, and asserts every file found is covered by a `COPY` source in the Dockerfile's
`inference-builder`/`inference` stages - plus the two `__init__.py` files the package needs to be
importable at all.

A COPY line alone isn't enough, though: BuildKit also consults `Dockerfile.dockerignore` (the file
next to this Dockerfile) when assembling the build context, and a COPY source excluded there fails the
build before the COPY even runs - exactly what happened to `models/__init__.py` here (only
`models/yolo26s_obb_bridges/`, `models/conftest.py` and `models/test_integration.py` were allow-listed).
So this also re-checks every required file against that ignore file.
"""

import ast
from pathlib import Path

PACK_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = PACK_DIR.parent.parent
DOCKERFILE = PACK_DIR / "Dockerfile"
DOCKERIGNORE = PACK_DIR / "Dockerfile.dockerignore"


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


def _dockerignore_allows(rel_path: str, dockerignore_text: str) -> bool:
    """Minimal `*` deny-all + `!path` / `!path/` re-include matcher - enough for this pack's simple
    allow-list style dockerignore, not a full gitignore-pattern implementation."""
    allowed = False
    for line in dockerignore_text.splitlines():
        pattern = line.strip()
        if not pattern or pattern.startswith("#"):
            continue
        if pattern == "*":
            allowed = False
            continue
        if not pattern.startswith("!"):
            continue
        target = pattern[1:].rstrip("/")
        if rel_path == target or rel_path.startswith(f"{target}/"):
            allowed = True
    return allowed


def test_pipeline_local_imports_are_all_copied_into_inference_image():
    required = _local_module_files(PACK_DIR / "pipeline.py", PACK_DIR)
    required |= {PACK_DIR / "__init__.py", REPO_ROOT / "models" / "__init__.py"}
    required_rel = sorted(str(f.relative_to(REPO_ROOT)).replace("\\", "/") for f in required)

    copied = _copy_sources_in_inference_stages(DOCKERFILE.read_text())
    missing_copy = sorted(rel for rel in required_rel if rel not in copied)
    assert not missing_copy, f"Dockerfile inference-builder stage doesn't COPY: {missing_copy}"

    dockerignore_text = DOCKERIGNORE.read_text()
    excluded = sorted(rel for rel in required_rel if not _dockerignore_allows(rel, dockerignore_text))
    assert not excluded, f"Dockerfile.dockerignore excludes files the inference stage COPYs: {excluded}"
