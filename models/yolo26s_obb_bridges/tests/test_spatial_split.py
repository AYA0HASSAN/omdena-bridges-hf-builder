"""Spatial-split correctness tests (ticket [BRIDGES-XX], agent feedback #858 - mixed-zoom split
leakage): `_spatial_split_names` grouped OAM-x-y-z tiles by their raw, un-scaled `(x // block_size,
y // block_size)`, ignoring zoom entirely. A parent tile and its own child at a finer zoom can have
different raw x/y blocks, so they could land in different splits - leaking val into train.

Pure-Python, no new dependencies: tiles are referenced purely by name (no chip files on disk), so these
run in the lightweight `test` CI job (plain pytest + fair-py-ops, no rasterio/geopandas/torch) alongside
test_dockerfile_copies_package.py and test_serving_imports.py - see pipeline-ci.yml.
"""

import random
import re

import pytest

from models.yolo26s_obb_bridges.pipeline import (
    DEFAULT_BLOCK_SIZE,
    _assert_no_cross_split_overlap,
    _parse_oam_xyz,
    _spatial_split_names,
)


def _old_spatial_split_names(
    chip_names: list[str], val_ratio: float, seed: int, block_size: int
) -> tuple[list[str], list[str]]:
    """Byte-for-byte copy of the pre-fix `_spatial_split_names`: groups by the raw, un-scaled
    `(x // block_size, y // block_size)`, ignoring zoom. Kept only as the reference implementation for
    test 1 (must fail against it) and test 6 (the new code must reproduce it exactly for single-zoom
    datasets)."""

    def chip_tile_xyz(name: str) -> tuple[int, int, int] | None:
        m = re.match(r"OAM-(\d+)-(\d+)-(\d+)", name)
        return tuple(int(v) for v in m.groups()) if m else None  # type: ignore[return-value]

    blocks: dict[tuple[int, int], list[str]] = {}
    unmatched: list[str] = []
    for name in chip_names:
        xyz = chip_tile_xyz(name)
        if xyz is None:
            unmatched.append(name)
            continue
        x, y, _z = xyz
        blocks.setdefault((x // block_size, y // block_size), []).append(name)

    block_keys = sorted(blocks)
    random.Random(seed).shuffle(block_keys)
    val_block_count = max(1, round(len(block_keys) * val_ratio)) if block_keys and val_ratio > 0 else 0
    val_blocks = set(block_keys[:val_block_count])

    train_names = [n for key, names in blocks.items() if key not in val_blocks for n in names]
    val_names = [n for key, names in blocks.items() if key in val_blocks for n in names]

    shuffled_unmatched = sorted(unmatched)
    random.Random(seed).shuffle(shuffled_unmatched)
    val_unmatched_count = round(len(shuffled_unmatched) * val_ratio)
    val_names += shuffled_unmatched[:val_unmatched_count]
    train_names += shuffled_unmatched[val_unmatched_count:]
    return sorted(train_names), sorted(val_names)


def _split_of(train_names: list[str], val_names: list[str]) -> dict[str, str]:
    return {n: "train" for n in train_names} | {n: "val" for n in val_names}


def _tiles_overlap(a: tuple[int, int, int], b: tuple[int, int, int]) -> bool:
    """Ground-truth ancestor/descendant check, independent of any REF_Z choice: shift the finer tile's
    x/y down to the coarser tile's zoom and see if they land on the same tile."""
    (xa, ya, za), (xb, yb, zb) = a, b
    if za >= zb:
        d = za - zb
        return (xa >> d, ya >> d) == (xb, yb)
    d = zb - za
    return (xa, ya) == (xb >> d, yb >> d)


def _make_descendant(
    root_x: int, root_y: int, root_z: int, target_z: int, rng: random.Random
) -> tuple[int, int, int]:
    d = target_z - root_z
    if d == 0:
        return root_x, root_y, root_z
    return (root_x << d) | rng.getrandbits(d), (root_y << d) | rng.getrandbits(d), target_z


# --- 1. repro: must fail on the old code, pass on the new, for every seed.
def test_parent_and_child_tile_share_split_across_seeds():
    # Raw old blocks (block_size=16): parent 99 // 16 == 6; child 199 // 16 == 12 - a different raw
    # block, even though the child is geographically inside the parent.
    parent = "OAM-99-50-17.tif"
    child = "OAM-199-101-18.tif"
    assert 99 // 16 != 199 // 16  # sanity: this pair really does cross an old raw block boundary
    names = [parent, child, "OAM-5-5-18.tif", "OAM-500-500-18.tif", "OAM-6-6-18.tif"]

    old_breaks = False
    for seed in range(100):
        train, val = _old_spatial_split_names(names, val_ratio=0.3, seed=seed, block_size=16)
        if _split_of(train, val)[parent] != _split_of(train, val)[child]:
            old_breaks = True
            break
    assert old_breaks, "expected the pre-fix grouping to split this parent/child pair for some seed"

    for seed in range(100):
        train, val, _ref_zoom = _spatial_split_names(names, val_ratio=0.3, seed=seed, block_size=16)
        split = _split_of(train, val)
        assert split[parent] == split[child], f"seed={seed}: parent and child landed in different splits"


# --- 2. a coarse tile and several z18 children in different z14 blocks it covers all share its split.
def test_coarse_tile_and_its_children_share_split():
    coarse = "OAM-10-20-12.tif"  # at REF_Z=14 (default, mixed zooms below), covers x in [40,43], y in [80,83]
    children = [
        "OAM-643-1283-18.tif",  # REF14 block (40, 80)
        "OAM-643-1340-18.tif",  # REF14 block (40, 83)
        "OAM-700-1283-18.tif",  # REF14 block (43, 80)
        "OAM-700-1340-18.tif",  # REF14 block (43, 83)
    ]
    names = [coarse, *children, "OAM-900-900-18.tif", "OAM-901-901-18.tif"]

    for seed in range(100):
        train, val, ref_zoom = _spatial_split_names(names, val_ratio=0.3, seed=seed, block_size=DEFAULT_BLOCK_SIZE)
        assert ref_zoom == 14
        split = _split_of(train, val)
        assert len({split[coarse], *(split[c] for c in children)}) == 1, (
            f"seed={seed}: coarse tile and its children didn't all share one split"
        )


# --- 3. deep zooms: z19/z20 descendants of a z18 tile follow it.
def test_deep_zoom_descendants_follow_their_ancestor():
    tile_z18 = "OAM-50-50-18.tif"  # REF14 block (3, 3)
    child_z19 = "OAM-100-100-19.tif"  # genuine z19 child of tile_z18, also REF14 block (3, 3)
    grandchild_z20 = "OAM-200-200-20.tif"  # genuine z20 grandchild, also REF14 block (3, 3)
    names = [tile_z18, child_z19, grandchild_z20, "OAM-900-900-18.tif", "OAM-901-901-18.tif"]

    for seed in range(100):
        train, val, _ref_zoom = _spatial_split_names(names, val_ratio=0.3, seed=seed, block_size=DEFAULT_BLOCK_SIZE)
        split = _split_of(train, val)
        assert split[tile_z18] == split[child_z19] == split[grandchild_z20], (
            f"seed={seed}: z18/z19/z20 ancestor chain didn't all share one split"
        )


# --- 4. property test: 300 random mixed-zoom tile sets; every genuinely overlapping pair shares a
#        split, and the step-6 guard never fires.
def test_mixed_zoom_overlaps_always_share_split():
    rng_master = random.Random(20261102)
    for trial in range(300):
        rng = random.Random(rng_master.randint(0, 2**32 - 1))
        tiles: dict[str, tuple[int, int, int]] = {}

        for _ in range(rng.randint(2, 4)):
            root_z = rng.randint(12, 16)
            root_x = rng.randint(0, 2**root_z - 1)
            root_y = rng.randint(0, 2**root_z - 1)
            tiles[f"OAM-{root_x}-{root_y}-{root_z}.tif"] = (root_x, root_y, root_z)
            for _ in range(rng.randint(1, 3)):
                target_z = rng.randint(root_z, 20)
                dx, dy, dz = _make_descendant(root_x, root_y, root_z, target_z, rng)
                tiles[f"OAM-{dx}-{dy}-{dz}.tif"] = (dx, dy, dz)

        for _ in range(rng.randint(1, 3)):
            z = rng.randint(14, 20)
            x = rng.randint(0, 2**z - 1)
            y = rng.randint(0, 2**z - 1)
            tiles[f"OAM-{x}-{y}-{z}.tif"] = (x, y, z)

        names = list(tiles)
        train, val, resolved_ref_zoom = _spatial_split_names(
            names, val_ratio=0.25, seed=trial, block_size=DEFAULT_BLOCK_SIZE
        )
        split = _split_of(train, val)

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a_name, b_name = names[i], names[j]
                if _tiles_overlap(tiles[a_name], tiles[b_name]):
                    assert split[a_name] == split[b_name], (
                        f"trial {trial}: overlapping tiles {a_name} and {b_name} landed in different splits"
                    )

        _assert_no_cross_split_overlap(names, split, resolved_ref_zoom)  # must not raise


# --- 5. not degenerate: a large, spread-out single-zoom dataset still splits sensibly.
def test_large_non_overlapping_dataset_splits_both_non_empty_near_ratio():
    names = [f"OAM-{(i % 50) * 5000}-{(i // 50) * 5000}-18.tif" for i in range(2000)]
    assert len({_parse_oam_xyz(n) for n in names}) == 2000  # sanity: all distinct tiles

    for seed in range(5):
        train, val, _ref_zoom = _spatial_split_names(names, val_ratio=0.2, seed=seed, block_size=DEFAULT_BLOCK_SIZE)
        assert train and val
        val_share = len(val) / len(names)
        assert abs(val_share - 0.2) <= 0.05, f"seed={seed}: val share {val_share:.3f} not within +/-5pts of 0.2"


# --- 6. regression: for z18-only names with block_size=16, the new assignment equals the old one exactly.
def test_single_zoom_regression_matches_old_grouping_exactly():
    rng = random.Random(7)
    names = [f"OAM-{rng.randint(0, 4000)}-{rng.randint(0, 4000)}-18.tif" for _ in range(300)]
    names = list(dict.fromkeys(names))  # drop accidental duplicate tile names

    for seed in range(20):
        old_train, old_val = _old_spatial_split_names(names, val_ratio=0.25, seed=seed, block_size=16)
        new_train, new_val, ref_zoom = _spatial_split_names(names, val_ratio=0.25, seed=seed, block_size=16)
        assert ref_zoom == 14
        assert new_train == old_train
        assert new_val == old_val


# --- 7. determinism: same seed gives the same result; input order doesn't matter.
def test_deterministic_and_order_independent():
    rng = random.Random(99)
    names = []
    for _ in range(50):
        z = rng.randint(12, 20)
        names.append(f"OAM-{rng.randint(0, 2**z - 1)}-{rng.randint(0, 2**z - 1)}-{z}.tif")
    names = list(dict.fromkeys(names))

    result_a = _spatial_split_names(names, val_ratio=0.3, seed=11, block_size=DEFAULT_BLOCK_SIZE)
    result_b = _spatial_split_names(names, val_ratio=0.3, seed=11, block_size=DEFAULT_BLOCK_SIZE)
    assert result_a == result_b

    shuffled = list(names)
    random.Random(1).shuffle(shuffled)
    result_c = _spatial_split_names(shuffled, val_ratio=0.3, seed=11, block_size=DEFAULT_BLOCK_SIZE)
    assert result_a == result_c


# --- 8. invalid names raise ValueError.
@pytest.mark.parametrize(
    "name",
    [
        "OAM-999999-0-10.tif",  # x >= 2**z (2**10 == 1024)
        "OAM--5-10-18.tif",  # negative x
        "OAM-abc-10-18.tif",  # non-numeric x
        "OAM-10-20.tif",  # missing z
    ],
)
def test_invalid_oam_names_raise_value_error(name):
    with pytest.raises(ValueError):
        _parse_oam_xyz(name)
    with pytest.raises(ValueError):
        _spatial_split_names([name], val_ratio=0.2, seed=0, block_size=DEFAULT_BLOCK_SIZE)
