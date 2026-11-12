from __future__ import annotations

from pathlib import Path

import pytest

from tests.generate_synthetic_clips import ClipSpec, _truth, make_duplicate, render


@pytest.fixture(scope="session")
def clips(tmp_path_factory) -> dict[str, Path]:
    """Small synthetic clips rendered once per test session."""
    out = tmp_path_factory.mktemp("clips")
    t = _truth(1, 5, 4, 5, 5, 0, 5)
    specs = {
        "ideal10": ClipSpec("ideal10", 10, t, "usable", seed=1),
        "other10": ClipSpec("other10", 10, t, "usable", seed=2),
        "blur10": ClipSpec("blur10", 10, t, "reject", seed=4, jitter=28, blur=31, bob=18),
        "dark10": ClipSpec("dark10", 10, t, "reject", seed=5, gain=0.12),
        "tripod10": ClipSpec(
            "tripod10",
            10,
            t,
            "reject",
            seed=6,
            hands=0,
            bob=0,
            pan_speed=0,
            static_third_person=True,
        ),
        "long60": ClipSpec("long60", 60, t, "usable", seed=3),
        "long75": ClipSpec("long75", 75, t, "usable", seed=8),
    }
    paths = {}
    for key, spec in specs.items():
        p = out / f"{key}.mp4"
        render(spec, p)
        paths[key] = p
    dup = out / "ideal10_dup.mp4"
    make_duplicate(paths["ideal10"], dup)
    paths["ideal10_dup"] = dup
    return paths
