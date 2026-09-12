"""Phase 0 check: the config loads, derives the right context, and refuses
the three mistakes that silently corrupt an experiment.

Run: python tests/test_config.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chromgraph.config import Config, load_config  # noqa: E402


def _expect(exc_substring, fn):
    try:
        fn()
    except ValueError as e:
        assert exc_substring in str(e), f"wrong error: {e}"
        return
    raise AssertionError(f"expected ValueError containing {exc_substring!r}, got none")


def test_loads():
    cfg = load_config()
    assert isinstance(cfg, Config)
    # 128 nodes x 5 kb = 640 kb of context per sample.
    assert cfg.data.context_bp == 640_000, cfg.data.context_bp
    assert cfg.train.arm == "full"


def test_splits_are_disjoint():
    d = load_config().data
    assert not (set(d.chroms_train) & set(d.chroms_val))
    assert not (set(d.chroms_train) & set(d.chroms_test))
    assert not (set(d.chroms_val) & set(d.chroms_test))


def test_rejects_chromosome_leak():
    _expect("chromosome leak", lambda: load_config(**{"data.chroms_val": ["chr1", "chr10"]}))


def test_rejects_unknown_arm():
    _expect("train.arm must be one of", lambda: load_config(**{"train.arm": "b9_typo"}))


def test_rejects_overlapping_edge_classes():
    # min_separation <= local_radius makes "distal" edges indistinguishable
    # from local ones, which quietly turns the B2 control into the full model.
    _expect("must exceed local_radius", lambda: load_config(**{"data.min_separation": 3}))


def test_rejects_typo_key():
    _expect("unknown key", lambda: load_config(**{"model.d_modell": 256}))


def test_override_roundtrip(tmp=Path("results/_config_roundtrip.yaml")):
    cfg = load_config(**{"train.arm": "b3_shuffled_hic", "seed": 7})
    assert cfg.train.arm == "b3_shuffled_hic" and cfg.seed == 7
    cfg.save(tmp)
    assert load_config(tmp).train.arm == "b3_shuffled_hic"
    tmp.unlink()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("\nPhase 0 config check passed.")
