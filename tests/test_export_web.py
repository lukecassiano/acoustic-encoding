"""export_web's per-subject cleanup must leave the perception readout (written by
src.perception.export into the same folder) in place. tmp dirs only."""

from src import export_web


def test_cleanup_preserves_readout_files(tmp_path):
    sd = tmp_path / "sub-001"
    sd.mkdir()
    own = ["meta.json", "w_region.f32", "w_region.f32.gz.b64.txt", "check.json", "stale_old_file.bin"]
    readout = ["readout.json", "readout_boot.f32", "readout_boot.f32.gz.b64.txt"]
    for n in own + readout:
        (sd / n).write_bytes(n.encode())
    kept = export_web.clear_subject_dir(sd)
    assert kept == sorted(readout)
    assert sorted(p.name for p in sd.iterdir()) == sorted(readout)
    assert all((sd / n).read_bytes() == n.encode() for n in readout)
    assert export_web.PRESERVE == frozenset(readout)


def test_cleanup_without_readout(tmp_path):
    sd = tmp_path / "sub-002"
    sd.mkdir()
    (sd / "meta.json").write_text("{}")
    assert export_web.clear_subject_dir(sd) == []
    assert list(sd.iterdir()) == []
