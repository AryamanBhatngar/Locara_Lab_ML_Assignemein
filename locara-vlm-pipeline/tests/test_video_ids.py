"""A video_id comes from the content, or from the capture app when it supplies one."""

import hashlib
import json
import shutil
import uuid
from pathlib import Path

import pytest

import score_videos
from src.config import Settings
from src.schemas import scoring_schema
from src.scorer import quality_scorer, s3_io
from src.scorer.dedup import HashRegistry
from src.scorer.quality_scorer import QualityScorer, VideoInput, resolve_inputs

SUPPLIED_ID = "3f2b8c1e-7a4d-4e9b-8c21-0d5f6a7b8c9d"
MB = 1024 * 1024


def place(clips, key, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(clips[key], path)
    return path


def id_of(path):
    return resolve_inputs([str(path)], Settings())[0].video_id


def test_same_name_with_different_content_gets_different_ids(tmp_path, clips):
    first = place(clips, "ideal10", tmp_path / "run1" / "clip.mp4")
    second = place(clips, "other10", tmp_path / "run2" / "clip.mp4")
    assert id_of(first) != id_of(second)


def test_same_content_under_a_different_name_gets_the_same_id(tmp_path, clips):
    first = place(clips, "ideal10", tmp_path / "a" / "morning.mp4")
    second = place(clips, "ideal10", tmp_path / "b" / "evening.mp4")
    assert id_of(first) == id_of(second)


def test_id_is_the_sha256_of_the_bytes_in_uuid4_shape(tmp_path):
    data = b"x" * (3 * MB + 17)
    path = tmp_path / "clip.mp4"
    path.write_bytes(data)
    expected = bytearray(hashlib.sha256(data).digest()[:16])
    expected[6] = (expected[6] & 0x0F) | 0x40
    expected[8] = (expected[8] & 0x3F) | 0x80

    video_id = scoring_schema.video_id_for_file(path)
    assert video_id == str(uuid.UUID(bytes=bytes(expected)))
    assert uuid.UUID(video_id).version == 4


def test_file_is_hashed_in_1_mb_chunks(monkeypatch, tmp_path):
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"x" * (2 * MB + 1))
    sizes = []
    real_open = Path.open

    class Recording:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.handle.close()

        def read(self, size=-1):
            sizes.append(size)
            return self.handle.read(size)

    monkeypatch.setattr(Path, "open", lambda self, *a, **k: Recording(real_open(self, *a, **k)))
    scoring_schema.video_id_for_file(path)
    assert sizes and set(sizes) == {MB}


class FakeS3:
    def __init__(self, etag="abc123", body=b"video bytes"):
        self.etag, self.body, self.downloads = etag, body, []

    def head_object(self, Bucket, Key):
        return {"ContentLength": len(self.body), "ETag": f'"{self.etag}"'}

    def download_file(self, bucket, key, path):
        self.downloads.append(key)
        Path(path).write_bytes(self.body)


@pytest.fixture
def s3(monkeypatch, tmp_path):
    fake = FakeS3()
    monkeypatch.setattr(s3_io, "_client", lambda settings: fake)
    fake.settings = Settings(cache_dir=tmp_path / "cache")
    return fake


def test_s3_id_comes_from_the_object_details_without_downloading(s3):
    def id_for(uri="s3://bucket/a.mp4", etag="abc", size=10):
        return s3_io.object_id(uri, s3_io.ObjectInfo(etag, size))

    assert id_for() == id_for()
    assert len({id_for(), id_for(etag="def"), id_for(size=11), id_for("s3://bucket/b.mp4")}) == 4
    assert id_for() != id_for("s3://other/a.mp4")
    assert uuid.UUID(id_for()).version == 4

    info = s3_io.head("s3://bucket/a.mp4", s3.settings)
    assert (info.etag, info.size) == ("abc123", len(s3.body))
    assert s3.downloads == []


def test_s3_input_uses_the_object_id(s3):
    uri = "s3://bucket/a.mp4"
    expected = s3_io.object_id(uri, s3_io.head(uri, s3.settings))
    assert resolve_inputs([uri], s3.settings)[0].video_id == expected


def test_replaced_s3_object_gets_a_new_id_and_new_bytes(s3):
    uri = "s3://bucket/a.mp4"
    before = resolve_inputs([uri], s3.settings)[0]
    s3.etag, s3.body = "def456", b"other bytes"
    after = resolve_inputs([uri], s3.settings)[0]
    assert after.video_id != before.video_id
    assert s3_io.download(uri, s3.settings, after.info).read_bytes() == b"other bytes"


def test_unchanged_s3_object_comes_from_the_cache(s3):
    uri = "s3://bucket/a.mp4"
    s3_io.download(uri, s3.settings)
    s3_io.download(uri, s3.settings)
    assert s3.downloads == ["a.mp4"]


def write_list(path, *rows):
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return f"@{path}"


def test_video_id_column_in_a_list_file_wins(tmp_path, clips):
    listing = write_list(tmp_path / "v.txt", "path,video_id", f"{clips['ideal10']},{SUPPLIED_ID}")
    assert resolve_inputs([listing], Settings())[0].video_id == SUPPLIED_ID


def test_blank_video_id_cell_falls_back_to_the_content(tmp_path, clips):
    listing = write_list(tmp_path / "v.txt", "path,video_id", f"{clips['ideal10']},")
    assert resolve_inputs([listing], Settings())[0].video_id == id_of(clips["ideal10"])


def test_supplied_id_is_used_for_s3_too(tmp_path, s3):
    listing = write_list(tmp_path / "v.txt", "path,video_id", f"s3://bucket/a.mp4,{SUPPLIED_ID}")
    assert resolve_inputs([listing], s3.settings)[0].video_id == SUPPLIED_ID


def test_supplied_id_reaches_the_output(monkeypatch, tmp_path, clips):
    monkeypatch.chdir(tmp_path)
    listing = write_list(tmp_path / "v.txt", "path,video_id", f"{clips['ideal10']},{SUPPLIED_ID}")
    assert score_videos.main([listing, "--provider", "offline", "-o", "out.json"]) == 0
    assert json.loads(Path("out.json").read_text(encoding="utf-8"))["video_id"] == SUPPLIED_ID


@pytest.mark.parametrize("bad", ["not-a-uuid", "3f2b8c1e-7a4d-1e9b-8c21-0d5f6a7b8c9d"])
def test_supplied_id_must_be_a_uuid4(tmp_path, clips, bad):
    listing = write_list(tmp_path / "v.txt", "path,video_id", f"{clips['ideal10']},{bad}")
    with pytest.raises(ValueError, match=r"v\.txt"):
        resolve_inputs([listing], Settings())


def test_supplied_id_needs_a_single_video(tmp_path, clips):
    listing = write_list(
        tmp_path / "v.txt", "path,video_id", f"{clips['ideal10'].parent},{SUPPLIED_ID}"
    )
    with pytest.raises(ValueError, match="single video"):
        resolve_inputs([listing], Settings())


def test_list_header_needs_a_path_column(tmp_path, clips):
    listing = write_list(tmp_path / "v.txt", "file,video_id", f"{clips['ideal10']},{SUPPLIED_ID}")
    with pytest.raises(ValueError, match="path column"):
        resolve_inputs([listing], Settings())


def test_plain_list_keeps_commas_in_paths(tmp_path, clips):
    clip = place(clips, "ideal10", tmp_path / "take 1, final.mp4")
    listing = write_list(tmp_path / "v.txt", str(clip))
    assert resolve_inputs([listing], Settings())[0].local_path == str(clip)


def test_unreadable_file_fails_that_video_only(monkeypatch, tmp_path, clips):
    bad = place(clips, "ideal10", tmp_path / "bad.mp4")
    good = place(clips, "other10", tmp_path / "good.mp4")
    real = scoring_schema.video_id_for_file

    def refuse(path):
        if Path(path).name == "bad.mp4":
            raise PermissionError("access denied")
        return real(path)

    monkeypatch.setattr(quality_scorer, "video_id_for_file", refuse, raising=False)
    first, second = resolve_inputs([str(bad), str(good)], Settings())
    assert "access denied" in first.error
    assert second.error is None


def scored_run(registry, *inputs):
    """One CLI run sharing a registry file. Returns {source: duplicate_of}."""
    args = [*inputs, "--provider", "offline", "--hash-registry", str(registry), "-o", "out.json"]
    assert score_videos.main(args) == 0
    records = json.loads(Path("out.json").read_text(encoding="utf-8"))
    records = records if isinstance(records, list) else [records]
    return {r["meta"]["source"]: r["meta"]["duplicate_of"] for r in records}


def test_registry_catches_a_duplicate_that_reuses_the_file_name(monkeypatch, tmp_path, clips):
    monkeypatch.chdir(tmp_path)
    registry = tmp_path / "registry.json"
    scored_run(registry, str(place(clips, "ideal10", tmp_path / "run1" / "clip.mp4")))

    re_encoded = place(clips, "ideal10_dup", tmp_path / "run2" / "clip.mp4")
    assert scored_run(registry, str(re_encoded))[re_encoded.as_posix()] is not None


def test_registry_across_runs_flags_the_real_duplicate_and_nothing_else(
    monkeypatch, tmp_path, clips
):
    monkeypatch.chdir(tmp_path)
    registry = tmp_path / "registry.json"
    original = place(clips, "ideal10", tmp_path / "run1" / "clip.mp4")
    place(clips, "tripod10", tmp_path / "run1" / "other.mp4")
    assert not any(scored_run(registry, str(tmp_path / "run1")).values())

    # A different clip that happens to share the name must neither match nor replace the original.
    same_name = place(clips, "other10", tmp_path / "run2" / "clip.mp4")
    assert scored_run(registry, str(same_name)) == {same_name.as_posix(): None}

    exact_copy = place(clips, "ideal10", tmp_path / "run3" / "reupload.mp4")
    re_encoded = place(clips, "ideal10_dup", tmp_path / "run3" / "re_encoded.mp4")
    third_run = scored_run(registry, str(tmp_path / "run3"))
    assert all(third_run[p.as_posix()] for p in (exact_copy, re_encoded))

    assert scored_run(registry, str(original)) == {original.as_posix(): None}


def test_exact_copy_in_one_batch_is_a_duplicate_of_the_first(tmp_path, clips):
    first = place(clips, "ideal10", tmp_path / "a" / "clip.mp4")
    second = place(clips, "ideal10", tmp_path / "b" / "clip.mp4")
    videos = resolve_inputs([str(first), str(second)], Settings())
    original, copy = QualityScorer(Settings(), workers=2).score_batch(videos)
    assert original["meta"]["duplicate_of"] is None
    assert copy["meta"]["duplicate_of"] == original["video_id"]
    assert copy["tier"] == "reject"


def test_scoring_the_same_file_again_is_not_a_duplicate():
    registry = HashRegistry()
    hashes = [0x0F0F0F0F0F0F0F0F] * 8
    assert registry.check_and_add("id", hashes, "a.mp4") is None
    assert registry.check_and_add("id", hashes, "a.mp4") is None


def test_same_bytes_from_elsewhere_is_a_duplicate_even_for_a_blank_clip():
    registry = HashRegistry()
    registry.check_and_add("id", [0] * 8, "a.mp4")
    match = registry.check_and_add("id", [0] * 8, "b.mp4")
    assert match is not None and match.is_duplicate and match.video_id == "id"


def test_the_first_location_stays_the_original(tmp_path):
    registry = HashRegistry(tmp_path / "registry.json")
    hashes = [0x0F0F0F0F0F0F0F0F] * 8
    registry.check_and_add("id", hashes, "a.mp4")
    registry.check_and_add("id", hashes, "b.mp4")
    registry.save()

    reloaded = HashRegistry(tmp_path / "registry.json")
    assert reloaded.check_and_add("id", hashes, "a.mp4") is None
    assert reloaded.check_and_add("id", hashes, "b.mp4").is_duplicate


def test_a_registry_from_before_content_ids_is_refused(tmp_path):
    path = tmp_path / "registry.json"
    path.write_text('{"some-id": ["0f0f0f0f0f0f0f0f"]}', encoding="utf-8")
    with pytest.raises(ValueError, match=r"registry\.json"):
        HashRegistry(path)


def test_origin_is_the_same_for_relative_and_absolute_paths(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    relative = VideoInput("clip.mp4", "clip.mp4", "id")
    absolute = VideoInput((tmp_path / "clip.mp4").as_posix(), "clip.mp4", "id")
    assert relative.origin == absolute.origin
    assert VideoInput("s3://bucket/a.mp4", "/cache/a.mp4", "id").origin == "s3://bucket/a.mp4"


def test_scorer_keeps_an_empty_registry_it_is_given():
    registry = HashRegistry()
    assert QualityScorer(Settings(), registry=registry).registry is registry


def test_first_run_saves_the_registry(monkeypatch, tmp_path, clips):
    monkeypatch.chdir(tmp_path)
    scored_run(tmp_path / "registry.json", str(clips["ideal10"]))
    assert len(HashRegistry(tmp_path / "registry.json")) == 1


def test_rescoring_a_clip_gives_the_same_answer_as_the_first_time():
    registry = HashRegistry()
    hashes = [0x0F0F0F0F0F0F0F0F] * 8
    registry.check_and_add("older", hashes, "older.mp4")
    assert registry.check_and_add("newer", hashes, "newer.mp4").video_id == "older"
    assert registry.check_and_add("older", hashes, "older.mp4") is None
    assert registry.check_and_add("newer", hashes, "newer.mp4").video_id == "older"
