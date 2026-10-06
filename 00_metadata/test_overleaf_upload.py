"""Facts about telling Overleaf's drag-and-drop project what changed, on tiny tmp_path folders."""

from pathlib import Path

import pytest
from overleaf_upload import UploadDiff, diff_upload, main, report

CURRENT = ("a.tex", "b.png", "c.tex")
PREVIOUS = ("a.tex", "old.png")


class TestDiffUpload:
    @pytest.fixture
    def diff(self) -> UploadDiff:
        return diff_upload(CURRENT, PREVIOUS)

    def test_names_the_files_new_since_the_last_upload(self, diff) -> None:
        assert diff.added == ("b.png", "c.tex")

    def test_names_the_files_gone_since_the_last_upload(self, diff) -> None:
        assert diff.removed == ("old.png",)

    def test_names_the_files_in_both(self, diff) -> None:
        assert diff.unchanged == ("a.tex",)


def test_without_a_previous_upload_every_file_is_added() -> None:
    assert diff_upload(CURRENT, ()).added == CURRENT


class TestReport:
    @pytest.fixture
    def lines(self) -> list[str]:
        return report(diff_upload(CURRENT, PREVIOUS)).splitlines()

    def test_lists_the_added_files(self, lines) -> None:
        assert lines[0] == "added: b.png, c.tex"

    def test_lists_the_removed_files_to_delete_on_overleaf(self, lines) -> None:
        assert lines[1] == "removed since the last upload (delete these on Overleaf): old.png"

    def test_ends_with_the_counts(self, lines) -> None:
        assert lines[-1] == "2 added, 1 removed, 1 unchanged"


def test_report_says_none_when_nothing_was_removed() -> None:
    assert "(delete these on Overleaf): none" in report(diff_upload(CURRENT, CURRENT))


class TestMain:
    @pytest.fixture
    def folder(self, tmp_path: Path) -> Path:
        folder = tmp_path / "overleaf"
        folder.mkdir()
        for name in CURRENT:
            (folder / name).write_text(name)
        return folder

    def test_first_run_reports_every_file_added(self, folder, tmp_path, capsys) -> None:
        main([str(folder), str(tmp_path / "list" / "last-upload.txt")])
        assert capsys.readouterr().out.splitlines()[-1] == "3 added, 0 removed, 0 unchanged"

    def test_writes_the_new_list(self, folder, tmp_path) -> None:
        listing = tmp_path / "list" / "last-upload.txt"
        main([str(folder), str(listing)])
        assert listing.read_text() == "a.tex\nb.png\nc.tex\n"

    def test_second_run_with_nothing_changed_reports_nothing(
        self, folder, tmp_path, capsys
    ) -> None:
        listing = str(tmp_path / "last-upload.txt")
        main([str(folder), listing])
        main([str(folder), listing])
        assert capsys.readouterr().out.splitlines()[-1] == "0 added, 0 removed, 3 unchanged"

    def test_refuses_a_folder_with_a_subfolder(self, folder, tmp_path) -> None:
        (folder / "nested").mkdir()
        with pytest.raises(AssertionError, match="flat"):
            main([str(folder), str(tmp_path / "last-upload.txt")])
