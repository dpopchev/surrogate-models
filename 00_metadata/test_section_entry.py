"""Facts about the stand-alone entry file of one paper section, on a tiny article entry."""

from pathlib import Path

import pytest
from section_entry import main, section_entry, sections

ARTICLE = (
    "\\documentclass{article}\n"
    "\\input{preamble}\n"
    "\\input{x_num}\n"
    "\\begin{document}\n"
    "\\maketitle\n"
    "\\input{10_a}\n"
    "\\input{20_b}\n"
    "\\bibliographystyle{plain}\n"
    "\\bibliography{refs}\n"
    "\\end{document}\n"
)


def test_sections_are_the_inputs_of_the_document_body() -> None:
    assert sections(ARTICLE) == ("10_a", "20_b")


class TestSectionEntry:
    @pytest.fixture
    def entry(self) -> str:
        return section_entry(ARTICLE, "20_b", cites=False)

    def test_keeps_the_section(self, entry) -> None:
        assert "\\input{20_b}" in entry

    def test_drops_the_other_sections(self, entry) -> None:
        assert "\\input{10_a}" not in entry

    def test_keeps_the_generated_numbers(self, entry) -> None:
        assert "\\input{x_num}" in entry

    def test_drops_the_title(self, entry) -> None:
        assert "\\maketitle" not in entry

    def test_loads_xr_hyper_before_the_preamble_loads_hyperref(self, entry) -> None:
        assert entry.index("\\usepackage{xr-hyper}") < entry.index("\\input{preamble}")

    def test_reads_the_labels_of_the_full_build(self, entry) -> None:
        assert "\\externaldocument{main}" in entry

    def test_drops_the_bibliography_when_the_section_cites_nothing(self, entry) -> None:
        assert "\\bibliography{refs}" not in entry


def test_loads_xr_hyper_after_the_documentclass_below_leading_comments() -> None:
    entry = section_entry(f"% a comment\n{ARTICLE}", "20_b", cites=False)
    assert entry.index("\\documentclass") < entry.index("\\usepackage{xr-hyper}")


def test_keeps_the_bibliography_when_the_section_cites() -> None:
    assert "\\bibliography{refs}" in section_entry(ARTICLE, "20_b", cites=True)


def test_an_unknown_section_names_the_valid_ones() -> None:
    with pytest.raises(ValueError, match="10_a, 20_b"):
        section_entry(ARTICLE, "99_z", cites=False)


class TestMain:
    @pytest.fixture
    def written(self, tmp_path: Path) -> Path:
        article, folder = tmp_path / "article.tex", tmp_path / "20_b"
        article.write_text(ARTICLE)
        (folder / "21_c").mkdir(parents=True)
        (folder / "20_b.tex").write_text("\\section{B}\n")
        (folder / "21_c" / "21_c.tex").write_text("as shown \\citep{x}\n")
        main([str(article), str(folder), str(tmp_path / "out")])
        return tmp_path / "out" / "section_20_b.tex"

    def test_writes_the_entry_named_after_the_section(self, written) -> None:
        assert "\\input{20_b}" in written.read_text()

    def test_keeps_the_bibliography_for_a_citation_in_a_subfolder(self, written) -> None:
        assert "\\bibliography{refs}" in written.read_text()
