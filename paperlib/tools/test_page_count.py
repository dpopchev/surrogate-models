"""Facts about the page count of the main text and the appendix read from a LaTeX aux file."""

from pathlib import Path

import pytest

from paperlib.tools.page_count import main, read_pages

pytestmark = pytest.mark.model

AUX = (
    "\\newlabel{sec:intro}{{1}{1}{Introduction}{section.1}{}}\n"
    "\\newlabel{sec:intro@cref}{{[section][1][]1}{[1][1][]1}{}{}{}}\n"
    "\\newlabel{sec:eda}{{A}{4}{Data}{appendix.A}{}}\n"
    "\\newlabel{fig:last}{{7}{6}{Last figure}{figure.7}{}}\n"
)


def test_page_count_reads_the_main_text_and_the_appendix() -> None:
    pages = read_pages(AUX)
    assert (pages.main, pages.appendix) == (3, 3)


def test_aux_without_an_appendix_is_rejected() -> None:
    with pytest.raises(ValueError, match="appendix"):
        read_pages("\\newlabel{sec:intro}{{1}{1}{Introduction}{section.1}{}}\n")


def test_main_prints_the_pages_and_warns_when_the_appendix_is_longer(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    aux = tmp_path / "main.aux"
    aux.write_text(AUX.replace("{A}{4}", "{A}{2}"))
    main([str(aux)])
    assert capsys.readouterr().out == (
        "main text 1 pages, appendix 5 pages\n"
        "[WARN] appendix longer than the main text by 4 pages\n"
    )


def test_main_says_so_when_the_appendix_is_no_longer(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    aux = tmp_path / "main.aux"
    aux.write_text(AUX.replace("{7}{6}", "{7}{4}"))
    main([str(aux)])
    assert "appendix no longer than the main text" in capsys.readouterr().out


def test_missing_aux_file_names_its_path(tmp_path: Path) -> None:
    missing = tmp_path / "none.aux"
    with pytest.raises(FileNotFoundError, match=r"none\.aux"):
        main([str(missing)])
