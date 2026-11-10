from pathlib import Path

import pytest

from nrly.contracts import CaseSpec, ExperimentSpec, NrlyError
from nrly.review import render_run_report, write_run_report
from nrly.store import Store


def test_report_escapes_untrusted_text_and_labels_unscored_fixture(tmp_path: Path) -> None:
    record, _ = Store(tmp_path / "store").create(
        ExperimentSpec(
            name='<script>alert("name")</script>',
            cases=[CaseSpec(id="one", input="[2,4]", expected="</pre><script>attack()</script>")],
        )
    )
    report = render_run_report(record)
    assert "<script>" not in report
    assert "&lt;script&gt;" in report
    assert "Content-Security-Policy" in report
    assert "Engineering fixture" in report
    assert "Unscored" in report
    destination = tmp_path / "report.html"
    assert write_run_report(record, destination) == destination
    with pytest.raises(NrlyError, match="already exists"):
        write_run_report(record, destination)
