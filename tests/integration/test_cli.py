import pytest

from jitpi05.cli.jitrl import build_parser
from jitpi05.reporting.generate import build_parser as build_report_parser


@pytest.mark.integration
def test_jitrl_cli_help(capsys) -> None:
    with pytest.raises(SystemExit) as result:
        build_parser().parse_args(["--help"])
    assert result.value.code == 0
    help_text = capsys.readouterr().out
    assert "--summarize-only" in help_text
    assert "--counterfactual-recovery" in help_text


@pytest.mark.integration
def test_report_cli_requires_known_profile() -> None:
    args = build_report_parser().parse_args(["--profile", "positive-reward"])
    assert args.profile == "positive-reward"
