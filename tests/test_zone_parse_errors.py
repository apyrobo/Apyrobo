"""A typo in a zone names the flag's format and the bad part, not a bare float error."""

import subprocess
import sys

import pytest

from apyrobo.mcp_server import load_policy_file, parse_zone


@pytest.mark.parametrize(
    ("spec", "field", "part"),
    [("1,x,2,2", "y_min", "'x'"), ("a,0,1,1", "x_min", "'a'"), ("0,0,1,2m", "y_max", "'2m'")],
)
def test_parse_zone_names_the_bad_part(spec, field, part):
    with pytest.raises(ValueError) as exc:
        parse_zone(spec)
    msg = str(exc.value)
    assert msg.startswith("no-go zone must be x_min,y_min,x_max,y_max in metres")
    assert f"{field} {part} is not a number" in msg
    assert repr(spec) in msg
    assert "could not convert" not in msg


def test_parse_zone_allowed_area_label():
    with pytest.raises(ValueError, match="^allowed area must be x_min,y_min,x_max,y_max"):
        parse_zone("0,0,one,1", "allowed area")


def test_policy_file_string_zone_names_the_bad_part(tmp_path):
    policy = tmp_path / "p.yaml"
    policy.write_text('no_go_zones: ["1,x,2,2"]\n')
    with pytest.raises(ValueError, match=r"no_go_zones\[0\]: no-go zone must be .* y_min 'x'"):
        load_policy_file(policy)


def test_policy_file_int_too_big_for_a_float_is_a_readable_error(tmp_path):
    policy = tmp_path / "p.yaml"
    big = "1" + "0" * 400
    policy.write_text(f"no_go_zones:\n  - {{x_min: {big}, y_min: 0, x_max: 1, y_max: 1}}\n")
    with pytest.raises(ValueError, match=r"no_go_zones\[0\]\.x_min must be a finite number"):
        load_policy_file(policy)


def test_cli_reports_the_zone_format_on_a_typo():
    out = subprocess.run(
        [sys.executable, "-m", "apyrobo.cli", "mcp", "selftest", "--no-go", "1,x,2,2"],
        capture_output=True, text=True, timeout=60,
    )
    assert out.returncode == 1
    err = out.stderr + out.stdout
    assert "no-go zone must be x_min,y_min,x_max,y_max in metres" in err
    assert "Traceback" not in err
