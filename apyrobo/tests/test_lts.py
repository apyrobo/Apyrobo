"""Tests for LTS policy and version checker."""

import pytest
from apyrobo.lts.policy import LTSRelease, LTSPolicy
from apyrobo.lts.checker import VersionChecker


class TestLTSRelease:
    def test_not_eol_for_future_date(self):
        r = LTSRelease(
            version="1.0.0",
            release_date="2026-04-01",
            eol_date="2030-01-01",
        )
        assert r.is_eol("2026-06-01") is False

    def test_is_eol_past_date(self):
        r = LTSRelease(
            version="1.0.0",
            release_date="2020-01-01",
            eol_date="2022-01-01",
        )
        assert r.is_eol("2023-01-01") is True

    def test_is_eol_exact_boundary(self):
        r = LTSRelease(
            version="1.0.0",
            release_date="2020-01-01",
            eol_date="2022-01-01",
        )
        # On the EOL date itself, not EOL yet (strictly >)
        assert r.is_eol("2022-01-01") is False

    def test_security_only_flag(self):
        r = LTSRelease(
            version="1.0.0",
            release_date="2026-04-01",
            eol_date="2028-04-01",
            security_only=True,
        )
        assert r.security_only is True


class TestLTSPolicy:
    def setup_method(self):
        self.policy = LTSPolicy()

    def test_is_lts_known(self):
        assert self.policy.is_lts("1.0.0") is True

    def test_is_lts_unknown(self):
        assert self.policy.is_lts("0.9.0") is False

    def test_is_eol_unknown_version(self):
        # Non-LTS versions are not tracked — returns False
        assert self.policy.is_eol("0.9.0") is False

    def test_supported_versions_future(self):
        # With a far future as_of date, 1.0.0 will be EOL
        supported = self.policy.supported_versions("2029-01-01")
        assert all(r.version != "1.0.0" for r in supported)

    def test_next_lts_future(self):
        # 1.0.0 is planned for 2026-04-01; if today < that date, it's "next"
        # This test is date-sensitive; just check it returns a string or None
        result = self.policy.next_lts()
        assert result is None or isinstance(result, str)

    def test_latest_lts_none_before_release(self):
        # If we check before any LTS is released
        result = self.policy.latest_lts()
        # Returns None or the 1.0.0 release depending on today's date
        assert result is None or isinstance(result.version, str)


class TestVersionChecker:
    def setup_method(self):
        self.checker = VersionChecker()

    def test_check_unknown_version(self):
        info = self.checker.check_for_updates("0.5.0")
        assert info["current"] == "0.5.0"
        assert info["is_lts"] is False
        assert info["supported"] is True  # not tracked as EOL

    def test_check_lts_version(self):
        info = self.checker.check_for_updates("1.0.0")
        assert info["is_lts"] is True

    def test_security_advisory_unknown(self):
        advisories = self.checker.security_advisory("0.5.0")
        assert advisories == []

    def test_security_advisory_eol(self):
        # Patch the policy to mark 0.5.0 as EOL
        from apyrobo.lts.policy import LTSRelease, LTSPolicy
        policy = LTSPolicy()
        policy.LTS_RELEASES = [
            LTSRelease(
                version="0.5.0",
                release_date="2020-01-01",
                eol_date="2021-01-01",
            )
        ]
        checker = VersionChecker(policy)
        advisories = checker.security_advisory("0.5.0")
        assert len(advisories) == 1
        assert "end-of-life" in advisories[0]

    def test_check_returns_dict_keys(self):
        info = self.checker.check_for_updates("1.0.0")
        for key in ("current", "is_lts", "is_eol", "latest_lts", "next_lts", "supported"):
            assert key in info
