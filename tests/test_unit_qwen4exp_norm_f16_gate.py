"""Retired publication arms cannot silently run as a no-op campaign."""
import pytest

from scripts.qwen4exp_halo_box_campaign_ab import build_parser


def test_publication_arm_is_not_available():
    parser = build_parser()
    assert "norm-f16-publish" not in parser._option_string_actions["--route-package"].choices
    with pytest.raises(SystemExit):
        parser.parse_args(["--route-package", "norm-f16-publish"])
