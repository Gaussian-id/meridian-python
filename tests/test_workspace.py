import re

import meridian


def test_the_package_reports_a_semver_version() -> None:
    assert re.fullmatch(r'\d+\.\d+\.\d+', meridian.__version__)
