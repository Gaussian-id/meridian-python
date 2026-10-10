import re

import gauss_meridian


def test_the_package_reports_a_semver_version() -> None:
    assert re.fullmatch(r'\d+\.\d+\.\d+', gauss_meridian.__version__)
