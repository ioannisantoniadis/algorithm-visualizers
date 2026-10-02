"""Every page renders without an exception, driven the way users reach it: through
Home.py's st.navigation. (Running apps/<page>.py directly would put apps/ first on the
import path, where apps/attention.py shadows the attention package.)"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
PAGES = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "apps").glob("*.py"))


def test_there_are_21_pages():
    assert len(PAGES) == 21


def test_home_renders():
    at = AppTest.from_file(str(ROOT / "Home.py"), default_timeout=120).run()
    assert not at.exception


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_with_charts(page):
    at = AppTest.from_file(str(ROOT / "Home.py"), default_timeout=300).run()
    at.switch_page(page).run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.get("plotly_chart")) >= 1
