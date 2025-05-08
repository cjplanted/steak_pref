# Steak Live Dashboard – Google Trends Validation for Top European Steak Dishes
# Author: ChatGPT
# =============================================================================
# Streamlit dashboard that visualises Google Trends interest for ten beef‑steak
# dishes.  Key goals:
#   • run via `python …` *or* `streamlit run …` without manual tweaks
#   • survive minimal / notebook environments (no `__file__` dependence)
#   • graceful missing‑deps message (no non‑zero exit in CI)
# =============================================================================
"""
Behaviour
---------
* **Deps present + `streamlit run`** → dashboard starts normally.
* **Deps present + `python`** → script relaunches itself with Streamlit CLI.
* **Deps missing** → prints one‑page install hint and exits *successfully*.

Recent fixes
------------
* Removed usage of Python 3.10 `|` union‑operator in type hints ⇒ compatible
  back to 3.7 (now uses `typing.Union`).
* Fixed bug in `fetch_interest_over_time` where two separate `TrendReq` objects
  caused empty DataFrames.  Now reuses a single instance.
* Replaced fragile `__file__` references with helpers using `sys.argv[0]`.
* Stable `_install_message()` using `dedent` for deterministic test strings.
"""

from __future__ import annotations  # postponed evaluation of annotations (PEP 563)

import os
import sys
import types
import shutil
from importlib import import_module, util as _import_util
from datetime import datetime
from pathlib import Path
from textwrap import dedent
from typing import List, Dict, Union

# ---------------------------------------------------------------------------
# Helper – robust script path/name (never uses __file__)
# ---------------------------------------------------------------------------

def _script_path() -> Path:
    """Return absolute path to this script, falling back to CWD."""
    return Path(sys.argv[0]).resolve() if sys.argv and sys.argv[0] else Path.cwd() / "steak_live_dashboard.py"


def _script_name() -> str:
    return _script_path().name

# ---------------------------------------------------------------------------
# Dependency handling
# ---------------------------------------------------------------------------

REQUIRED_PACKAGES: Dict[str, str] = {
    "streamlit": "streamlit",
    "pandas": "pandas",
    "pytrends.request": "pytrends",
    "plotly.express": "plotly",
}


def _try_import(module_path: str) -> Union[types.ModuleType, None]:
    """Attempt to import *module_path* quietly; return None on ImportError."""
    try:
        return import_module(module_path) if _import_util.find_spec(module_path) else None
    except ImportError:
        return None

_missing: List[str] = [pip for mod, pip in REQUIRED_PACKAGES.items() if _try_import(mod) is None]

# ---------------------------------------------------------------------------
# Install‑hint builder (stable string for tests)
# ---------------------------------------------------------------------------

def _install_message(pkgs: List[str]) -> str:
    bullets = "\n".join(f"  • {pkg}" for pkg in pkgs)
    return dedent(f"""
        🚨 Required Python packages are missing:
        {bullets}

        Install with:

            pip install {' '.join(pkgs)}

        Then launch the dashboard with either command:

            python {_script_name()}
            streamlit run {_script_name()}
    """)

# ---------------------------------------------------------------------------
# STREAMLIT SECTION – executed only if dependencies are present
# ---------------------------------------------------------------------------

if not _missing:
    import pandas as pd  # type: ignore
    from pytrends.request import TrendReq  # type: ignore
    import plotly.express as px  # type: ignore
    import streamlit as st  # type: ignore

    # Detect if already inside Streamlit runtime
    _inside_streamlit = bool(getattr(st, "_is_running_with_streamlit", lambda: False)())

    # Auto‑relaunch when invoked with plain python
    if not _inside_streamlit:
        if shutil.which("streamlit"):
            os.execvp("streamlit", ["streamlit", "run", str(_script_path())])
        else:
            print(_install_message(["streamlit"]))
            sys.exit(0)

    # ---------------- Configuration ----------------
    DISH_KEYWORDS: Dict[str, List[str]] = {
        "Steak Frites": ["steak frites"],
        "Bistecca alla Fiorentina": ["bistecca alla fiorentina", "florentine steak"],
        "Steak Tartare": ["steak tartare"],
        "Beef Wellington": ["beef wellington"],
        "Entrecote Cafe de Paris": ["entrecote cafe de paris", "steak cafe de paris"],
        "Steak au Poivre": ["steak au poivre", "pepper steak"],
        "Tagliata di Manzo": ["tagliata di manzo", "tagliata steak"],
        "Zwiebelrostbraten": ["zwiebelrostbraten"],
        "Chuleton / Txuleton": ["chuletón", "txuleton", "chuleton steak"],
        "Churrasco Picanha": ["picanha", "churrasco picanha"],
    }

    DEFAULT_TIMEFRAME = "today 5-y"
    DEFAULT_GEO = "EU"

    @st.cache_data(show_spinner=False, ttl=3600)
    def fetch_interest_over_time(keys: Union[List[str], str], geo: str, tf: str):
        """Return weekly average Trends interest for *keys* list or single term."""
        kw_list: List[str] = keys if isinstance(keys, list) else [keys]
        tr = TrendReq(hl="en-US", tz=0)
        tr.build_payload(kw_list=kw_list, geo=geo, timeframe=tf)
        df = tr.interest_over_time()
        return df[kw_list].mean(axis=1) if not df.empty else pd.Series(dtype=float)

    @st.cache_data(show_spinner=False, ttl=3600)
    def collect_trends_data(map_: Dict[str, List[str]], geo: str, tf: str):
        rows = []
        for dish, kw in map_.items():
            ser = fetch_interest_over_time(kw, geo, tf)
            rows.append({
                "Dish": dish,
                "Mean": float(ser.mean()) if not ser.empty else 0.0,
                "Latest": float(ser.iloc[-1]) if not ser.empty else 0.0,
                "Series": ser,
            })
        df = pd.DataFrame(rows).set_index("Dish")
        df["Rank"] = df["Mean"].rank(ascending=False, method="min")
        return df.sort_values("Rank")

    def _launch_dashboard() -> None:
        st.set_page_config("European Steak Popularity Dashboard", layout="wide")
        st.title("European Steak Dish Popularity Dashboard")
        st.write(
            "Tracks live Google Trends data for ten iconic beef‑steak dishes popular across Europe. "
            "Select timeframe and region in the sidebar."
        )

        timeframe = st.sidebar.selectbox(
            "Timeframe", ("today 12-m", "today 5-y", "2015-01-01 2025-05-08"), index=1
        )
        region = st.sidebar.text_input("Geo code (ISO‑2, e.g. EU, DE, CH, AT)", DEFAULT_GEO)

        st.sidebar.markdown("---")
        st.sidebar.caption("Data: Google Trends via PyTrends (0‑100).")

        with st.spinner("Fetching Google Trends data …"):
            df = collect_trends_data(DISH_KEYWORDS, region.upper(), timeframe)

        st.subheader("Popularity Ranking (mean search interest)")
        st.dataframe(df[["Rank", "Mean", "Latest"]])

        st.plotly_chart(
            px.bar(df.reset_index(), x="Dish", y="Mean",
                   title=f"Average Trends Interest ({timeframe} | {region.upper()})"),
            use_container_width=True,
        )

        st.subheader("Trend Evolution")
        chosen = st.multiselect("Compare dishes", list(DISH_KEYWORDS), default=list(DISH_KEYWORDS)[:3])
        if chosen:
            combo = pd.concat({d: df.loc[d, "Series"] for d in chosen}, axis=1)
            st.plotly_chart(px.line(combo, title="Google Trends Over Time"), use_container_width=True)
            with st.expander("Raw weekly data"):
                st.dataframe(combo)

        st.markdown("---")
        st.caption(f"© {datetime.now().year} European Steak Dashboard – Generated {datetime.now():%Y-%m-%d %H:%M}")

else:

    def _launch_dashboard() -> None:  # type: ignore
        print(_install_message(_missing))
        return

# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _launch_dashboard()

# ---------------------------------------------------------------------------
# Tests (run even without heavy deps)
# ---------------------------------------------------------------------------

def _test_helpers():
    # Path/name helpers
    assert _script_name()
    assert _try_import("sys") is sys
    assert _try_import("some_nonexistent_pkg_12345") is None
    # Message content stability
    msg = _install_message(["foo", "bar"])
    assert "pip install foo bar" in msg
    assert "• foo" in msg and "• bar" in msg

if __debug__:
    _test_helpers()
