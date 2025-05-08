# Steak Live Dashboard – Google Trends Validation for Top European Steak Dishes
# Author: ChatGPT
# =============================================================================
# Streamlit dashboard that visualises Google Trends interest for ten beef‑steak
# dishes. Runs both locally (`python …` or `streamlit run …`) **and** on
# Streamlit Community Cloud. Compatible back to Python 3.7.
# =============================================================================
"""
Key behaviour
-------------
* `streamlit run steak_live_dashboard.py` → dashboard starts immediately.
* `python steak_live_dashboard.py` locally → identical behaviour (no relaunch logic).
* In Streamlit Cloud the script runs once; port 8501 opens for health‑check.
* Missing dependencies trigger a one‑page install hint, *exit 0*.
"""

from __future__ import annotations

import os
import sys
import types
from importlib import import_module, util as _import_util
from datetime import datetime
from pathlib import Path
from textwrap import dedent
from typing import List, Dict, Union

# ---------------------------------------------------------------------------
# Helper – robust script path/name (never uses __file__)
# ---------------------------------------------------------------------------

def _script_path() -> Path:
    return Path(sys.argv[0]).resolve() if sys.argv and sys.argv[0] else Path.cwd() / "steak_live_dashboard.py"


def _script_name() -> str:
    return _script_path().name

# ---------------------------------------------------------------------------
# Dependency handling
# ---------------------------------------------------------------------------

REQUIRED_PACKAGES: Dict[str, str] = {
    "streamlit": "streamlit",
    "pandas": "pandas",
    "pytrends": "pytrends",
    "plotly": "plotly",
}


def _try_import(module_path: str) -> Union[types.ModuleType, None]:
    try:
        return import_module(module_path) if _import_util.find_spec(module_path) else None
    except ImportError:
        return None

_missing: List[str] = [pip for mod, pip in REQUIRED_PACKAGES.items() if _try_import(mod) is None]

# ---------------------------------------------------------------------------
# Install‑hint builder
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
# STREAMLIT SECTION (imports only if deps are present)
# ---------------------------------------------------------------------------

if not _missing:
    import pandas as pd  # type: ignore
    from pytrends.request import TrendReq  # type: ignore
    import plotly.express as px  # type: ignore
    import streamlit as st  # type: ignore

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

    # ---------------- Data helpers ----------------
    @st.cache_data(show_spinner=False, ttl=3600)
    def fetch_interest_over_time(keys: Union[List[str], str], geo: str, tf: str):
        """Return weekly average Trends interest or empty series on network error."""
        kw = keys if isinstance(keys, list) else [keys]
        try:
            tr = TrendReq(
                hl="en-US",
                tz=0,
                requests_args={"timeout": (10, 25)},
                retries=2,
                backoff_factor=0.3,
            )
            tr.build_payload(kw_list=kw, geo=geo, timeframe=tf)
            df = tr.interest_over_time()
            if df.empty:
                st.warning(f"No Google Trends data returned for {', '.join(kw)}.")
                return pd.Series(dtype=float)
            return df[kw].mean(axis=1)
        except Exception as err:  # noqa: BLE001
            st.error(
                f"Google Trends request for {', '.join(kw)} failed: {err}. "
                "If this persists, reload later or run the app locally."
            )
            return pd.Series(dtype=float)

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

    # ---------------- UI helpers ----------------
    def _render_charts(df, timeframe: str, region: str):
        st.subheader("Popularity Ranking (mean search interest)")
        st.dataframe(df[["Rank", "Mean", "Latest"]])

        st.plotly_chart(
            px.bar(
                df.reset_index(),
                x="Dish",
                y="Mean",
                title=f"Average Trends Interest ({timeframe} | {region.upper()})",
                height=420,
            ),
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

    def _launch_dashboard() -> None:
        st.set_page_config("European Steak Popularity Dashboard", layout="wide")
        st.title("European Steak Dish Popularity Dashboard")
        st.write(
            "Fetch and compare live Google Trends data for ten iconic beef‑steak dishes popular across Europe.\n"
            "First data pull can take up to 30 s; click **Load data** when ready."
        )

        timeframe = st.sidebar.selectbox("Timeframe", ("today 12-m", "today 5-y", "2015-01-01 2025-05-08"), index=1)
        region = st.sidebar.text_input("Geo code (ISO‑2, e.g. EU, DE, CH, AT)", DEFAULT_GEO)
        st.sidebar.caption("Data: Google Trends via PyTrends (0‑100 scale).")
        st.sidebar.markdown("---")

        if st.button("🔄 Load data / refresh", type="primary"):
            with st.spinner("Fetching Google Trends data… this may take up to 30 s"):
                df = collect_trends_data(DISH_KEYWORDS, region.upper(), timeframe)
            _render_charts(df, timeframe, region)
        else:
            st.info("Press **Load data / refresh** to query Google Trends.")

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
# Lightweight tests (safe without heavy deps)
# ---------------------------------------------------------------------------

def _test_helpers():
    assert _script_name()
    assert _try_import("sys") is sys
    assert _try_import("pkg_does_not_exist_12345") is None
    msg = _install_message(["foo", "bar"])
    assert "pip install foo bar" in msg and "• foo" in msg and "• bar" in msg

if __debug__ and not os.getenv("STREAMLIT_SERVER_PORT"):
    _test_helpers()
