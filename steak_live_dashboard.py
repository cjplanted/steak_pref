# steak_live_dashboard.py  ─ European Steak Popularity (Google Trends)
# =============================================================================
# Streamlit dashboard: compares Google Trends search interest for ten iconic
# beef‑steak dishes across Europe.  Runs both locally and on Streamlit Cloud.
# Compatible with Python ≥ 3.7.
# =============================================================================
"""Quick start (local)
$ pip install streamlit pandas pytrends plotly requests
$ streamlit run steak_live_dashboard.py

For Streamlit Cloud include this file **plus** a requirements.txt:
    streamlit>=1.35
    pandas
    pytrends
    plotly
    requests
"""

from __future__ import annotations

import os
import sys
import types
from datetime import datetime
from importlib import import_module, util as import_util
from pathlib import Path
from textwrap import dedent
from typing import List, Dict, Optional

# ---------------------------------------------------------------------------
# Resolve script name / path (safe in Cloud & interactive envs)
# ---------------------------------------------------------------------------

SCRIPT_PATH = Path(sys.argv[0]).resolve() if sys.argv and sys.argv[0] else Path.cwd() / "steak_live_dashboard.py"
SCRIPT_NAME = SCRIPT_PATH.name

# ---------------------------------------------------------------------------
# Optional imports – only load heavy deps if present
# ---------------------------------------------------------------------------

REQUIRED_PKG: Dict[str, str] = {
    "streamlit": "streamlit",
    "pandas": "pandas",
    "pytrends": "pytrends",
    "plotly": "plotly",
}


def _try_import(name: str) -> Optional[types.ModuleType]:
    """Attempt to import *name*; return None on failure without raising."""
    try:
        return import_module(name) if import_util.find_spec(name) else None
    except ImportError:
        return None

missing = [pip for mod, pip in REQUIRED_PKG.items() if _try_import(mod) is None]

# ---------------------------------------------------------------------------
# Fallback install hint (printed when deps are absent)
# ---------------------------------------------------------------------------

def _install_hint(pkgs: List[str]) -> str:
    bullets = "\n".join(f"  • {p}" for p in pkgs)
    return dedent(
        f"""
        🚨 Missing Python packages:
        {bullets}

        Install and rerun:

            pip install {' '.join(pkgs)}

        Then start the dashboard:

            streamlit run {SCRIPT_NAME}
        """
    )

# ---------------------------------------------------------------------------
# Main dashboard (only if all dependencies exist)
# ---------------------------------------------------------------------------

if not missing:
    import pandas as pd  # type: ignore
    import plotly.express as px  # type: ignore
    import streamlit as st  # type: ignore
    from pytrends.request import TrendReq  # type: ignore

    # -------------------- config --------------------
    DISH_KEYWORDS: Dict[str, List[str]] = {
        "Steak Frites": ["steak frites"],
        "Bistecca alla Fiorentina": ["bistecca alla fiorentina", "florentine steak"],
        "Steak Tartare": ["steak tartare"],
        "Beef Wellington": ["beef wellington"],
        "Entrecôte Café de Paris": ["entrecote cafe de paris", "steak cafe de paris"],
        "Steak au Poivre": ["steak au poivre", "pepper steak"],
        "Tagliata di Manzo": ["tagliata di manzo", "tagliata steak"],
        "Zwiebelrostbraten": ["zwiebelrostbraten"],
        "Txuleton (Chuleta)": ["chuletón", "txuleton", "chuleton steak"],
        "Picanha (Churrasco)": ["picanha", "churrasco picanha"],
    }

    DEFAULT_TIMEFRAME = "today 5-y"
    DEFAULT_GEO = "EU"

    # -------------------- data helpers --------------------
    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch_trends(keywords: List[str] | str, geo: str, timeframe: str) -> pd.Series:
        """Return weekly average Google Trends interest; retry once on failure."""
        kw_list = keywords if isinstance(keywords, list) else [keywords]
        for attempt in (1, 2):
            try:
                tr = TrendReq(hl="en-US", tz=0, timeout=(10, 30))  # connect/read
                tr.build_payload(kw_list=kw_list, geo=geo, timeframe=timeframe)
                df = tr.interest_over_time()
                if df.empty:
                    return pd.Series(dtype=float)
                return df[kw_list].mean(axis=1)
            except Exception as exc:  # pragma: no cover
                if attempt == 2:
                    st.warning(
                        f"Trend fetch failed for {', '.join(kw_list)} → {exc}. "
                        "Google may throttle anonymous requests. Retry later."
                    )
        return pd.Series(dtype=float)

    @st.cache_data(ttl=3600, show_spinner=False)
    def collect_trends(geo: str, timeframe: str) -> pd.DataFrame:
        rows = []
        for dish, kw in DISH_KEYWORDS.items():
            ser = fetch_trends(kw, geo, timeframe)
            rows.append({
                "Dish": dish,
                "Mean": float(ser.mean()) if not ser.empty else 0.0,
                "Latest": float(ser.iloc[-1]) if not ser.empty else 0.0,
                "Series": ser,
            })
        df = pd.DataFrame(rows).set_index("Dish")
        df["Rank"] = df["Mean"].rank(ascending=False, method="min")
        return df.sort_values("Rank")

    # -------------------- UI helpers --------------------
    def render_charts(df: pd.DataFrame, timeframe: str, region: str) -> None:
        st.subheader("Popularity ranking (mean search interest)")
        st.dataframe(df[["Rank", "Mean", "Latest"]])

        st.plotly_chart(
            px.bar(
                df.reset_index(),
                x="Dish",
                y="Mean",
                title=f"Average Trends Interest — {region.upper()} ({timeframe})",
                height=420,
            ),
            use_container_width=True,
        )

        st.subheader("Trend evolution")
        chosen = st.multiselect("Compare dishes", list(DISH_KEYWORDS), default=list(DISH_KEYWORDS)[:3])
        if chosen:
            combo = pd.concat({d: df.loc[d, "Series"] for d in chosen}, axis=1)
            st.plotly_chart(px.line(combo, title="Weekly Interest"), use_container_width=True)
            with st.expander("Raw weekly data"):
                st.dataframe(combo)

        st.caption(f"Updated {datetime.now():%Y-%m-%d %H:%M}")

    # -------------------- main Streamlit app --------------------
    def main() -> None:
        st.set_page_config("European Steak Popularity", layout="wide")
        st.title("European Steak‑Dish Popularity Dashboard")
        st.write(
            "Live Google Trends comparison for ten famous beef‑steak dishes across Europe."
        )

        timeframe = st.sidebar.selectbox("Timeframe", ("today 12-m", "today 5-y", "2015-01-01 2025-05-08"), index=1)
        region = st.sidebar.text_input("Geo code (ISO‑2, e.g. EU, DE, CH)", DEFAULT_GEO)
        st.sidebar.markdown("---")

        if st.button("🔄 Load / refresh data", type="primary"):
            with st.spinner("Fetching Google Trends … may take 15–30 s"):
                df_trends = collect_trends(region.upper(), timeframe)
            render_charts(df_trends, timeframe, region)
        else:
            st.info("Click the button to query Google Trends.")

else:
    def main() -> None:  # type: ignore
        print(_install_hint(missing))

# ---------------------------------------------------------------------------
# Run script
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# Self‑test (skipped on Streamlit Cloud)
# ---------------------------------------------------------------------------

def _self_test() -> None:
    assert SCRIPT_NAME
    assert _try_import("sys") is sys
    assert _try_import("nonexistent_pkg_12345") is None
    assert "pip install foo bar" in _install_hint(["foo", "bar"])

if __debug__ and not os.getenv("STREAMLIT_SERVER_PORT"):
    _self_test()
