# steak_live_dashboard.py — European Steak Popularity (Google Trends)
# =============================================================================
# Streamlit dashboard that compares Google‑Trends search interest for ten iconic
# beef‑steak dishes across Europe.  Runs locally (`streamlit run …`) and on
# Streamlit Community Cloud.  Works with Python ≥ 3.7.
# =============================================================================
"""Quick start (local)
$ pip install streamlit pandas pytrends plotly requests
$ streamlit run steak_live_dashboard.py

For Streamlit Cloud add this *requirements.txt*:
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
# Script path helpers (safe when __file__ is undefined)
# ---------------------------------------------------------------------------

SCRIPT_PATH: Path = Path(sys.argv[0]).resolve() if sys.argv and sys.argv[0] else Path.cwd() / "steak_live_dashboard.py"
SCRIPT_NAME: str = SCRIPT_PATH.name

# ---------------------------------------------------------------------------
# Lazy‑import check — quit gracefully if heavy deps are missing
# ---------------------------------------------------------------------------

REQUIRED_PKGS: Dict[str, str] = {
    "streamlit": "streamlit",
    "pandas": "pandas",
    "pytrends": "pytrends",
    "plotly": "plotly",
}


def _try_import(mod: str) -> Optional[types.ModuleType]:
    try:
        return import_module(mod) if import_util.find_spec(mod) else None
    except ImportError:
        return None

missing = [pip for mod, pip in REQUIRED_PKGS.items() if _try_import(mod) is None]

# ---------------------------------------------------------------------------
# Install‑hint for headless / missing‑dep situations
# ---------------------------------------------------------------------------

def _install_hint(pkgs: List[str]) -> str:
    bullet_lines = "\n".join(f"  • {p}" for p in pkgs)
    return dedent(f"""
        🚨 Missing Python packages:
        {bullet_lines}

        Install and rerun:

            pip install {' '.join(pkgs)}

        Then launch the dashboard:

            streamlit run {SCRIPT_NAME}
    """)

# ---------------------------------------------------------------------------
# Main Streamlit dashboard — only defined if deps are present
# ---------------------------------------------------------------------------

if not missing:
    import pandas as pd  # type: ignore
    import plotly.express as px  # type: ignore
    import streamlit as st  # type: ignore
    from pytrends.request import TrendReq  # type: ignore

    # --------------------- Configuration ---------------------
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

    # --------------------- Data helpers ---------------------
    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch_trends(keywords: List[str] | str, geo: str, timeframe: str) -> pd.Series:
        """Return weekly average Google‑Trends interest; retry once; polite UA."""
        kw_list = keywords if isinstance(keywords, list) else [keywords]
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123 Safari/537.36"
            )
        }
        for attempt in (1, 2):
            try:
                tr = TrendReq(hl="en-US", tz=0, timeout=(10, 30), requests_args={"headers": headers})
                tr.build_payload(kw_list=kw_list, geo=geo, timeframe=timeframe)
                df = tr.interest_over_time()
                if df.empty:
                    raise ValueError("empty dataframe")
                return df[kw_list].mean(axis=1)
            except Exception as exc:  # pragma: no cover
                if attempt == 2:
                    st.warning(
                        f"Trend fetch failed for {', '.join(kw_list)} → {exc}. "
                        "Google may block anonymous cloud requests. Try again later or run locally."
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

    # --------------------- UI helpers ---------------------
    def render_charts(df: pd.DataFrame, timeframe: str, region: str) -> None:
        st.subheader("Popularity ranking (mean search interest)")
        st.dataframe(df[["Rank", "Mean", "Latest"]])

        st.plotly_chart(
            px.bar(
                df.reset_index(), x="Dish", y="Mean",
                title=f"Average Trends Interest — {region.upper()} ({timeframe})",
                height=420,
            ),
            use_container_width=True,
        )

        st.subheader("Trend evolution")
        chosen = st.multiselect("Compare dishes", list(DISH_KEYWORDS), default=list(DISH_KEYWORDS)[:3])
        if chosen:
            combined = pd.concat({d: df.loc[d, "Series"] for d in chosen}, axis=1)
            st.plotly_chart(px.line(combined, title="Weekly Interest"), use_container_width=True)
            with st.expander("Raw weekly data"):
                st.dataframe(combined)

        st.caption(f"Updated {datetime.now():%Y-%m-%d %H:%M}")

    # --------------------- Streamlit app ---------------------
    def main() -> None:
        st.set_page_config("European Steak Popularity", layout="wide")
        st.title("European Steak‑Dish Popularity Dashboard")
        st.write("Live Google‑Trends comparison for ten signature beef‑steak dishes across Europe.")

        timeframe = st.sidebar.selectbox("Timeframe", ("today 12-m", "today 5-y", "2015-01-01 2025-05-08"), index=1)
        region = st.sidebar.text_input("Geo code (ISO‑2, e.g. EU, DE, CH)", DEFAULT_GEO)
        st.sidebar.markdown("---")

        if st.button("🔄 Load / refresh data", type="primary"):
            with st.spinner("Fetching Google Trends … may take 15–30 s"):
                df = collect_trends(region.upper(), timeframe)
            render_charts(df, timeframe, region)
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
# Basic self‑test (skipped on Streamlit Cloud)
# ---------------------------------------------------------------------------

def _self_test() -> None:
    assert SCRIPT_NAME
    assert _try_import("sys") is sys
    assert _try_import("surely_nonexistent_pkg___") is None
    assert "pip install foo bar" in _install_hint(["foo", "bar"])

if __debug__ and not os.getenv("STREAMLIT_SERVER_PORT"):
    _self_test()
