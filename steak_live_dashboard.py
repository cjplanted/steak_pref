# steak_live_dashboard.py  ─ European Steak Popularity (Google Trends)
# =============================================================================
# Streamlit dashboard: compares search interest for 10 iconic beef‑steak dishes
# across Europe using the Google Trends API (PyTrends).
# • Runs locally via  `streamlit run`  *or*  `python steak_live_dashboard.py`
# • Runs on Streamlit Community Cloud without extra configuration
# • Compatible with Python ≥ 3.7
# =============================================================================
"""Minimal usage
$ pip install streamlit pandas pytrends plotly
$ streamlit run steak_live_dashboard.py

On Streamlit Cloud place this file and *requirements.txt* in your repo:
    streamlit>=1.35
    pandas
    pytrends
    plotly
    requests

The first data pull may take ~30 s because Google throttles anonymous queries.
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

# -----------------------------------------------------------------------------
# Utility – robust script path
# -----------------------------------------------------------------------------

SCRIPT_PATH: Path = Path(sys.argv[0]).resolve() if sys.argv and sys.argv[0] else Path.cwd() / "steak_live_dashboard.py"
SCRIPT_NAME: str = SCRIPT_PATH.name

# -----------------------------------------------------------------------------
# Optional imports – decide at runtime whether to launch Streamlit UI or print
# an install hint when dependencies are missing.
# -----------------------------------------------------------------------------

REQUIRED_PKG: Dict[str, str] = {
    "streamlit": "streamlit",
    "pandas": "pandas",
    "pytrends": "pytrends",
    "plotly": "plotly",
}


def _try_import(mod: str) -> Optional[types.ModuleType]:
    """Import *mod* if found else return *None* without raising."""
    try:
        return import_module(mod) if import_util.find_spec(mod) else None
    except ImportError:
        return None

missing_pkgs: List[str] = [pip for mod, pip in REQUIRED_PKG.items() if _try_import(mod) is None]

# -----------------------------------------------------------------------------
# Pretty install hint (only printed in headless/no‑deps situations)
# -----------------------------------------------------------------------------

def _install_hint(pkgs: List[str]) -> str:
    bullets = "\n".join(f"  • {pkg}" for pkg in pkgs)
    return dedent(
        f"""
        🚨 Missing Python packages:
        {bullets}

        Install them and rerun:

            pip install {' '.join(pkgs)}

        Then start the dashboard with:

            streamlit run {SCRIPT_NAME}
        """
    )

# -----------------------------------------------------------------------------
# Main Streamlit app – only defined/executed if dependencies are satisfied
# -----------------------------------------------------------------------------

if not missing_pkgs:
    import pandas as pd  # type: ignore
    import plotly.express as px  # type: ignore
    import streamlit as st  # type: ignore
    from pytrends.request import TrendReq  # type: ignore

    # ---------------------------- configuration -----------------------------
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
    DEFAULT_GEO = "EU"  # Europe‑wide

    # ---------------------------- data helpers -----------------------------
    @st.cache_data(ttl=3600, show_spinner=False)
    def _fetch_trends(term_or_list: List[str] | str, geo: str, timeframe: str) -> pd.Series:
        """Return weekly mean search interest for *term_or_list* or empty series."""
        kw_list = term_or_list if isinstance(term_or_list, list) else [term_or_list]
        try:
            tr = TrendReq(
                hl="en-US",
                tz=0,
                timeout=(10, 30),  # connect, read (sec)
                retries=2,
                backoff_factor=0.4,
            )
            tr.build_payload(kw_list=kw_list, geo=geo, timeframe=timeframe)
            df = tr.interest_over_time()
            return df[kw_list].mean(axis=1) if not df.empty else pd.Series(dtype=float)
        except Exception as exc:  # broad catch to keep dashboard alive
            st.warning(f"Trend fetch failed for {', '.join(kw_list)} → {exc}")
            return pd.Series(dtype=float)

    @st.cache_data(ttl=3600, show_spinner=False)
    def _collect_trends(geo: str, timeframe: str) -> pd.DataFrame:
        rows: List[Dict[str, object]] = []
        for dish, kw in DISH_KEYWORDS.items():
            ser = _fetch_trends(kw, geo, timeframe)
            rows.append(
                {
                    "Dish": dish,
                    "Mean": float(ser.mean()) if not ser.empty else 0.0,
                    "Latest": float(ser.iloc[-1]) if not ser.empty else 0.0,
                    "Series": ser,
                }
            )
        df = pd.DataFrame(rows).set_index("Dish")
        df["Rank"] = df["Mean"].rank(ascending=False, method="min")
        return df.sort_values("Rank")

    # ----------------------------- UI helpers ------------------------------
    def _render_charts(df: pd.DataFrame, timeframe: str, region: str) -> None:
        st.subheader("Popularity ranking (mean search interest)")
        st.dataframe(df[["Rank", "Mean", "Latest"]])

        st.plotly_chart(
            px.bar(
                df.reset_index(),
                x="Dish",
                y="Mean",
                title=f"Average Trends Interest – {region.upper()} ({timeframe})",
                height=420,
            ),
            use_container_width=True,
        )

        st.subheader("Trend evolution")
        selected = st.multiselect("Compare dishes", list(DISH_KEYWORDS), default=list(DISH_KEYWORDS)[:3])
        if selected:
            combined = pd.concat({d: df.loc[d, "Series"] for d in selected}, axis=1)
            st.plotly_chart(px.line(combined, title="Weekly Interest"), use_container_width=True)
            with st.expander("Raw weekly data"):
                st.dataframe(combined)

        st.caption(f"Updated {datetime.now():%Y‑%m‑%d %H:%M}")

    # ----------------------------- main app -------------------------------
    def _main() -> None:
        st.set_page_config("European Steak Popularity", layout="wide")
        st.title("European Steak‑Dish Popularity Dashboard")
        st.write("Live Google Trends comparison of ten signature beef‑steak dishes across Europe.")

        timeframe = st.sidebar.selectbox(
            "Timeframe", ("today 12-m", "today 5-y", "2015-01-01 2025-05-08"), index=1
        )
        region = st.sidebar.text_input("Geo code (ISO‑2, e.g. EU, DE, CH)", DEFAULT_GEO)
        st.sidebar.markdown("---")

        if st.button("🔄 Load / refresh data", type="primary"):
            with st.spinner("Fetching Google Trends … this may take 15‑30 s"):
                df_trends = _collect_trends(region.upper(), timeframe)
            _render_charts(df_trends, timeframe, region)
        else:
            st.info("Click the button to query Google Trends.")

else:

    def _main() -> None:  # type: ignore
        print(_install_hint(missing_pkgs))

# -----------------------------------------------------------------------------
# Execute when run as script
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    _main()

# -----------------------------------------------------------------------------
# Lightweight self‑test (skipped on Streamlit Cloud)
# -----------------------------------------------------------------------------

def _self_test() -> None:
    assert SCRIPT_NAME
    assert _try_import("sys") is sys
    assert _try_import("definitely_nonexistent_pkg_xyz") is None
    assert "pip install foo bar" in _install_hint(["foo", "bar"])

if __debug__ and not os.getenv("STREAMLIT_SERVER_PORT"):
    _self_test()
