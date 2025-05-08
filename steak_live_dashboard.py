# steak_live_dashboard.py — European Steak Popularity Dashboard
# =============================================================================
# Streamlit app that compares multiple popularity signals for ten iconic
# beef‑steak dishes across Europe.
#   • Google Trends           – weekly search interest (country‑level)
#   • Wikipedia Pageviews     – daily article traffic (global)
#   • Instagram hashtag count – placeholder (returns 0 until real token added)
#
# Runs locally (`streamlit run steak_live_dashboard.py`) and on Streamlit Cloud.
# Python ≥ 3.7 compatible.
# =============================================================================
"""Quick start (local)
$ pip install streamlit pandas pytrends plotly requests
$ streamlit run steak_live_dashboard.py

requirements.txt for Streamlit Cloud:
    streamlit>=1.35
    pandas
    pytrends
    plotly
    requests
"""

from __future__ import annotations

import os
import sys
import time
import types
from datetime import date, datetime
from importlib import import_module, util as import_util
from pathlib import Path
from textwrap import dedent
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Path helpers (cover environments where __file__ may be undefined)
# ---------------------------------------------------------------------------

SCRIPT_PATH: Path = (
    Path(sys.argv[0]).resolve() if sys.argv and sys.argv[0] else Path.cwd() / "steak_live_dashboard.py"
)
SCRIPT_NAME: str = SCRIPT_PATH.name

# ---------------------------------------------------------------------------
# Dependency detection — fail gracefully with a hint if missing
# ---------------------------------------------------------------------------

REQUIRED: Dict[str, str] = {
    "streamlit": "streamlit",
    "pandas": "pandas",
    "plotly": "plotly",
    "pytrends": "pytrends",
    "requests": "requests",
}


def _try_import(name: str) -> Optional[types.ModuleType]:
    try:
        return import_module(name) if import_util.find_spec(name) else None
    except ImportError:
        return None

missing = [pip for mod, pip in REQUIRED.items() if _try_import(mod) is None]

# ---------------------------------------------------------------------------
# Install‑hint message (printed when deps are absent)
# ---------------------------------------------------------------------------

def _install_hint(pkgs: List[str]) -> str:
    bullets = "\n".join(f"  • {p}" for p in pkgs)
    return dedent(
        f"""
        🚨 Required packages are missing:
        {bullets}

        Install them, then run:
            pip install {' '.join(pkgs)}

        Start the dashboard with:
            streamlit run {SCRIPT_NAME}
        """
    )

# ---------------------------------------------------------------------------
# Streamlit dashboard (executes only if dependencies exist)
# ---------------------------------------------------------------------------

if not missing:
    import requests
    import pandas as pd  # type: ignore
    import plotly.express as px  # type: ignore
    import streamlit as st  # type: ignore
    from pytrends.request import TrendReq  # type: ignore

    # ---------------- Configuration ----------------
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

    WIKI_TITLES: Dict[str, str] = {
        "Steak Frites": "Steak_frites",
        "Bistecca alla Fiorentina": "Bistecca_alla_fiorentina",
        "Steak Tartare": "Steak_tartare",
        "Beef Wellington": "Beef_Wellington",
        "Entrecôte Café de Paris": "Café_de_Paris_sauce",
        "Steak au Poivre": "Steak_au_poivre",
        "Tagliata di Manzo": "Tagliata",
        "Zwiebelrostbraten": "Zwiebelrostbraten",
        "Txuleton (Chuleta)": "Txuleton",
        "Picanha (Churrasco)": "Picanha",
    }

    DATA_SOURCES = ["Google Trends", "Wikipedia Pageviews", "Instagram (demo)"]
    DEFAULT_TIMEFRAME = "today 12-m"
    DEFAULT_GEO = "EU"
    REQUEST_DELAY_SEC = 1.2  # politeness delay between API calls

    # ---------------- Fetcher helpers ----------------
    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch_trends(keywords: List[str] | str, geo: str, timeframe: str) -> pd.Series:
        """Weekly Google Trends interest (Series) or empty series if blocked."""
        kw = keywords if isinstance(keywords, list) else [keywords]
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124 Safari/537.36"
            )
        }
        try:
            tr = TrendReq(hl="en-US", tz=0, timeout=(10, 30), requests_args={"headers": headers})
            tr.build_payload(kw_list=kw, geo=geo, timeframe=timeframe)
            df = tr.interest_over_time()
            return df[kw].mean(axis=1) if not df.empty else pd.Series(dtype=float)
        except Exception as exc:
            st.warning(f"Google Trends blocked ({', '.join(kw)}): {exc}")
            return pd.Series(dtype=float)

    @st.cache_data(ttl=86_400, show_spinner=False)
    def fetch_wiki_pageviews(page: str, days: int = 365) -> pd.Series:
        end = date.today().strftime("%Y%m%d")
        start = (date.today() - pd.Timedelta(days=days)).strftime("%Y%m%d")
        url = (
            "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
            f"en.wikipedia/all-access/user/{page}/daily/{start}/{end}"
        )
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            items = r.json()["items"]
            s = pd.Series({i["timestamp"][:8]: i["views"] for i in items})
            s.index = pd.to_datetime(s.index)
            return s
        except Exception as exc:
            st.warning(f"Wikipedia request failed for {page}: {exc}")
            return pd.Series(dtype=float)

    @st.cache_data(ttl=10_800, show_spinner=False)
    def fetch_instagram_hashtag(tag: str) -> int:
        # Placeholder until a real IG API or scraper is added
        return 0

    # ---------------- Metric router ----------------
    def get_metric(dish: str, source: str, geo: str, timeframe: str):
        if source == "Google Trends":
            return fetch_trends(DISH_KEYWORDS[dish], geo, timeframe)
        if source == "Wikipedia Pageviews":
            page = WIKI_TITLES.get(dish, dish.replace(" ", "_"))
            return fetch_wiki_pageviews(page)
        if source.startswith("Instagram"):
            tag = DISH_KEYWORDS[dish][0].replace(" ", "")
            return fetch_instagram_hashtag(tag)
        return pd.Series(dtype=float)

    def collect_data(source: str, geo: str, timeframe: str) -> pd.DataFrame:
        rows = []
        for dish in DISH_KEYWORDS:
            metric = get_metric(dish, source, geo, timeframe)
            time.sleep(REQUEST_DELAY_SEC)  # politeness delay
            if isinstance(metric, pd.Series):
                mean_val = float(metric.mean()) if not metric.empty else 0.0
                latest_val = float(metric.iloc[-1]) if not metric.empty else 0.0
            else:
                mean_val = latest_val = float(metric)
                metric = pd.Series(dtype=float)
            rows.append({
                "Dish": dish,
                "Mean": mean_val,
                "Latest": latest_val,
                "Series": metric,
            })
        df = pd.DataFrame(rows).set_index("Dish")
        df["Rank"] = df["Mean"].rank(ascending=False, method="min")
        return df.sort_values("Rank")

    # ---------------- UI helpers ----------------
    def render_charts(df: pd.DataFrame, source: str, timeframe: str, geo: str):
        st.subheader("Popularity ranking (mean value)")
        st.dataframe(df[["Rank", "Mean", "Latest"]])
        st.plotly_chart(
            px.bar(df.reset_index(), x="Dish", y="Mean",
                   title=f"Average – {source} ({timeframe if 'Google' in source else 'last year'})",
                   height=420),
            use_container_width=True,
        )

        st.subheader("Metric evolution")
        chosen = st.multiselect("Compare dishes",
                                list(DISH_KEYWORDS.keys()),
                                default=list(DISH_KEYWORDS.keys())[:3])
        if chosen:
            combo = pd.concat({d: df.loc[d, "Series"] for d in chosen}, axis=1)
            if not combo.empty:
                st.plotly_chart(px.line(combo, title="Time‑series comparison"), use_container_width=True)
                with st.expander("Raw data"):
                    st.dataframe(combo)
            else:
                st.info("Selected source does not provide time‑series data.")

        st.caption(f"Updated {datetime.now():%Y-%m-%d %H:%M}")

    # ---------------- Main Streamlit app ----------------
    def main() -> None:
        st.set_page_config("European Steak Popularity", layout="wide")
        st.title("European Steak‑Dish Popularity Dashboard")
        st.write("Compare Google search interest, Wikipedia pageviews and (demo) Instagram counts for classic beef‑steak dishes.")

        source = st.sidebar.selectbox("Data source", DATA_SOURCES, index=0)
        timeframe = st.sidebar.selectbox("Timeframe (Google Trends)",
                                         ("today 12-m", "today 5-y", "2015-01-01 2025-05-08"), index=0)
        geo = st.sidebar.text_input("Geo code (Google Trends)", DEFAULT_GEO)
        st.sidebar.markdown("---")

        if st.button("🔄 Load / refresh data", type="primary"):
            with st.spinner("Fetching data – please wait …"):
                df_metrics = collect_data(source, geo.upper(), timeframe)
            render_charts(df_metrics, source, timeframe, geo)
        else:
            st.info("Click the button to pull data. First run may take up to a minute due to API delays.")

else:
    def main() -> None:  # type: ignore
        print(_install_hint(missing))

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# Basic self‑test (skipped on Streamlit Cloud)
# ---------------------------------------------------------------------------

def _self_test():
    assert _install_hint(["foo", "bar"]).startswith("\n        🚨")
    assert _try_import("sys") is sys
    assert "/" not in SCRIPT_NAME  # simple sanity

if __debug__ and not os.getenv("STREAMLIT_SERVER_PORT"):
    _self_test()
