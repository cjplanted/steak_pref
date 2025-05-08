# steak_live_dashboard.py — European Steak Popularity Dashboard
# =============================================================================
# Streamlit app comparing multiple popularity signals for ten iconic beef‑steak
# dishes across Europe.
#   • Google Trends           – weekly search interest (country‑level)
#   • Wikipedia Pageviews     – daily article traffic (global)
#   • Instagram hashtag count – placeholder (0 until proper token added)
#
# Works with Python ≥ 3.7, locally or on Streamlit Cloud.
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
# Robust path helpers (cover environments without __file__)
# ---------------------------------------------------------------------------
SCRIPT_PATH = Path(sys.argv[0]).resolve() if sys.argv and sys.argv[0] else Path.cwd() / "steak_live_dashboard.py"
SCRIPT_NAME = SCRIPT_PATH.name

# ---------------------------------------------------------------------------
# Dependency check — show friendly hint if anything is missing
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
# Install‑hint
# ---------------------------------------------------------------------------

def _install_hint(pkgs: List[str]) -> str:
    bullets = "\n".join(f"  • {p}" for p in pkgs)
    return dedent(
        f"""
        🚨 Required packages are missing:
        {bullets}

        Install them and rerun:
            pip install {' '.join(pkgs)}

        Then launch the dashboard:
            streamlit run {SCRIPT_NAME}
        """
    )

# ---------------------------------------------------------------------------
# Main dashboard (only if deps are present)
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
    REQUEST_DELAY = 1.2  # seconds between API calls to stay polite

    # ---------------- Fetchers ----------------
    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch_trends(keywords: List[str] | str, geo: str, timeframe: str) -> pd.Series:
        kw = keywords if isinstance(keywords, list) else [keywords]
        headers = {"User-Agent": "Mozilla/5.0 Chrome/124"}
        try:
            tr = TrendReq(hl="en-US", tz=0, timeout=(10, 30), requests_args={"headers": headers})
            tr.build_payload(kw_list=kw, geo=geo, timeframe=timeframe)
            df = tr.interest_over_time()
            return df[kw].mean(axis=1) if not df.empty else pd.Series(dtype=float)
        except Exception as exc:
            st.warning(f"Google Trends error for {', '.join(kw)} → {exc}")
            return pd.Series(dtype=float)

    @st.cache_data(ttl=86400, show_spinner=False)
    def fetch_wiki_pageviews(page: str, days: int = 365) -> pd.Series:
        end = date.today().strftime("%Y%m%d")
        start = (date.today() - pd.Timedelta(days=days)).strftime("%Y%m%d")
        url = (
            "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
            f"en.wikipedia/all-access/user/{page}/daily/{start}/{end}"
        )
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            items = resp.json()["items"]
            series = pd.Series({i["timestamp"][:8]: i["views"] for i in items})
            series.index = pd.to_datetime(series.index)
            return series
        except Exception as exc:
            st.warning(f"Wikipedia error for {page} → {exc}")
            return pd.Series(dtype=float)

    @st.cache_data(ttl=10800, show_spinner=False)
    def fetch_instagram_hashtag(tag: str) -> int:
        return 0  # placeholder until real IG integration

    # ---------------- Data collection ----------------
    def get_metric(dish: str, source: str, geo: str, timeframe: str):
        if source == "Google Trends":
            return fetch_trends(DISH_KEYWORDS[dish], geo, timeframe)
        if source == "Wikipedia Pageviews":
            return fetch_wiki_pageviews(WIKI_TITLES[dish])
        if source.startswith("Instagram"):
            return fetch_instagram_hashtag(DISH_KEYWORDS[dish][0].replace(" ", ""))
        return pd.Series(dtype=float)

    def collect_data(source: str, geo: str, timeframe: str) -> pd.DataFrame:
        rows = []
        for dish in DISH_KEYWORDS:
            metric = get_metric(dish, source, geo, timeframe)
            time.sleep(REQUEST_DELAY)
            if isinstance(metric, pd.Series):
                mean_val = metric.mean() if not metric.empty else 0.0
                latest_val = metric.iloc[-1] if not metric.empty else 0.0
            else:
                mean_val = latest_val = float(metric)
                metric = pd.Series(dtype=float)
            rows.append({"Dish": dish, "Mean": mean_val, "Latest": latest_val, "Series": metric})
        df = pd.DataFrame(rows).set_index("Dish")
        df["Rank"] = df["Mean"].rank(ascending=False, method="min")
        return df.sort_values("Rank")

    # ---------------- UI helpers ----------------
    def render_charts(df: pd.DataFrame, source: str):
        st.subheader("Popularity ranking (mean value)")
        st.dataframe(df[["Rank", "Mean", "Latest"]])
        st.plotly_chart(
            px.bar(df.reset_index(), x="Dish", y="Mean", title=f"Average – {source}", height=420),
            use_container_width=True,
        )
        st.subheader("Metric evolution")
        chosen = st.multiselect("Compare dishes", list(DISH_KEYWORDS.keys()), default=list(DISH_KEYWORDS.keys())[:3])
        if chosen:
            combo = pd.concat({d: df.loc[d, "Series"] for d in chosen}, axis=1)
            if not combo.empty:
                st.plotly_chart(px.line(combo, title="Time‑series comparison"), use_container_width=True)
                with st.expander("Raw data"):
                    st.dataframe(combo)
            else:
                st.info("Selected source has no time‑series data.")
        st.caption(f"Updated {datetime.now():%Y‑%m‑%d %H:%M}")

    # ---------------- Main app ----------------
    def main() -> None:
        st.set_page_config("European Steak Popularity", layout="wide")
        st.title("European Steak‑Dish Popularity Dashboard")
        st.write("Compare Google search, Wikipedia interest, and Instagram tags (demo) for classic steak dishes.")

        source = st.sidebar.selectbox("Data source", DATA_SOURCES, index=0)
        timeframe = st.sidebar.selectbox("Timeframe (Google Trends)",
                                         ("today 12-m", "today 5-y", "2015-01-01 2025-05-08"), index=0)
        geo = st.sidebar.text_input("Geo code (Google Trends)", DEFAULT_GEO)
        st.sidebar.markdown("---")

        if st.button("🔄 Load / refresh data", type="primary"):
            with st.spinner("Fetching data…"):
                df = collect_data(source, geo.upper(), timeframe)
            render_charts(df, source)
        else:
            st.info("Press the button to fetch data. First run may take a minute.")

else:
    def main() -> None:  # type: ignore
        print(_install_hint(missing))

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# Lightweight self‑test (skipped in Streamlit Cloud)
# ---------------------------------------------------------------------------

def _self_test():
    assert "🚨" in _install_hint(["foo", "bar"]) and "foo" in _install_hint(["foo", "bar"])
    assert _try_import("sys") is sys

if __debug__ and not os.getenv("STREAMLIT_SERVER_PORT"):
    _self_test()
