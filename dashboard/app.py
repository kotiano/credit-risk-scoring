"""Credit Risk Analytics & Model Validation dashboard (Streamlit).

Run with ``make dashboard`` after ``make all``. All figures are computed from the
pipeline outputs; portfolio metrics are recomputed for the active filters with the same
metric functions the tests check against the SQL marts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

import loaders
import theme as T

from credit_risk.analytics import metrics as M
from credit_risk.modeling.metrics import calibration_table, decile_table
from credit_risk.modeling.scorecard import band_order
from credit_risk.segmentation.clustering import SEGMENT_ORDER

CFG = loaders.SETTINGS
RISK = CFG.risk
AS_OF = pd.Timestamp(CFG.generator.snapshot_date)
MODEL_NAMES = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "hist_gradient_boosting": "Hist Gradient Boosting",
}

INSET_LEGEND = {
    "orientation": "v",
    "x": 0.98,
    "xanchor": "right",
    "bgcolor": "rgba(252,252,251,0.92)",
    "bordercolor": T.GRID,
    "borderwidth": 1,
}

st.set_page_config(page_title="Credit Risk Analytics", page_icon="📊", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1400px;}
      h1, h2, h3 {letter-spacing: -0.01em;}
      .kpi {background:#fcfcfb; border:1px solid rgba(11,11,11,0.10); border-radius:10px;
            padding:14px 16px; height:100%;}
      .kpi .label {font-size:0.78rem; color:#52514e; text-transform:uppercase;
                   letter-spacing:0.04em;}
      .kpi .value {font-size:1.55rem; font-weight:600; color:#0b0b0b; margin-top:2px;}
      .kpi .sub {font-size:0.8rem; color:#898781; margin-top:2px;}
      .insight {border-left:3px solid #2a78d6; background:#fcfcfb; padding:10px 14px;
                margin:6px 0; border-radius:0 8px 8px 0; color:#0b0b0b;}
      .banner {font-size:0.82rem; color:#52514e;}
      div[data-testid="stTabs"] button p {font-size:0.92rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------------ helpers
def kpi(col: st.delta_generator.DeltaGenerator, label: str, value: str, sub: str = "") -> None:
    size = ' style="font-size:1.15rem;padding-top:6px"' if len(value) > 12 else ""
    col.markdown(
        f'<div class="kpi"><div class="label">{label}</div><div class="value"{size}>{value}</div>'
        f'<div class="sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def money(x: float) -> str:
    if np.isnan(x):
        return "–"
    for unit, div in (("bn", 1e9), ("m", 1e6), ("k", 1e3)):
        if abs(x) >= div:
            return f"{x / div:,.2f}{unit}"
    return f"{x:,.0f}"


def pct(x: float, digits: int = 1) -> str:
    return "–" if x is None or np.isnan(x) else f"{x:.{digits}%}"


def insight(text: str) -> None:
    st.markdown(f'<div class="insight">{text}</div>', unsafe_allow_html=True)


def chart(fig: go.Figure) -> None:
    st.plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": False})


def monthly_book(snapshot: pd.DataFrame) -> pd.DataFrame:
    """Recompute month-end PAR and book size from loan-month rows (filtered)."""
    on_book = snapshot[snapshot["loan_state"].isin(M.ON_BOOK_STATES)]
    grouped = on_book.groupby("month_end")
    out = pd.DataFrame(
        {
            "outstanding_principal": grouped["outstanding_principal"].sum(),
            "active_loans": grouped.size(),
        }
    )
    for t in RISK.par_thresholds:
        num = on_book[on_book["dpd"] >= t].groupby("month_end")["outstanding_principal"].sum()
        out[f"par{t}"] = num.reindex(out.index, fill_value=0) / out["outstanding_principal"]
    out["collections"] = snapshot.groupby("month_end")["collected_in_month"].sum()
    return out.reset_index()


# ------------------------------------------------------------------------ data
ready, message = loaders.warehouse_ready()
if not ready:
    st.error(message)
    st.stop()

loans_all = loaders.load_loans()
snapshot_all = loaders.load_monthly_snapshot()
repayments_all = loaders.load_repayments()
evaluation = loaders.load_json("evaluation")
monitoring = loaders.load_json("monitoring")
dq = loaders.load_json("data_quality")

# ------------------------------------------------------------------------ filters
with st.sidebar:
    st.markdown("### Filters")
    st.caption(
        "Applied to every portfolio metric and chart. Model validation curves use "
        "the fixed test window."
    )
    min_d = loans_all["disbursement_date"].min().date()
    max_d = loans_all["disbursement_date"].max().date()
    dates = st.date_input("Disbursement date", (min_d, max_d), min_value=min_d, max_value=max_d)
    start, end = dates if isinstance(dates, tuple) and len(dates) == 2 else (min_d, max_d)

    def multiselect(label: str, options: list[str]) -> tuple[str, ...]:
        chosen = st.multiselect(label, options, default=options)
        return tuple(chosen or options)

    filters = loaders.Filters(
        start=start,
        end=end,
        products=multiselect("Loan product", sorted(loans_all["product_type"].unique())),
        regions=multiselect("Region", sorted(loans_all["region"].unique())),
        segments=multiselect(
            "Borrower segment", [s for s in SEGMENT_ORDER if s in set(loans_all["risk_segment"])]
        ),
        bands=multiselect("Risk band (model)", band_order(CFG.modeling)),
        states=multiselect("Loan status", ["active", "overdue", "repaid", "written_off"]),
    )
    st.markdown("---")
    st.markdown(
        f'<div class="banner">Snapshot date <b>{AS_OF.date()}</b><br>'
        "Synthetic data - portfolio demonstration, not a lending system.</div>",
        unsafe_allow_html=True,
    )

loans = loaders.apply_filters(loans_all, filters)
if loans.empty:
    st.warning("No loans match the current filters.")
    st.stop()
loan_ids = set(loans["loan_id"])
snapshot = snapshot_all[snapshot_all["loan_id"].isin(loan_ids)]
repayments = repayments_all[repayments_all["loan_id"].isin(loan_ids)]
kpis = M.portfolio_kpis(loans, RISK, AS_OF)
book = monthly_book(snapshot)

st.title("Credit Risk Analytics & Model Validation")
st.markdown(
    f'<div class="banner">{len(loans):,} loans · {loans["borrower_id"].nunique():,} borrowers '
    f"· disbursed {filters.start} → {filters.end} · metrics as of {AS_OF.date()}</div>",
    unsafe_allow_html=True,
)

tabs = st.tabs(
    [
        "Executive overview",
        "Portfolio trends",
        "Delinquency",
        "Collections",
        "Borrower risk",
        "Model performance",
        "Monitoring",
        "Data quality",
    ]
)

# ======================================================================== overview
with tabs[0]:
    cols = st.columns(4)
    kpi(
        cols[0],
        "Total disbursed",
        money(kpis["total_disbursed"]),
        f"{int(kpis['total_loans']):,} loans",
    )
    kpi(
        cols[1],
        "Outstanding principal",
        money(kpis["outstanding_principal"]),
        f"{int(kpis['active_loans']):,} on-book loans",
    )
    kpi(cols[2], "Repayment rate", pct(kpis["repayment_rate"]), "repaid / due, matured loans")
    kpi(
        cols[3],
        "On-time collection rate",
        pct(kpis["on_time_collection_rate"]),
        "paid by due date / due",
    )
    st.write("")
    cols = st.columns(4)
    kpi(
        cols[0],
        "Default rate (90+ DPD)",
        pct(kpis["default_rate"]),
        "loans ≥ 90 days past due date",
    )
    kpi(cols[1], "Bad rate (30+ DPD)", pct(kpis["bad_30dpd_rate"]), "model target, matured")
    kpi(cols[2], "PAR30", pct(kpis["par30"]), "principal ≥ 30 DPD / on-book")
    kpi(cols[3], "PAR90", pct(kpis["par90"]), "principal ≥ 90 DPD / on-book")

    st.markdown("#### What the numbers say")
    recent = book.tail(4)
    if len(recent) >= 4:
        delta = recent["par30"].iloc[-1] - recent["par30"].iloc[0]
        insight(
            f"PAR30 moved <b>{delta:+.1%}</b> over the last three month-ends "
            f"(from {pct(recent['par30'].iloc[0])} to {pct(recent['par30'].iloc[-1])})."
        )
    by_product = loans.groupby("product_type").apply(M.bad_rate, include_groups=False)
    if by_product.notna().any():
        insight(
            f"Highest 30+ DPD rate by product: <b>{by_product.idxmax()}</b> "
            f"({pct(by_product.max())}); lowest: <b>{by_product.idxmin()}</b> "
            f"({pct(by_product.min())})."
        )
    seq = loans.assign(seq=np.minimum(loans["borrower_loan_sequence"], 5))
    seq_rates = seq.groupby("seq").apply(M.bad_rate, include_groups=False)
    if 1 in seq_rates and 5 in seq_rates:
        insight(
            f"First loans reach 30+ DPD at {pct(seq_rates[1])} vs {pct(seq_rates[5])} for "
            "a borrower's 5th+ loan - repayment history is the strongest risk signal."
        )
    perf = pd.DataFrame(monitoring["performance"]).set_index("window")
    insight(
        f"Champion model calibration (mean PD / actual bad rate) is "
        f"{perf.loc['train', 'calibration_ratio']:.2f} in training and "
        f"<b>{perf.loc['recent', 'calibration_ratio']:.2f}</b> on the latest matured loans."
    )

    c1, c2 = st.columns(2)
    with c1:
        fig = T.figure("PAR30 and PAR90 by month-end", yaxis_tickformat=".0%")
        T.line(fig, book["month_end"], book["par30"], "PAR30", 0, ".1%")
        T.line(fig, book["month_end"], book["par90"], "PAR90", 1, ".1%")
        chart(fig)
    with c2:
        disb = loans.groupby("disbursement_month")["loan_amount"].sum()
        fig = T.figure("Monthly disbursements", showlegend=False)
        T.bars(fig, disb.index, disb.values, "Disbursed", T.SERIES[0])
        chart(fig)

# ======================================================================== trends
with tabs[1]:
    c1, c2 = st.columns(2)
    with c1:
        monthly = loans.groupby("disbursement_month").agg(
            loans=("loan_id", "size"), repeat=("is_first_loan", lambda s: (~s).mean())
        )
        fig = T.figure("Loans disbursed per month", showlegend=False)
        T.bars(fig, monthly.index, monthly["loans"], "Loans", T.SERIES[0])
        chart(fig)
    with c2:
        due = loans.groupby(loans["due_date"].dt.to_period("M").dt.to_timestamp("M"))[
            "total_due"
        ].sum()
        coll = book.set_index("month_end")["collections"]
        frame = pd.DataFrame({"due": due, "collected": coll}).dropna().loc[:AS_OF]
        fig = T.figure("Amount falling due vs cash collected")
        T.line(fig, frame.index, frame["due"], "Falling due", 0, ",.0f")
        T.line(fig, frame.index, frame["collected"], "Collected", 1, ",.0f")
        chart(fig)
    c1, c2 = st.columns(2)
    with c1:
        fig = T.figure("Portfolio at Risk by threshold", yaxis_tickformat=".0%")
        for i, t in enumerate(RISK.par_thresholds):
            T.line(fig, book["month_end"], book[f"par{t}"], f"PAR{t}", i, ".1%")
        chart(fig)
    with c2:
        defaults = loans[loans["is_default"]].copy()
        defaults["default_month"] = (
            (defaults["due_date"] + pd.Timedelta(days=RISK.default_dpd))
            .dt.to_period("M")
            .dt.to_timestamp()
        )
        series = defaults.groupby("default_month").size().loc[:AS_OF]
        fig = T.figure("New defaults per month (loans crossing 90 DPD)", showlegend=False)
        T.bars(fig, series.index, series.values, "New defaults", T.SERIES[1])
        chart(fig)
    vint = loans[loans["is_target_observable"]].groupby("disbursement_month")["bad_30dpd"].mean()
    fig = T.figure(
        "30+ DPD rate by disbursement month (matured loans only)",
        yaxis_tickformat=".0%",
        showlegend=False,
    )
    T.line(fig, vint.index, vint.values, "30+ DPD rate", 0, ".1%")
    fig.add_vline(x=pd.Timestamp(CFG.generator.stress_start), line_dash="dot", line_color=T.MUTED)
    fig.add_annotation(
        x=pd.Timestamp(CFG.generator.stress_start),
        y=1,
        yref="paper",
        text="stress episode begins",
        showarrow=False,
        xanchor="left",
        font={"color": T.MUTED, "size": 11},
    )
    chart(fig)

# ======================================================================== delinquency
with tabs[2]:
    on_book = loans[loans["loan_state"].isin(M.ON_BOOK_STATES)]
    stats = M.delinquency_stats(loans)
    cols = st.columns(4)
    kpi(cols[0], "Delinquency rate", pct(stats["delinquency_rate"]), "on-book loans with DPD > 0")
    kpi(cols[1], "Average DPD", f"{stats['avg_dpd']:.0f}", "delinquent loans")
    kpi(cols[2], "Median DPD", f"{stats['median_dpd']:.0f}", "delinquent loans")
    kpi(cols[3], "Max DPD", f"{stats['max_dpd']:.0f}", f"write-off at {RISK.write_off_dpd}")
    st.caption(
        "Short-tenor loans that perform turn over within weeks, so the on-book stock "
        "is skewed toward delinquent loans; flow metrics (bad rate by vintage) are the "
        "better read of new-business quality."
    )
    c1, c2 = st.columns(2)
    with c1:
        dist = (
            on_book.groupby("dpd_bucket")["outstanding_principal"]
            .sum()
            .reindex(RISK.bucket_labels, fill_value=0)
        )
        colors = [T.NEUTRAL, *T.ORDINAL_5]
        fig = T.figure("Outstanding principal by DPD bucket (snapshot)", showlegend=False)
        fig.add_trace(
            go.Bar(
                x=dist.index,
                y=dist.values,
                marker={"color": colors, "cornerradius": 4},
                hovertemplate="%{y:,.0f}<extra></extra>",
            )
        )
        chart(fig)
    with c2:
        mix = (
            snapshot[snapshot["loan_state"].isin(M.ON_BOOK_STATES)]
            .pivot_table(
                index="month_end",
                columns="dpd_bucket",
                values="outstanding_principal",
                aggfunc="sum",
                fill_value=0,
            )
            .reindex(columns=RISK.bucket_labels, fill_value=0)
        )
        share = mix.div(mix.sum(axis=1), axis=0)
        fig = T.figure(
            "Delinquency bucket mix of outstanding principal",
            yaxis_tickformat=".0%",
            barmode="stack",
        )
        for label, color in zip(RISK.bucket_labels, [T.NEUTRAL, *T.ORDINAL_5], strict=True):
            fig.add_trace(
                go.Bar(
                    x=share.index,
                    y=share[label],
                    name=label,
                    marker={"color": color, "line": {"color": T.SURFACE, "width": 1}},
                    hovertemplate=f"%{{y:.1%}}<extra>{label}</extra>",
                )
            )
        chart(fig)
    # Roll rates recomputed from filtered loan-month rows.
    seq = snapshot.sort_values(["loan_id", "month_end"]).copy()
    grp = seq.groupby("loan_id")
    seq["next_state"], seq["next_bucket"] = grp["loan_state"].shift(-1), grp["dpd_bucket"].shift(-1)
    trans = seq[seq["loan_state"].isin(M.ON_BOOK_STATES) & seq["next_state"].notna()].copy()
    trans["from_state"] = trans["dpd_bucket"]
    trans["to_state"] = np.where(
        trans["next_state"].isin(["repaid", "written_off"]),
        trans["next_state"],
        trans["next_bucket"],
    )
    trans["n_loans"] = 1
    if not trans.empty:
        matrix = M.roll_rate_matrix(trans)
        order_to = [*RISK.bucket_labels, "repaid", "written_off"]
        matrix = matrix.reindex(
            index=[b for b in RISK.bucket_labels if b in matrix.index],
            columns=[c for c in order_to if c in matrix.columns],
            fill_value=0,
        )
        fig = T.figure(
            "Month-over-month roll rates (row = bucket at M, column = state at M+1)",
            height=360,
            hovermode="closest",
        )
        fig.add_trace(
            go.Heatmap(
                z=matrix.values,
                x=matrix.columns,
                y=matrix.index,
                zmin=0,
                zmax=1,
                colorscale=[[0, "#fcfcfb"], [0.5, T.BLUE[350]], [1, T.BLUE[700]]],
                text=np.vectorize(lambda v: f"{v:.0%}" if v >= 0.005 else "")(matrix.values),
                texttemplate="%{text}",
                xgap=2,
                ygap=2,
                hovertemplate="%{y} → %{x}: %{z:.1%}<extra></extra>",
                showscale=False,
            )
        )
        fig.update_yaxes(autorange="reversed")
        chart(fig)

# ======================================================================== collections
with tabs[3]:
    matured = loans[loans["due_date"] <= AS_OF]
    by_due = matured.groupby(matured["due_date"].dt.to_period("M").dt.to_timestamp()).agg(
        due=("total_due", "sum"),
        on_time=("paid_by_due_date", "sum"),
        d30=("paid_within_30d_of_due", "sum"),
        to_date=("total_paid", "sum"),
    )
    cols = st.columns(4)
    kpi(cols[0], "Expected collections", money(by_due["due"].sum()), "matured loans")
    kpi(
        cols[1],
        "Collected to date",
        money(by_due["to_date"].sum()),
        pct(by_due["to_date"].sum() / by_due["due"].sum()) + " efficiency",
    )
    kpi(
        cols[2],
        "Collection gap",
        money(by_due["due"].sum() - by_due["to_date"].sum()),
        "expected - collected",
    )
    rec = repayments[repayments["is_post_default_recovery"]]["amount"].sum()
    kpi(cols[3], "Post-default recoveries", money(rec), "paid after 90 DPD")
    c1, c2 = st.columns(2)
    with c1:
        fig = T.figure("Collection efficiency by due month", yaxis_tickformat=".0%")
        T.line(fig, by_due.index, by_due["on_time"] / by_due["due"], "By due date", 0, ".1%")
        T.line(fig, by_due.index, by_due["d30"] / by_due["due"], "Within 30 days", 1, ".1%")
        T.line(fig, by_due.index, by_due["to_date"] / by_due["due"], "To date", 2, ".1%")
        chart(fig)
        st.caption(
            "The latest due months are still inside their 30-day collection window, "
            "so their efficiency is not yet final."
        )
    with c2:
        age = (
            repayments.groupby("dpd_bucket_at_payment")["amount"]
            .sum()
            .reindex(RISK.bucket_labels, fill_value=0)
        )
        fig = T.figure("Cash collected by delinquency age at payment", showlegend=False)
        fig.add_trace(
            go.Bar(
                x=age.index,
                y=age.values,
                marker={"color": [T.NEUTRAL, *T.ORDINAL_5], "cornerradius": 4},
                hovertemplate="%{y:,.0f}<extra></extra>",
            )
        )
        chart(fig)
    c1, c2 = st.columns(2)
    with c1:
        ch = repayments.groupby("payment_channel")["amount"].sum().sort_values()
        fig = T.figure("Collections by payment channel", showlegend=False)
        T.bars(fig, ch.index, ch.values, "Collected", T.SERIES[0], orientation="h")
        chart(fig)
    with c2:
        seg = (
            loans[loans["due_date"] <= AS_OF]
            .groupby("risk_segment")
            .agg(due=("total_due", "sum"), paid=("total_paid", "sum"))
        )
        seg = (seg["paid"] / seg["due"]).reindex([s for s in SEGMENT_ORDER if s in seg.index])
        fig = T.figure(
            "Collection efficiency to date by borrower segment",
            showlegend=False,
            xaxis_tickformat=".0%",
        )
        T.bars(fig, seg.index, seg.values, "Efficiency", T.SERIES[0], fmt=".1%", orientation="h")
        chart(fig)

# ======================================================================== borrower risk
with tabs[4]:
    borrowers = loans.drop_duplicates("borrower_id")
    seg_counts = (
        borrowers["risk_segment"]
        .value_counts()
        .reindex([s for s in SEGMENT_ORDER if s in set(borrowers["risk_segment"])])
    )
    c1, c2 = st.columns(2)
    with c1:
        fig = T.figure("Borrowers by rule-based risk segment", showlegend=False)
        T.bars(fig, seg_counts.index, seg_counts.values, "Borrowers", T.SERIES[0], orientation="h")
        fig.update_yaxes(autorange="reversed")
        chart(fig)
    with c2:
        scored = loans[loans["is_target_observable"] & loans["risk_band"].notna()]
        bands = (
            scored.groupby("risk_band")
            .agg(pred=("pd", "mean"), actual=("bad_30dpd", "mean"), n=("loan_id", "size"))
            .reindex(band_order(CFG.modeling))
            .dropna()
        )
        fig = T.figure(
            "Risk bands: predicted PD vs actual 30+ DPD rate",
            yaxis_tickformat=".0%",
            barmode="group",
        )
        T.bars(fig, bands.index, bands["pred"], "Predicted PD", T.SERIES[0], ".1%")
        T.bars(fig, bands.index, bands["actual"], "Actual bad rate", T.SERIES[1], ".1%")
        chart(fig)
    st.markdown("##### Segment performance")
    seg_perf = (
        loans.groupby("risk_segment")
        .apply(
            lambda g: pd.Series(
                {
                    "borrowers": g["borrower_id"].nunique(),
                    "loans": len(g),
                    "bad_30dpd_rate": M.bad_rate(g),
                    "default_rate": M.default_rate(g),
                    "outstanding_principal": g.loc[
                        g["loan_state"].isin(M.ON_BOOK_STATES), "outstanding_principal"
                    ].sum(),
                }
            ),
            include_groups=False,
        )
        .reindex([s for s in SEGMENT_ORDER if s in set(loans["risk_segment"])])
    )
    st.dataframe(
        seg_perf.style.format(
            {
                "borrowers": "{:,.0f}",
                "loans": "{:,.0f}",
                "bad_30dpd_rate": "{:.1%}",
                "default_rate": "{:.1%}",
                "outstanding_principal": "{:,.0f}",
            }
        ),
        use_container_width=True,
    )
    st.caption(
        "Segments are assigned as of the snapshot using the rules in "
        "sql/marts/07_borrower_mart.sql; outcome rates are descriptive, not forecasts."
    )

    st.markdown("##### Vintage curves - cumulative share reaching 30+ DPD by months on book")
    v = loans[["disbursement_month", "due_date", "max_dpd_to_date"]].copy()
    v["reached"] = np.where(
        v["max_dpd_to_date"] >= RISK.target_dpd,
        v["due_date"] + pd.Timedelta(days=RISK.target_dpd),
        pd.NaT,
    )
    v["quarter"] = v["disbursement_month"].dt.to_period("Q").astype(str)
    fig = T.figure(None, height=340, yaxis_tickformat=".0%", hovermode="closest")
    quarters = sorted(v["quarter"].unique())
    for i, q in enumerate(quarters[-8:]):
        g = v[v["quarter"] == q]
        start_q = g["disbursement_month"].min()
        mobs, rates = [], []
        for k in range(0, 13):
            mob_end = (start_q + pd.DateOffset(months=k + 3)) - pd.Timedelta(days=1)
            if mob_end > AS_OF:
                break
            mobs.append(k)
            rates.append((pd.to_datetime(g["reached"]) <= mob_end).mean())
        T.line(fig, mobs, rates, q, i % len(T.SERIES), ".1%")
    fig.update_xaxes(title_text="months after the cohort quarter ends", title_font_color=T.MUTED)
    chart(fig)

    st.markdown("##### Cohort performance by product and borrower type")
    cohort = loans.assign(borrower_type=np.where(loans["is_first_loan"], "new", "repeat"))
    table = cohort.groupby(["product_type", "borrower_type"]).apply(
        lambda g: pd.Series(
            {
                "loans": len(g),
                "disbursed": g["loan_amount"].sum(),
                "repayment_rate": M.repayment_rate(g, AS_OF),
                "bad_30dpd_rate": M.bad_rate(g),
                "default_rate": M.default_rate(g),
                "par30_now": M.portfolio_at_risk(g, 30),
            }
        ),
        include_groups=False,
    )
    st.dataframe(
        table.style.format(
            {
                "loans": "{:,.0f}",
                "disbursed": "{:,.0f}",
                "repayment_rate": "{:.1%}",
                "bad_30dpd_rate": "{:.1%}",
                "default_rate": "{:.1%}",
                "par30_now": "{:.1%}",
            },
            na_rep="–",
        ),
        use_container_width=True,
    )

# ======================================================================== model
with tabs[5]:
    meta = evaluation["metadata"]
    champion = meta["champion"]
    perf = pd.DataFrame(evaluation["performance"])
    test_perf = perf[perf["split"] == "test"].set_index("model")
    cols = st.columns(4)
    kpi(cols[0], "Champion", MODEL_NAMES[champion], "parsimony rule on validation AUC")
    kpi(
        cols[1],
        "Test ROC-AUC",
        f"{test_perf.loc[champion, 'roc_auc']:.3f}",
        f"Gini {test_perf.loc[champion, 'gini']:.3f} · KS {test_perf.loc[champion, 'ks']:.3f}",
    )
    kpi(
        cols[2],
        "Test PR-AUC",
        f"{test_perf.loc[champion, 'pr_auc']:.3f}",
        f"baseline = bad rate {test_perf.loc[champion, 'bad_rate']:.1%}",
    )
    kpi(
        cols[3],
        "Test Brier",
        f"{test_perf.loc[champion, 'brier']:.3f}",
        f"calibration ratio {test_perf.loc[champion, 'calibration_ratio']:.2f}",
    )
    st.caption(
        meta["selection_rationale"] + " Curves below use the out-of-time test window "
        "(not affected by sidebar filters)."
    )

    c1, c2 = st.columns(2)
    curves = evaluation["curves_test"]
    with c1:
        fig = T.figure("ROC curve (test)", height=340, hovermode="closest")
        for i, (fam, cur) in enumerate(curves.items()):
            T.line(
                fig,
                cur["roc"]["fpr"],
                cur["roc"]["tpr"],
                f"{MODEL_NAMES[fam]} ({test_perf.loc[fam, 'roc_auc']:.3f})",
                i,
                ".3f",
            )
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                showlegend=False,
                line={"color": T.AXIS, "dash": "dot", "width": 1},
                hoverinfo="skip",
            )
        )
        fig.update_layout(legend=INSET_LEGEND | {"y": 0.04, "yanchor": "bottom"})
        fig.update_xaxes(title_text="False positive rate")
        fig.update_yaxes(title_text="True positive rate")
        chart(fig)
    with c2:
        fig = T.figure("Precision-recall curve (test)", height=340, hovermode="closest")
        for i, (fam, cur) in enumerate(curves.items()):
            T.line(
                fig,
                cur["pr"]["recall"],
                cur["pr"]["precision"],
                f"{MODEL_NAMES[fam]} ({test_perf.loc[fam, 'pr_auc']:.3f})",
                i,
                ".3f",
            )
        fig.add_hline(y=test_perf.loc[champion, "bad_rate"], line_dash="dot", line_color=T.AXIS)
        fig.update_layout(legend=INSET_LEGEND | {"y": 0.98, "yanchor": "top"})
        fig.update_xaxes(title_text="Recall")
        fig.update_yaxes(title_text="Precision")
        chart(fig)

    c1, c2 = st.columns(2)
    with c1:
        fig = T.figure(
            "Calibration: mean predicted PD vs observed bad rate",
            height=340,
            hovermode="closest",
            xaxis_tickformat=".0%",
            yaxis_tickformat=".0%",
        )
        for i, split in enumerate(["validation", "test", "recent"]):
            cal = pd.DataFrame(evaluation["calibration"][split])
            fig.add_trace(
                go.Scatter(
                    x=cal["mean_pd"],
                    y=cal["observed_rate"],
                    name=split,
                    mode="lines+markers",
                    line={"color": T.SERIES[i], "width": 2},
                    marker={"size": 8, "line": {"color": T.SURFACE, "width": 2}},
                    hovertemplate=f"PD %{{x:.1%}} → observed %{{y:.1%}}<extra>{split}</extra>",
                )
            )
        fig.add_trace(
            go.Scatter(
                x=[0, 0.6],
                y=[0, 0.6],
                mode="lines",
                showlegend=False,
                line={"color": T.AXIS, "dash": "dot", "width": 1},
                hoverinfo="skip",
            )
        )
        chart(fig)
    with c2:
        dec = pd.DataFrame(evaluation["deciles_test"])
        fig = T.figure(
            "Risk deciles (test): predicted vs actual",
            height=340,
            yaxis_tickformat=".0%",
            barmode="group",
        )
        T.bars(
            fig, dec["risk_decile"], dec["avg_predicted_pd"], "Avg predicted PD", T.SERIES[0], ".1%"
        )
        T.bars(
            fig,
            dec["risk_decile"],
            dec["actual_default_rate"],
            "Actual bad rate",
            T.SERIES[1],
            ".1%",
        )
        fig.update_xaxes(title_text="Risk decile (1 = riskiest)", dtick=1)
        chart(fig)

    c1, c2 = st.columns(2)
    with c1:
        fig = T.figure(
            "Cumulative gains (test)",
            height=320,
            hovermode="closest",
            xaxis_tickformat=".0%",
            yaxis_tickformat=".0%",
        )
        T.line(
            fig,
            [0, *dec["cum_population_pct"]],
            [0, *dec["cum_bads_captured_pct"]],
            "Model",
            0,
            ".1%",
        )
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random",
                line={"color": T.AXIS, "dash": "dot", "width": 1},
            )
        )
        fig.update_xaxes(title_text="Share of applications (riskiest first)")
        chart(fig)
    with c2:
        fig = T.figure("Lift by decile (test)", height=320, showlegend=False)
        T.bars(fig, dec["risk_decile"], dec["lift"], "Lift", T.SERIES[0], ".2f")
        fig.add_hline(y=1, line_dash="dot", line_color=T.AXIS)
        fig.update_xaxes(dtick=1, title_text="Risk decile")
        chart(fig)

    st.markdown("##### Predicted vs actual for the current filter selection")
    sel = loans[loans["is_target_observable"] & loans["pd"].notna()]
    if len(sel) >= 200 and sel["bad_30dpd"].nunique() == 2:
        dsel = decile_table(sel["bad_30dpd"].to_numpy(), sel["pd"].to_numpy())
        st.dataframe(
            dsel[
                ["risk_decile", "population", "avg_predicted_pd", "actual_default_rate", "lift"]
            ].style.format(
                {
                    "population": "{:,.0f}",
                    "avg_predicted_pd": "{:.1%}",
                    "actual_default_rate": "{:.1%}",
                    "lift": "{:.2f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
        cal = calibration_table(sel["bad_30dpd"].to_numpy(), sel["pd"].to_numpy(), bins=5)
        st.caption(
            f"{len(sel):,} matured loans in selection · mean PD {sel['pd'].mean():.1%} vs "
            f"actual {sel['bad_30dpd'].mean():.1%} · worst calibration bin gap "
            f"{(cal['observed_rate'] - cal['mean_pd']).abs().max():.1%}"
        )
    else:
        st.info("Select at least 200 matured loans with both outcomes to show deciles.")

    c1, c2 = st.columns(2)
    with c1:
        pi = pd.DataFrame(evaluation["permutation_importance"])
        fam = st.selectbox("Feature importance for", list(MODEL_NAMES), format_func=MODEL_NAMES.get)
        top = pi[pi["model"] == fam].head(12).iloc[::-1]
        fig = T.figure(
            "Permutation importance (validation ROC-AUC drop)",
            height=400,
            showlegend=False,
            hovermode="closest",
        )
        fig.add_trace(
            go.Bar(
                x=top["auc_drop_mean"],
                y=top["feature"],
                orientation="h",
                marker={"color": T.SERIES[0], "cornerradius": 4},
                error_x={
                    "type": "data",
                    "array": top["auc_drop_std"],
                    "color": T.MUTED,
                    "thickness": 1,
                },
                hovertemplate="%{x:.4f}<extra>%{y}</extra>",
            )
        )
        chart(fig)
    with c2:
        th = pd.DataFrame(evaluation["thresholds_test"])
        st.markdown("**Threshold analysis (test)** - analytical, not a credit policy")
        st.dataframe(
            th[
                [
                    "threshold",
                    "approval_rate",
                    "bad_rate_approved",
                    "precision",
                    "recall",
                    "specificity",
                    "f1",
                ]
            ].style.format("{:.3f}"),
            use_container_width=True,
            hide_index=True,
            height=360,
        )

    st.markdown("##### Individual explanation (champion)")
    ex_key = st.radio(
        "Example application",
        list(evaluation["local_explanations"]),
        horizontal=True,
        format_func=lambda k: k.replace("_", " "),
    )
    ex = evaluation["local_explanations"][ex_key]
    contrib = pd.DataFrame(ex["contributions"]).iloc[::-1]
    fig = T.figure(
        f"Loan {ex['loan_id']} - PD {ex['pd']:.1%}, actual: "
        f"{'30+ DPD' if ex['actual_outcome'] else 'repaid / < 30 DPD'}",
        height=300,
        showlegend=False,
        hovermode="closest",
    )
    fig.add_trace(
        go.Bar(
            x=contrib["pd_contribution"],
            y=contrib["feature"],
            orientation="h",
            marker={
                "color": [
                    T.SERIES[1] if v > 0 else T.SERIES[0] for v in contrib["pd_contribution"]
                ],
                "cornerradius": 4,
            },
            customdata=np.stack([contrib["value"], contrib["reference"]], axis=1),
            hovertemplate="%{y}: value %{customdata[0]} (ref %{customdata[1]})<br>"
            "PD contribution %{x:+.2%}<extra></extra>",
        )
    )
    chart(fig)
    st.caption(
        "Contribution = PD with actual value − PD with the feature set to the training "
        "median/mode. Orange raises risk, blue lowers it. No demographic attributes are "
        "model inputs."
    )

    st.markdown("##### Responsible-modeling review (test window)")
    fair = pd.DataFrame(evaluation["fairness_test"])
    st.dataframe(
        fair[
            [
                "attribute",
                "group",
                "n",
                "bad_rate",
                "mean_pd",
                "calibration_ratio",
                "roc_auc",
                "flagged_rate",
                "tpr",
                "fpr",
            ]
        ].style.format(
            {
                "n": "{:,.0f}",
                "bad_rate": "{:.1%}",
                "mean_pd": "{:.1%}",
                "calibration_ratio": "{:.2f}",
                "roc_auc": "{:.3f}",
                "flagged_rate": "{:.1%}",
                "tpr": "{:.1%}",
                "fpr": "{:.1%}",
            },
            na_rep="–",
        ),
        use_container_width=True,
        hide_index=True,
    )

# ======================================================================== monitoring
with tabs[6]:
    monthly = pd.DataFrame(monitoring["monthly"])
    monthly["month"] = pd.to_datetime(monthly["month"])
    pred = pd.DataFrame(monitoring["prediction_drift"]).set_index("window")
    feat = pd.DataFrame(monitoring["feature_drift"])
    cur = feat[feat["window"] == "current"]
    cols = st.columns(4)
    kpi(
        cols[0],
        "Score PSI (current vs train)",
        f"{pred.loc['current', 'score_psi']:.3f}",
        pred.loc["current", "status"],
    )
    kpi(
        cols[1],
        "Features with moderate drift",
        f"{(cur['status'] == 'moderate').sum()}",
        f"of {len(cur)} (PSI 0.10-0.25)",
    )
    kpi(
        cols[2],
        "Features with significant drift",
        f"{(cur['status'] == 'significant').sum()}",
        "PSI > 0.25",
    )
    perf = pd.DataFrame(monitoring["performance"]).set_index("window")
    kpi(
        cols[3],
        "Recent calibration ratio",
        f"{perf.loc['recent', 'calibration_ratio']:.2f}",
        "mean PD / actual, matured",
    )
    c1, c2 = st.columns(2)
    with c1:
        fig = T.figure("Predicted PD vs actual bad rate by decision month", yaxis_tickformat=".0%")
        T.line(fig, monthly["month"], monthly["mean_pd"], "Mean predicted PD", 0, ".1%")
        T.line(
            fig, monthly["month"], monthly["actual_bad_rate"], "Actual bad rate (matured)", 1, ".1%"
        )
        chart(fig)
    with c2:
        fig = T.figure("ROC-AUC by decision month (matured loans)")
        T.line(fig, monthly["month"], monthly["roc_auc"], "ROC-AUC", 0, ".3f")
        chart(fig)
    c1, c2 = st.columns(2)
    with c1:
        fig = T.figure("Score PSI vs training window, by month", showlegend=False)
        T.bars(fig, monthly["month"], monthly["score_psi_vs_train"], "PSI", T.SERIES[0], ".3f")
        fig.add_hline(
            y=CFG.monitoring.psi_moderate,
            line_dash="dot",
            line_color=T.MUTED,
            annotation_text="moderate",
            annotation_font_color=T.MUTED,
        )
        chart(fig)
    with c2:
        top = cur.sort_values("psi", ascending=False).head(12).iloc[::-1]
        fig = T.figure("Top feature PSI (current vs train)", showlegend=False, hovermode="closest")
        T.bars(fig, top["feature"], top["psi"], "PSI", T.SERIES[0], ".3f", orientation="h")
        chart(fig)
    st.caption(
        "Performance metrics require matured outcomes (due date + 30 days); the most "
        "recent months show partial maturity. Stable PSI with a falling calibration ratio "
        "means the risk level moved without the inputs shifting: recalibrate, don't "
        "rebuild."
    )
    st.dataframe(
        perf.style.format("{:.3f}", subset=[c for c in perf.columns if c != "matured_loans"]),
        use_container_width=True,
    )

# ======================================================================== data quality
with tabs[7]:
    summ = dq["summary"]
    checks = pd.DataFrame(dq["checks"])
    status = summ["overall_status"]
    cols = st.columns(4)
    kpi(
        cols[0],
        "Overall status",
        f"{T.STATUS_ICON[status]} {status}",
        f"clean layer: {summ['clean_layer_status']}",
    )
    kpi(
        cols[1],
        "Checks passed outright",
        pct(summ["check_pass_rate"], 0),
        f"{summ['checks_total']} checks",
    )
    kpi(cols[2], "Record retention", pct(summ["record_retention_rate"], 2), "clean / raw rows")
    kpi(
        cols[3],
        "Quarantined records",
        f"{sum(q['records'] for q in dq['quarantine']):,}",
        "kept with reason codes",
    )
    profile = pd.DataFrame(dq["profile"])
    st.markdown("##### Dataset profile")
    st.dataframe(
        profile.style.format(
            {
                "missing_cell_pct": "{:.3%}",
                "duplicate_pct": "{:.3%}",
                "raw_rows": "{:,}",
                "clean_rows": "{:,}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.markdown("##### Validation checks")
    checks["status"] = checks["status"].map(lambda s: f"{T.STATUS_ICON[s]} {s}")
    st.dataframe(
        checks[
            [
                "layer",
                "dataset",
                "name",
                "category",
                "failing_records",
                "failure_rate",
                "status",
                "treatment",
            ]
        ].style.format({"failure_rate": "{:.4%}", "failing_records": "{:,}"}),
        use_container_width=True,
        hide_index=True,
        height=420,
    )
    miss = pd.DataFrame(dq["missingness"])
    miss = miss[miss["missing"] > 0]
    if not miss.empty:
        fig = T.figure(
            "Missing values in raw extracts", showlegend=False, height=220, xaxis_tickformat=".1%"
        )
        T.bars(
            fig,
            miss["dataset"] + "." + miss["column"],
            miss["missing_pct"],
            "Missing",
            T.SERIES[0],
            ".2%",
            orientation="h",
        )
        chart(fig)
