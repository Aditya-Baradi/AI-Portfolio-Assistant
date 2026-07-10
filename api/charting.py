"""
Chart rendering for the chat UI.

Renders an equity-curve PNG (growth of $1: portfolio vs benchmark) styled to
match the app's editorial palette: ink text, hairline grid, muted green for
the portfolio and a neutral warm gray for the benchmark reference, with
direct labels at the line ends so identity never relies on color alone.
"""
from __future__ import annotations

import uuid
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

CHARTS_DIR = Path("charts")

INK = "#201f1b"
MUTED = "#6f6a60"
HAIRLINE = "#e3ddd1"
SURFACE = "#ffffff"
PORTFOLIO_COLOR = "#177a53"  # validated: chroma/CVD/contrast pass on light surface
BENCHMARK_COLOR = "#8a8378"  # deliberate neutral reference


def equity_curve_png(
    dates,
    portfolio,
    benchmark,
    benchmark_name: str = "S&P 500",
    title: str = "Growth of $1 — portfolio vs S&P 500",
) -> str:
    """
    Save an equity-curve chart and return its URL path (e.g. /charts/abc.png).
    `dates` are ISO strings; `portfolio`/`benchmark` are cumulative growth series.
    """
    CHARTS_DIR.mkdir(exist_ok=True)
    x = pd.to_datetime(list(dates))

    fig, ax = plt.subplots(figsize=(8.2, 4.0), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ax.plot(x, portfolio, color=PORTFOLIO_COLOR, linewidth=2.0, solid_capstyle="round")
    ax.plot(x, benchmark, color=BENCHMARK_COLOR, linewidth=2.0, solid_capstyle="round")

    # Direct labels at the line ends (identity not carried by color alone).
    ax.annotate(
        f"Portfolio  {portfolio[-1]:.2f}x",
        xy=(x[-1], portfolio[-1]), xytext=(6, 0), textcoords="offset points",
        color=PORTFOLIO_COLOR, fontsize=9, fontweight="bold", va="center",
    )
    ax.annotate(
        f"{benchmark_name}  {benchmark[-1]:.2f}x",
        xy=(x[-1], benchmark[-1]), xytext=(6, 0), textcoords="offset points",
        color=MUTED, fontsize=9, va="center",
    )

    # Recessive grid and axes.
    ax.grid(axis="y", color=HAIRLINE, linewidth=0.8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(HAIRLINE)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=0)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.margins(x=0.01)

    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=12)

    # Room for the end labels on the right.
    fig.subplots_adjust(left=0.06, right=0.80, top=0.88, bottom=0.10)

    name = f"backtest_{uuid.uuid4().hex[:12]}.png"
    path = CHARTS_DIR / name
    fig.savefig(path, facecolor=SURFACE, bbox_inches=None)
    plt.close(fig)
    return f"/charts/{name}"
