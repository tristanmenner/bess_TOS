"""Diagnostic plots for a TOS schedule."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe; must precede pyplot import

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import Config

COLOR_PRICE = "#1f77b4"
COLOR_CHARGE = "#2c7fb8"
COLOR_DISCHARGE = "#31a354"
COLOR_SOC = "#d95f02"
COLOR_REVENUE = "#6a51a3"
COLOR_IMPORT = "#1f77b4"
COLOR_EXPORT = "#31a354"
COLOR_NET = "#756bb1"


def plot_dashboard(
    frame: pd.DataFrame,
    config: Config,
    path: Path | str,
    title: str | None = None,
) -> Path:
    """Render the four-panel diagnostic plot and save it.

    Panel 1  settlement price ($/MWh)
    Panel 2  battery SOC (%)
    Panel 3  charge/discharge power (kW, charge shown negative)
    Panel 4  cumulative net revenue ($)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ts = pd.to_datetime(frame["timestamp"])
    price = frame["price_mwh"].to_numpy()
    soc = frame["soc_percent"].to_numpy()
    charge = frame["charge_power_kw"].to_numpy()
    discharge = frame["discharge_power_kw"].to_numpy()
    revenue = frame["cumulative_revenue"].to_numpy()

    if title is None:
        title = (
            f"{config.REGION} BESS TOS - Mode {config.MODE} - "
            f"{config.START_DATE} to {config.END_DATE}"
        )

    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)

    # ---- Panel 1: price ----------------------------------------------------
    ax = axes[0]
    ax.plot(ts, price, color=COLOR_PRICE, linewidth=0.6)
    ax.axhline(0.0, color="black", linewidth=0.6, linestyle="--", alpha=0.6)
    ax.set_ylabel("Price ($/MWh)")
    ax.set_title(title, fontsize=13)
    ax.grid(alpha=0.25)
    if (price < 0).any():
        ax.fill_between(
            ts, price, 0, where=(price < 0), color="red", alpha=0.2, label="negative price"
        )
        ax.legend(loc="upper left", fontsize=8)

    # ---- Panel 2: SOC ------------------------------------------------------
    ax = axes[1]
    ax.plot(ts, soc, color=COLOR_SOC, linewidth=1.0)
    ax.axhline(config.MIN_SOC_PERCENT, color="grey", linestyle=":", linewidth=0.8)
    ax.axhline(config.MAX_SOC_PERCENT, color="grey", linestyle=":", linewidth=0.8)
    ax.set_ylabel("SOC (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25)

    # ---- Panel 3: charge/discharge power ----------------------------------
    ax = axes[2]
    ax.fill_between(ts, 0, -charge, color=COLOR_CHARGE, alpha=0.75, linewidth=0, label="charge")
    ax.fill_between(ts, 0, discharge, color=COLOR_DISCHARGE, alpha=0.75, linewidth=0, label="discharge")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Power (kW)")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=8, ncols=2)

    # ---- Panel 4: cumulative revenue --------------------------------------
    ax = axes[3]
    ax.plot(ts, revenue, color=COLOR_REVENUE, linewidth=1.1)
    ax.fill_between(ts, 0, revenue, color=COLOR_REVENUE, alpha=0.15)
    ax.set_ylabel("Cumulative revenue ($)")
    ax.set_xlabel("Time (NEM time, AEST / UTC+10; interval ending)")
    ax.grid(alpha=0.25)

    locator = mdates.AutoDateLocator(minticks=4, maxticks=12)
    formatter = mdates.ConciseDateFormatter(locator)
    axes[-1].xaxis.set_major_locator(locator)
    axes[-1].xaxis.set_major_formatter(formatter)
    for a in axes:
        a.margins(x=0)

    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_cumulative_revenue(
    frame: pd.DataFrame, config: Config, path: Path | str
) -> Path:
    """Standalone cumulative-revenue chart (the primary benchmark visual)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ts = pd.to_datetime(frame["timestamp"])
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(ts, frame["cumulative_revenue"], color=COLOR_REVENUE, linewidth=1.2)
    ax.fill_between(ts, 0, frame["cumulative_revenue"], color=COLOR_REVENUE, alpha=0.15)
    ax.set_title(
        f"{config.REGION} BESS TOS cumulative revenue (Mode {config.MODE}, "
        f"perfect hindsight)"
    )
    ax.set_ylabel("Cumulative revenue ($)")
    ax.set_xlabel("Time (NEM time)")
    ax.grid(alpha=0.25)
    ax.margins(x=0)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_cumulative_energy(
    frame: pd.DataFrame, config: Config, path: Path | str, title: str | None = None
) -> Path:
    """Cumulative grid energy in kWh - for retailer access-tier selection.

    Plots cumulative grid import ("usage"), cumulative grid export and the net
    balance, and annotates the totals so the correct monthly tier can be chosen.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ts = pd.to_datetime(frame["timestamp"])
    imported = frame["cumulative_charge_energy_kwh"].to_numpy()
    exported = frame["cumulative_discharge_energy_kwh"].to_numpy()
    net = imported - exported

    total_in = float(imported[-1])
    total_out = float(exported[-1])

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(ts, imported, color=COLOR_IMPORT, linewidth=1.4, label=f"imported (usage): {total_in:,.1f} kWh")
    ax.plot(ts, exported, color=COLOR_EXPORT, linewidth=1.4, label=f"exported: {total_out:,.1f} kWh")
    ax.plot(ts, net, color=COLOR_NET, linewidth=1.4, linestyle="--", label=f"net: {total_in - total_out:,.1f} kWh")
    ax.fill_between(ts, 0, imported, color=COLOR_IMPORT, alpha=0.10)

    if title is None:
        title = (
            f"{config.REGION} BESS TOS cumulative grid energy "
            f"({config.START_DATE} to {config.END_DATE}, Mode {config.MODE})"
        )
    ax.set_title(title)
    ax.set_ylabel("Cumulative energy (kWh)")
    ax.set_xlabel("Time (NEM time, AEST / UTC+10; interval ending)")
    ax.grid(alpha=0.25)
    ax.margins(x=0)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


__all__ = ["plot_dashboard", "plot_cumulative_revenue", "plot_cumulative_energy"]
