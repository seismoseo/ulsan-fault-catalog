"""Annual epicenter + density plots 2010-2024 from the de-quarried PhaseNet+ catalog.

A 5x3 panel grid per figure (one panel per year), styled like the existing notebook's
subregion map cell. Catalog: catalog_phasenet_plus_2010_2024_blastclean.csv (QC + blast-cleaned).

Two figures saved to the same directory:
  annual_locations_2010_2024.png   — events as depth-coloured dots per year
  annual_density_2010_2024.png     — log-10 hexbin density per year

A box outlining the 2016 Gyeongju M5.4 + M5.8 + aftershock epicentral region is overlaid
on every panel (35.72-35.82 N, 129.15-129.25 E) so you can see how each year's seismicity
relates to that zone.

Reusable: edit the CATALOG / GYEONGJU_BOX / REGION constants and rerun.
"""
from __future__ import annotations

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle


# --- inputs ---------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
CATALOG = os.path.join(HERE, "catalog_phasenet_plus_2010_2024_blastclean.csv")

# Region: extended westward of the Ulsan-Fault subregion (129.25-129.55, 35.6-35.9) so the
# 2016 Gyeongju aftershock cluster at lon ~129.19 is also in-frame on every panel.
REGION = [129.05, 129.60, 35.55, 35.95]      # [lon0, lon1, lat0, lat1]

# 2016 Gyeongju epicentral zone: circumscribes the M5.4 foreshock (Sept 12 10:44 UTC,
# 35.77/129.19), the M5.8 mainshock (~50 min later, same coords), and the dense aftershock
# cloud (~600 events within 5 km). Adjust if you want a different framing.
GYEONGJU_BOX = [129.15, 129.25, 35.72, 35.82]  # [lon0, lon1, lat0, lat1]

# Ulsan Fault subregion (for reference, matches the existing notebook's `subregion`)
ULSAN_FAULT_BOX = [129.25, 129.55, 35.6, 35.9]

DMAX_KM = 30.0       # depth color clip for the location plot
HEXBIN_GRIDSIZE = 50 # density plot resolution


# --- plot helpers ---------------------------------------------------------
def _frame(ax, year: int, n: int) -> None:
    """Common framing for one panel: region limits, Gyeongju box overlay, year title."""
    ax.set_xlim(REGION[0], REGION[1])
    ax.set_ylim(REGION[2], REGION[3])
    ax.set_aspect("equal", adjustable="box")
    # Gyeongju 2016 box overlay (filled-edge rectangle, no fill)
    gb = GYEONGJU_BOX
    ax.add_patch(Rectangle((gb[0], gb[2]), gb[1] - gb[0], gb[3] - gb[2],
                            fill=False, edgecolor="red", linewidth=1.6, linestyle="-",
                            zorder=5))
    # Ulsan Fault subregion overlay (thin dashed) -- to see the contrast with the Gyeongju box
    ub = ULSAN_FAULT_BOX
    ax.add_patch(Rectangle((ub[0], ub[2]), ub[1] - ub[0], ub[3] - ub[2],
                            fill=False, edgecolor="black", linewidth=0.8, linestyle="--",
                            zorder=5))
    ax.set_title(f"{year}  ({n} events)", fontsize=10)
    ax.tick_params(labelsize=7)


def _shared_colorbar(fig, mappable, axes, label: str) -> None:
    """Single colorbar on the right of the grid (avoids per-panel duplication)."""
    cbar_ax = fig.add_axes([0.92, 0.15, 0.012, 0.70])
    cb = fig.colorbar(mappable, cax=cbar_ax)
    cb.set_label(label, fontsize=9)
    cb.ax.tick_params(labelsize=8)


# --- main figures ---------------------------------------------------------
def plot_annual_locations(cat: pd.DataFrame, years: range, out_path: str) -> None:
    """5x3 grid of epicenter maps, events coloured by depth, Gyeongju 2016 box overlay."""
    norm = mpl.colors.Normalize(vmin=0.0, vmax=DMAX_KM)
    cmap = plt.get_cmap("viridis_r")
    fig, axes = plt.subplots(3, 5, figsize=(15.5, 9.5), dpi=130, constrained_layout=False)
    for ax, year in zip(axes.ravel(), years):
        sub = cat[cat.year == year]
        ax.scatter(sub.lon, sub.lat, s=8, c=sub.depth, cmap=cmap, norm=norm,
                   edgecolor="k", linewidth=0.2, alpha=0.85, zorder=3)
        _frame(ax, year, len(sub))
    # bottom-row x-labels + left-column y-labels only
    for ax in axes[-1, :]:
        ax.set_xlabel("Longitude (°E)", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("Latitude (°N)", fontsize=9)
    fig.suptitle("Ulsan-Fault region annual epicenters 2010–2024  "
                 "(catalog_phasenet_plus_2010_2024_blastclean.csv)  "
                 "— red box = 2016 Gyeongju epicentral zone, dashed = Ulsan-Fault subregion",
                 fontsize=12, y=0.995)
    fig.subplots_adjust(left=0.04, right=0.91, top=0.93, bottom=0.06,
                       hspace=0.30, wspace=0.18)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    _shared_colorbar(fig, sm, axes, "Depth (km)")
    fig.savefig(out_path, bbox_inches="tight")
    print(f"  wrote {out_path}")
    plt.close(fig)


def plot_annual_density(cat: pd.DataFrame, years: range, out_path: str) -> None:
    """5x3 grid of hexbin density maps (log10 counts), Gyeongju 2016 box overlay.

    Shared color scale across all panels (computed from the global max log10-count) so
    panels are directly comparable between years."""
    # pre-compute per-panel hexbin to fix a common color range
    pcs = []
    fig, axes = plt.subplots(3, 5, figsize=(15.5, 9.5), dpi=130, constrained_layout=False)
    for ax, year in zip(axes.ravel(), years):
        sub = cat[cat.year == year]
        if len(sub) == 0:
            ax.text(0.5, 0.5, "no events", ha="center", va="center",
                    transform=ax.transAxes, color="0.5", fontsize=10)
            _frame(ax, year, 0)
            pcs.append(None); continue
        hb = ax.hexbin(sub.lon, sub.lat, gridsize=HEXBIN_GRIDSIZE,
                       extent=(REGION[0], REGION[1], REGION[2], REGION[3]),
                       bins="log", cmap="magma", mincnt=1, zorder=2)
        pcs.append(hb)
        _frame(ax, year, len(sub))
    # Equalize color scale across all panels
    vmax = max(h.get_array().max() for h in pcs if h is not None)
    for h in pcs:
        if h is not None:
            h.set_clim(vmin=1, vmax=vmax)
    # bottom-row x-labels + left-column y-labels only
    for ax in axes[-1, :]:
        ax.set_xlabel("Longitude (°E)", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("Latitude (°N)", fontsize=9)
    fig.suptitle("Ulsan-Fault region annual density 2010–2024  "
                 "(log₁₀ event count per cell, shared scale)  "
                 "— red box = 2016 Gyeongju zone, dashed = Ulsan-Fault subregion",
                 fontsize=12, y=0.995)
    fig.subplots_adjust(left=0.04, right=0.91, top=0.93, bottom=0.06,
                        hspace=0.30, wspace=0.18)
    # shared colorbar from the first non-empty hexbin (they're already clim-equalized)
    first = next((h for h in pcs if h is not None), None)
    if first is not None:
        _shared_colorbar(fig, first, axes, "log$_{10}$ count")
    fig.savefig(out_path, bbox_inches="tight")
    print(f"  wrote {out_path}")
    plt.close(fig)


def main() -> None:
    cat = pd.read_csv(CATALOG, parse_dates=["time"])
    # `year` column is already in the CSV; use the catalog's existing year field
    if "year" not in cat.columns:
        cat = cat.assign(year=cat.time.dt.year)
    cat = cat[cat.lon.between(REGION[0], REGION[1])
              & cat.lat.between(REGION[2], REGION[3])].copy()
    print(f"catalog: {CATALOG}")
    print(f"  {len(cat)} events in plot region ({REGION[0]}-{REGION[1]} E, "
          f"{REGION[2]}-{REGION[3]} N)")
    years = range(2010, 2025)
    plot_annual_locations(cat, years, os.path.join(HERE, "annual_locations_2010_2024.png"))
    plot_annual_density(cat, years, os.path.join(HERE, "annual_density_2010_2024.png"))


if __name__ == "__main__":
    main()
