#!/usr/bin/env python3
"""
Plot used_MB and cpu_percent from memlog.csv, with vertical dashed separators and labels
whenever the 'operation' value changes.

Expected columns:
timestamp,used_MB,free_MB,available_MB,cpu_percent,operation
"""
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.transforms import ScaledTranslation
import re


def fix_minute_timestamps_to_seconds(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.Series:
    """
    Convert minute-precision timestamps to 1-second resolution by assuming each row is 1 second.
    For each contiguous run of equal minute timestamp strings, add +0s, +1s, +2s, ... to that minute.

    This preserves ordering and handles >60 rows per minute by rolling over naturally.
    """
    # Parse to datetime (minute precision in the source)
    base = pd.to_datetime(df[ts_col], errors="raise")

    # Identify contiguous runs of the same base minute timestamp
    run_id = base.ne(base.shift(1)).cumsum()

    # Offset seconds within each run: 0,1,2,...
    offset = df.groupby(run_id).cumcount()

    # Corrected timestamp
    corrected = base + pd.to_timedelta(offset, unit="s")
    return corrected


def clean_operation(op: str) -> str:
    if not isinstance(op, str):
        return op

    # 1) Drop everything up to and including '--cjtype'
    if "--cjtype" in op:
        op = op.split("--cjtype", 1)[1]

    # 2) Remove '--action'
    op = op.replace("--action", "")

    # 3) Remove '--target-path <anything> '
    op = re.sub(
        r"--target-path\s+\S+",
        "",
        op
    )

    # 4) Remove '--env_vars'
    op = op.replace("--env_vars", "")

    # Normalize whitespace
    op = re.sub(r"\s+", " ", op).strip()

    return op


def main(csv_path: str = "memlog.csv") -> None:
    df = pd.read_csv(csv_path)

    required = {"timestamp", "used_MB", "cpu_percent", "operation"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Preserve file order (important for "each row is +1 second" assumption)
    df = df.reset_index(drop=True)

    # Fix timestamps first
    df["timestamp_fixed"] = fix_minute_timestamps_to_seconds(df, "timestamp")

    # Numeric columns
    df["used_MB"] = pd.to_numeric(df["used_MB"], errors="coerce")
    df["cpu_percent"] = pd.to_numeric(df["cpu_percent"], errors="coerce")
    df["operation"] = df["operation"].apply(clean_operation)

    # Optional: stop matplotlib from visually simplifying dense lines
    mpl.rcParams["path.simplify"] = False
    mpl.rcParams["agg.path.chunksize"] = 0

    # Plot
    fig, ax1 = plt.subplots(figsize=(30, 6))
    ax2 = ax1.twinx()
    mem_max = df["used_MB"].max()
    cpu_max = df["cpu_percent"].max()
    ln1 = ax1.plot(df["timestamp_fixed"], df["used_MB"], label=f"used memory (max. {round(mem_max/1000, 1)}GB)", linewidth=1.6, color='blue')
    ln2 = ax2.plot(df["timestamp_fixed"], df["cpu_percent"], label=f"cpu load (max. {round(cpu_max, 1)}%)", linewidth=1.6, color='orange')
    # Plot line for max. memory/cpu
    ax1.axhline(y=mem_max, linestyle="--", linewidth=1.0, alpha=0.4, color='blue')
    ax2.axhline(y=cpu_max, linestyle="--", linewidth=1.0, alpha=0.4, color='orange')

    #ax1.set_xlabel("timestamp")
    ax1.set_ylabel("used memory (MB)")
    ax2.set_ylabel("cpu utilization (%)")

    # Operation-change separators (based on row-to-row changes)
    change_idx = df.index[df["operation"].ne(df["operation"].shift(1))].tolist()
    if change_idx and change_idx[0] == 0:
        change_idx = change_idx[1:]

    # Put labels above the plot area
    y0, y1 = ax1.get_ylim()
    y_text = y1 * 1.02

    text_transform = (
            ax1.transData
            + ScaledTranslation(0, 10 / 72, fig.dpi_scale_trans)  # 10 pt
    )
    for i in change_idx:
        t = df.at[i, "timestamp_fixed"]
        op = str(df.at[i, "operation"])
        ax1.axvline(t, linestyle="--", linewidth=1.0, alpha=0.6, color='gray')
        ax1.text(
            t, y_text, op,
            rotation=70, va="bottom", ha="left",
            fontsize=4, clip_on=False,  transform=text_transform
        )

    # Expand ylim to make room for labels
    y0, y1 = ax1.get_ylim()
    ax1.set_ylim(y0, y1 * 1.08)

    # Grid
    ax1.yaxis.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.5)

    # Combined legend
    lines = ln1 + ln2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left")

    fig.autofmt_xdate()
    #fig.tight_layout()
    fig.subplots_adjust(top=0.6, left=0.06, right=0.94)

    csv_path = Path(csv_path)
    out_png = csv_path.with_suffix(".png")

    fig.savefig(out_png, dpi=300)
    print(f'Saved as {out_png}')
    #plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <argument>")
        sys.exit(1)

    main(sys.argv[1])
    #main("c:/!blockchains/CoinJoin/!perf/20251208/memlog.csv")
