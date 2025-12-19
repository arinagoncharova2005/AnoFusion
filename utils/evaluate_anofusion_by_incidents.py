"""Evaluate AnoFusion predictions with the incident metrics from final_baseline_modles.ipynb.

This script mirrors the notebook helpers (make_alert_series, eval_incident_metrics_robust)
and applies them to the mobile service labels/predictions used in utils/evaluate_metrics.py.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score


def read_csv_dataset(file_path: str | Path, sep: str = ",") -> pd.DataFrame:
    return pd.read_csv(file_path, sep=sep)


def convert_timestamps_to_datetime(
    df: pd.DataFrame, timestamp_column_name: str = "timestamp", timestamp_unit: str = "s"
) -> pd.DataFrame:
    df = df.copy()
    df["datetime"] = pd.to_datetime(df[timestamp_column_name], unit=timestamp_unit)
    return df


def get_data(
    file_path: str | Path,
    sep: str = ",",
    timestamp_column_name: str = "timestamp",
    timestamp_unit: str = "s",
) -> pd.DataFrame:
    df = read_csv_dataset(file_path, sep=sep)
    df = convert_timestamps_to_datetime(df, timestamp_column_name, timestamp_unit)
    return df


def extract_prefailure_intervals(
    model_preds: pd.Series, min_len: str | None = None, merge_gap: str | None = None
):
    """Return contiguous runs of True predictions as {start, end} dicts."""
    model_preds_prepared = pd.Series(model_preds).fillna(False).astype(bool).sort_index()

    runs, in_run, start, prev = [], False, None, None
    for ts, val in model_preds_prepared.items():
        if val and not in_run:
            in_run, start = True, ts
        elif not val and in_run:
            runs.append({"start": start, "end": prev})
            in_run, start = False, None
        prev = ts

    if in_run:
        runs.append({"start": start, "end": prev})

    return runs


def make_alert_series(
    y_score_or_bin: pd.Series,
    thr: float = 0.5,
    smooth: str | None = "3min",
    min_len: str | None = "2min",
    merge_gap: str | None = "3min",
    cooldown: str | None = "10min",
) -> tuple[pd.Series, list[dict]]:
    """Convert raw model outputs into a binary alert series and list of alert episodes."""
    model_predictions = pd.Series(y_score_or_bin).copy()

    if model_predictions.dtype not in (int, bool, np.int64, np.int32):
        model_predictions = (model_predictions >= float(thr)).astype(int)

    model_predictions = model_predictions.fillna(0).astype(int).sort_index()
    runs = extract_prefailure_intervals(model_predictions, min_len=min_len, merge_gap=merge_gap)

    alert_series = pd.Series(0, index=model_predictions.index, dtype=int)
    for run in runs:
        alert_series[(alert_series.index >= run["start"]) & (alert_series.index <= run["end"])] = 1

    return alert_series, runs


def eval_incident_metrics_robust(
    y_score_or_bin: pd.Series,
    incidents: pd.DataFrame,
    pre_minutes: int,
    thr: float = 0.5,
    smooth: str = "3min",
    min_len: str = "2min",
    merge_gap: str = "3min",
    cooldown: str = "10min",
    y_true: pd.Series | None = None,
):
    """
    Incident-level metrics from the notebook (precision/recall/F1 on incidents, MTTD, FP time).
    """
    alert_bin, alert_runs = make_alert_series(
        y_score_or_bin,
        thr=thr,
        smooth=smooth,
        min_len=min_len,
        merge_gap=merge_gap,
        cooldown=cooldown,
    )

    alert_bin = alert_bin.sort_index()
    idx = alert_bin.index

    inc_times = pd.to_datetime(incidents["datetime"]).sort_values()
    start, end = idx.min(), idx.max()
    pre = pd.Timedelta(minutes=pre_minutes)
    windows = [(t - pre, t) for t in inc_times]

    in_window = pd.Series(False, index=idx, dtype=bool)
    for ws, we in windows:
        in_window |= (idx >= ws) & (idx <= we)

    ts_series = pd.Series(idx, index=idx)
    deltas_sec = (ts_series.shift(-1) - ts_series).dt.total_seconds()
    deltas_sec = deltas_sec.fillna(0.0)
    deltas_min = deltas_sec / 60.0

    mask_alert = alert_bin.astype(bool)
    mask_fp_points = mask_alert & (~in_window)

    fp_minutes = float((deltas_min * mask_fp_points).sum())
    total_minutes = float(deltas_min.sum())
    alert_minutes = float((deltas_min * mask_alert).sum())

    non_window_mask = ~in_window
    non_window_minutes = float((deltas_min * non_window_mask).sum())

    fp_time_percent = fp_minutes / total_minutes * 100.0 if total_minutes > 0 else np.nan
    fp_time_alert_percent = fp_minutes / alert_minutes * 100.0 if alert_minutes > 0 else np.nan
    fp_time_nonwindow_percent = (
        fp_minutes / non_window_minutes * 100.0 if non_window_minutes > 0 else np.nan
    )

    TP, FN = 0, 0
    mttd_vals, rows = [], []

    def overlaps(run, ws, we):
        return not (run["end"] < ws or run["start"] >= we)

    for w_start, w_end in windows:
        hits = [r for r in alert_runs if overlaps(r, w_start, w_end)]
        if hits:
            first_alert_time = min(max(r["start"], w_start) for r in hits)
            mttd = (w_end - first_alert_time).total_seconds() / 60.0
            TP += 1
            mttd_vals.append(mttd)
            rows.append(
                {
                    "incident_time": w_end,
                    "detected": True,
                    "first_alert_time": first_alert_time,
                    "MTTD_min": mttd,
                }
            )
        else:
            FN += 1
            rows.append(
                {
                    "incident_time": w_end,
                    "detected": False,
                    "first_alert_time": pd.NaT,
                    "MTTD_min": np.nan,
                }
            )

    def overlaps_any(run, wins):
        return any(overlaps(run, ws, we) for (ws, we) in wins)

    fp_runs = [r for r in alert_runs if not overlaps_any(r, windows)]
    FP = len(fp_runs)

    prec_inc = TP / (TP + FP) if TP + FP > 0 else 0.0
    rec_inc = TP / (TP + FN) if TP + FN > 0 else 0.0
    f1_inc = 2 * prec_inc * rec_inc / (prec_inc + rec_inc) if prec_inc + rec_inc > 0 else 0.0
    mean_mttd = float(np.nanmean(mttd_vals)) if mttd_vals else np.nan
    p90_mttd = float(np.nanpercentile(mttd_vals, 90)) if mttd_vals else np.nan

    hours_total = (end - start).total_seconds() / 3600.0 if (pd.notna(end) and pd.notna(start)) else np.nan
    fp_per_hour = FP / hours_total if hours_total and hours_total > 0 else np.nan
    percentage_of_ones_in_prediction = float(alert_bin.mean() * 100.0)

    out = {
        "incident_precision": prec_inc,
        "incident_recall": rec_inc,
        "incident_f1": f1_inc,
        "TP_incidents": TP,
        "FN_incidents": FN,
        "FP_episodes": FP,
        "FP_per_hour": fp_per_hour,
        "mean_MTTD_min": mean_mttd,
        "p90_MTTD_min": p90_mttd,
        "percentage_of_ones_in_prediction": percentage_of_ones_in_prediction,
        "incident_table": pd.DataFrame(rows),
        "fp_episode_table": pd.DataFrame(fp_runs),
        "fp_time_min": fp_minutes,
        "fp_time_percent": fp_time_percent,
        "fp_time_alert_percent": fp_time_alert_percent,
        "fp_time_nonwindow_percent": fp_time_nonwindow_percent,
    }

    if y_true is not None:
        y_bin_input = pd.Series(y_score_or_bin)
        if y_bin_input.dtype != int:
            y_bin = (y_bin_input >= thr).astype(int)
        else:
            y_bin = y_bin_input.astype(int)

        y_bin = y_bin.reindex(alert_bin.index).fillna(0).astype(int)

        out.update(
            {
                "classic_precision": precision_score(y_true, y_bin, zero_division=0),
                "classic_recall": recall_score(y_true, y_bin, zero_division=0),
                "classic_f1": f1_score(y_true, y_bin, zero_division=0),
            }
        )

    return out


def load_predictions(pred_path: str | Path, value_column: str = "pred_bin") -> pd.DataFrame:
    """Load AnoFusion predictions and normalize column names/types."""
    pred_df = read_csv_dataset(pred_path)
    if "timestamp" not in pred_df and "timetamp" in pred_df:
        pred_df = pred_df.rename(columns={"timetamp": "timestamp"})

    drop_cols = [col for col in ["Unnamed: 0"] if col in pred_df.columns]
    pred_df = pred_df.drop(columns=drop_cols)

    if value_column not in pred_df.columns:
        raise ValueError(f"Column '{value_column}' not found in {pred_path}. Available columns: {pred_df.columns.tolist()}")

    # normalize prediction column name to 'label' to align with truth
    if value_column != "label" and "label" in pred_df.columns:
        pred_df = pred_df.drop(columns=["label"])
    pred_df = pred_df.rename(columns={value_column: "label"})
    pred_df["label"] = pd.to_numeric(pred_df["label"].squeeze(), errors="coerce")

    pred_df = convert_timestamps_to_datetime(pred_df, "timestamp")
    return pred_df


def align_true_and_pred(true_df: pd.DataFrame, pred_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Inner-join truth/prediction by timestamp and return aligned Series indexed by datetime."""
    merged_df = pd.merge(
        pred_df[["timestamp", "label", "datetime"]],
        true_df[["timestamp", "label", "datetime"]],
        on="timestamp",
        suffixes=("_pred", "_true"),
    ).sort_values("datetime_true")

    merged_df = merged_df.set_index("datetime_true")
    y_pred = merged_df["label_pred"]
    y_true = merged_df["label_true"]
    return merged_df, y_true, y_pred


def build_incidents_from_labels(y_true: pd.Series) -> pd.DataFrame:
    """Derive incident timestamps from ground-truth labels as 0->1 transitions."""
    y_aligned = pd.Series(y_true).fillna(0).astype(int)
    y_aligned = y_aligned.sort_index()
    prev = y_aligned.shift(1).fillna(0).astype(int)
    incident_starts = y_aligned[(y_aligned == 1) & (prev == 0)].index
    return pd.DataFrame({"datetime": incident_starts})


def main(
    true_path: str | Path = "labeled_service/mobservice2.csv",
    pred_path: str | Path = "utils/mobservice2_ed.csv",
    pre_minutes: int = 15,
    pred_column: str = "pred_bin",
):
    true_df = get_data(true_path)
    pred_df = load_predictions(pred_path, value_column=pred_column)

    merged_df, y_true, y_pred = align_true_and_pred(true_df, pred_df)
    incidents_df = build_incidents_from_labels(y_true)

    metrics = eval_incident_metrics_robust(
        y_pred,
        incidents_df,
        pre_minutes=pre_minutes,
        thr=0.5,
        smooth="3min",
        min_len="2min",
        merge_gap="3min",
        cooldown="10min",
        y_true=y_true,
    )

    print("=== Classic pointwise metrics ===")
    print(f"precision: {metrics['classic_precision']:.4f}")
    print(f"recall:    {metrics['classic_recall']:.4f}")
    print(f"f1:        {metrics['classic_f1']:.4f}")
    print("\n=== Incident-level metrics ===")
    print(f"incident_precision: {metrics['incident_precision']:.4f}")
    print(f"incident_recall:    {metrics['incident_recall']:.4f}")
    print(f"incident_f1:        {metrics['incident_f1']:.4f}")
    print(f"mean_MTTD_min:      {metrics['mean_MTTD_min']:.2f}")
    print(f"p90_MTTD_min:       {metrics['p90_MTTD_min']:.2f}")
    print(f"TP_incidents:       {metrics['TP_incidents']}")
    print(f"FN_incidents:       {metrics['FN_incidents']}")
    print(f"FP_episodes:        {metrics['FP_episodes']}")
    print(f"FP_per_hour:        {metrics['FP_per_hour']:.4f}")
    print(f"fp_time_nonwindow_percent: {metrics['fp_time_nonwindow_percent']:.2f}")
    print("\nPreview of incident_table:")
    print(metrics["incident_table"].head())


if __name__ == "__main__":
    main()
