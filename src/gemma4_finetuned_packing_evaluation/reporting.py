from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


def print_results_table(df: pd.DataFrame) -> None:
    label = df["model"].iloc[0]
    print(f"\n{'=' * 80}")
    print(f"  PER-SAMPLE RESULTS — {label}")
    print(f"{'=' * 80}")

    view = df[[
        "prompt", "trip_type", "n_items", "tps",
        "json_valid", "expert_recall", "bert_score_f1",
        "accuracy", "expertise", "logic",
    ]].copy()
    view["prompt"]     = view["prompt"].str[:42] + "..."
    view["json_valid"] = view["json_valid"].map({True: "YES", False: "NO"})
    view.columns = [
        "Prompt", "Trip Type", "# Items", "TPS",
        "JSON OK", "Expert Recall", "BERTScore F1",
        "Accuracy", "Expertise", "Logic",
    ]
    print(view.to_string(index=False))

    print(f"\n--- Aggregate — {label} ---")
    num_cols = ["tps", "expert_recall", "bert_score_f1", "accuracy", "expertise", "logic"]
    print(df[num_cols].agg(["mean", "std", "min", "max"]).round(3).to_string())
    print(f"\nJSON Validity Rate : {df['json_valid'].mean() * 100:.1f}%")
    print(f"Avg Items / Response: {df['n_items'].mean():.1f}")


def print_judge_comments(combined_df: pd.DataFrame) -> None:
    print(f"\n{'=' * 80}")
    print("  LLM JUDGE COMMENTS")
    print(f"{'=' * 80}")
    view = combined_df[["model", "trip_type", "accuracy", "expertise", "logic", "judge_comment"]].copy()
    view.columns = ["Model", "Trip Type", "Accuracy", "Expertise", "Logic", "Comment"]
    print(view.to_string(index=False))


def _normalize_metrics(df: pd.DataFrame, global_max_tps: float = 1.0) -> dict:
    def norm_judge(col: str) -> float:
        valid = df[col].dropna()
        return round((valid.mean() - 1) / 4, 4) if not valid.empty else 0.0

    return {
        "JSON Validity":     round(df["json_valid"].mean(), 4),
        "TPS (relative)":    round(df["tps"].mean() / global_max_tps, 4) if global_max_tps > 0 else 0.0,
        "Expert Recall":     round(df["expert_recall"].mean(), 4),
        "BERTScore F1":      round(df["bert_score_f1"].mean(), 4),
        "Accuracy (Judge)":  norm_judge("accuracy"),
        "Expertise (Judge)": norm_judge("expertise"),
        "Logic (Judge)":     norm_judge("logic"),
    }


def print_metrics_comparison(metrics_by_label: dict[str, dict]) -> None:
    print(f"\n{'=' * 65}")
    print("  NORMALISED METRICS (0 – 1)  ·  radar chart")
    print(f"{'=' * 65}")
    comp = pd.DataFrame(metrics_by_label).rename_axis("Metric")
    print(comp.round(4).to_string())


_PALETTE = [
    ("99,110,250", "rgba(99,110,250,0.18)"),
    ("239,85,59",  "rgba(239,85,59,0.18)"),
]


def plot_radar_comparison(metrics_by_label: dict[str, dict], title: str) -> go.Figure:
    fig = go.Figure()
    for (label, metrics), (line_rgb, fill_rgba) in zip(metrics_by_label.items(), _PALETTE):
        cats = list(metrics.keys())
        vals = list(metrics.values())
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=cats + [cats[0]],
            fill="toself", fillcolor=fill_rgba,
            line=dict(color=f"rgb({line_rgb})", width=2.5),
            name=label, mode="lines+markers", marker=dict(size=7),
        ))
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=20)),
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True, paper_bgcolor="white",
        width=760, height=630,
    )
    return fig


def print_summary(combined: pd.DataFrame) -> None:
    print(f"\n{'=' * 60}\nCOMPARISON SUMMARY\n{'=' * 60}")
    summary = (
        combined[["model", "json_valid", "tps", "expert_recall",
                   "bert_score_f1", "accuracy", "expertise", "logic"]]
        .groupby("model")
        .agg(
            json_validity_rate=("json_valid", "mean"),
            avg_tps=("tps", "mean"),
            avg_expert_recall=("expert_recall", "mean"),
            avg_bert_f1=("bert_score_f1", "mean"),
            avg_accuracy=("accuracy", "mean"),
            avg_expertise=("expertise", "mean"),
            avg_logic=("logic", "mean"),
        )
        .round(4)
    )
    summary["json_validity_rate"] = (summary["json_validity_rate"] * 100).map("{:.1f}%".format)
    print(summary.to_string())


def save_metrics_md(
    combined: pd.DataFrame,
    metrics_by_label: dict[str, dict],
    num_samples: int,
    path: str = "METRICS.md",
) -> None:
    from datetime import datetime

    lines = [
        "# Evaluation Metrics — SmartPack AI",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
        f"**Samples:** {num_samples} random test examples  ",
        f"**Models:** {', '.join(metrics_by_label.keys())}",
        "",
        "## Summary",
        "",
    ]

    summary = (
        combined[["model", "json_valid", "tps", "expert_recall",
                   "bert_score_f1", "accuracy", "expertise", "logic"]]
        .groupby("model")
        .agg(
            json_validity=("json_valid", "mean"),
            avg_tps=("tps", "mean"),
            avg_expert_recall=("expert_recall", "mean"),
            avg_bert_f1=("bert_score_f1", "mean"),
            avg_accuracy=("accuracy", "mean"),
            avg_expertise=("expertise", "mean"),
            avg_logic=("logic", "mean"),
        )
        .round(4)
    )
    summary["json_validity"] = (summary["json_validity"] * 100).map("{:.1f}%".format)

    # Markdown table
    cols = list(summary.columns)
    lines.append("| Model | " + " | ".join(cols) + " |")
    lines.append("|-------|" + "|".join(["-------"] * len(cols)) + "|")
    for model_name, row in summary.iterrows():
        lines.append(f"| {model_name} | " + " | ".join(str(v) for v in row) + " |")

    lines += ["", "## Normalised Metrics (0–1)", ""]
    metric_names = list(next(iter(metrics_by_label.values())).keys())
    lines.append("| Metric | " + " | ".join(metrics_by_label.keys()) + " |")
    lines.append("|--------|" + "|".join(["--------"] * len(metrics_by_label)) + "|")
    for metric in metric_names:
        vals = [str(round(metrics_by_label[label][metric], 4)) for label in metrics_by_label]
        lines.append(f"| {metric} | " + " | ".join(vals) + " |")

    lines += [""]

    Path(path).write_text("\n".join(lines), encoding="utf-8")
    print(f"Metrics saved to {path}")


def save_results(combined: pd.DataFrame, output_dir: str = "results") -> None:
    out = Path(output_dir)
    out.mkdir(exist_ok=True)
    combined.to_csv(out / "results.csv", index=False)
    # Save raw outputs separately (they're large)
    combined[["model", "prompt", "raw_output"]].to_json(
        out / "raw_outputs.jsonl", orient="records", lines=True, force_ascii=False
    )
    print(f"\nResults saved to {out}/")
