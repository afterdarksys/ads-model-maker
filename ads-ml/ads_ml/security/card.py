"""Write a model card a security model maker can hand to a reviewer."""

from __future__ import annotations

from pathlib import Path


def render_model_card(
    *,
    task: str,
    labels: list[str],
    metrics: dict,
    data_summary: dict,
    redaction_counts: dict[str, int],
    footprint: dict,
    intended_use: str,
    out_of_scope: str,
) -> str:
    per_class = metrics.get("per_class") or []
    lines = [
        f"# Model card: {task}",
        "",
        "## Intended use",
        "",
        intended_use,
        "",
        "## Out of scope",
        "",
        out_of_scope,
        "",
        "## Labels",
        "",
        ", ".join(labels) if labels else "(none)",
        "",
        "## Data",
        "",
        f"- Examples kept: {data_summary.get('kept', 0)}",
        f"- Near-duplicates removed: {data_summary.get('duplicates_removed', 0)}",
        f"- Secrets redacted: {sum(redaction_counts.values())}",
        "",
        "## Metrics",
        "",
        f"- Accuracy: {metrics.get('accuracy', 0):.4f}",
        f"- Macro F1: {metrics.get('macro_f1', 0):.4f}",
        f"- Eval examples: {metrics.get('examples', 0)}",
        "",
    ]
    if per_class:
        lines.append("| Label | Precision | Recall | F1 | Support |")
        lines.append("| --- | --- | --- | --- | --- |")
        for row in per_class:
            lines.append(
                f"| {row['label']} | {row['precision']:.3f} | {row['recall']:.3f} | "
                f"{row['f1']:.3f} | {row['support']} |"
            )
        lines.append("")

    lines.extend([
        "## Footprint",
        "",
        f"- Parameters: {footprint.get('parameters', 0)}",
        f"- Disk: {footprint.get('disk_mb', 0):.4f} MB",
        f"- Estimated peak RAM: {footprint.get('estimated_peak_ram_mb', 0):.4f} MB",
        "",
        "This card describes a text classifier. It is not a detector guarantee",
        "and it does not replace review of the alerts the model is unsure about.",
        "",
    ])
    return "\n".join(lines)


def write_model_card(path: Path | str, **kwargs) -> str:
    text = render_model_card(**kwargs)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")
    return text
