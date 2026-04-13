from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LinkedItem:
    title: str
    role: str
    impact: str
    links: list[dict[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class PortfolioProfile:
    full_name: str
    headline: str
    publications: list[LinkedItem] = field(default_factory=list)
    posters: list[LinkedItem] = field(default_factory=list)
    startups: list[LinkedItem] = field(default_factory=list)
    notes: str = ""


def _load_linked_items(raw_items: list[dict[str, Any]]) -> list[LinkedItem]:
    items: list[LinkedItem] = []
    for item in raw_items:
        items.append(
            LinkedItem(
                title=str(item.get("title", "Untitled project")),
                role=str(item.get("role", "Contributor")),
                impact=str(item.get("impact", "Describe measurable impact here.")),
                links=[
                    {"label": str(link.get("label", "Reference")), "url": str(link.get("url", ""))}
                    for link in item.get("links", [])
                    if link.get("url")
                ],
            )
        )
    return items


def load_profile(profile_path: Path) -> PortfolioProfile:
    data = json.loads(profile_path.read_text(encoding="utf-8"))
    return PortfolioProfile(
        full_name=str(data.get("full_name", "Your Name")),
        headline=str(data.get("headline", "Researcher and builder")),
        publications=_load_linked_items(data.get("publications", [])),
        posters=_load_linked_items(data.get("posters", [])),
        startups=_load_linked_items(data.get("startups", [])),
        notes=str(data.get("notes", "")),
    )


def _format_links(links: list[dict[str, str]]) -> str:
    if not links:
        return ""
    formatted = [f"{link['label']}: {link['url']}" for link in links]
    return " | ".join(formatted)


def _render_item(item: LinkedItem) -> str:
    link_text = _format_links(item.links)
    parts = [f"- {item.title}", f"  Role: {item.role}", f"  Impact: {item.impact}"]
    if link_text:
        parts.append(f"  Evidence: {link_text}")
    return "\n".join(parts)


def _compact_reference(item: LinkedItem) -> str:
    if not item.links:
        return item.title
    return " | ".join(
        f"{link.get('label', 'Reference')}: {link.get('url', '')}"
        for link in item.links
        if link.get("url")
    )


def build_admissions_answer(profile: PortfolioProfile, metrics: dict[str, Any]) -> str:
    accuracy = metrics.get("accuracy")
    f1_weighted = metrics.get("f1_weighted")
    dataset_rows = metrics.get("dataset_rows")
    training_mode = metrics.get("training_mode", "direct_fit")

    research_focus = profile.publications[0] if profile.publications else None
    poster_focus = profile.posters[0] if profile.posters else None
    startup_focus = profile.startups[0] if profile.startups else None

    answer = [
        f"{profile.full_name} is a {profile.headline} who combines applied research, public-interest engineering, and venture-style execution.",
        f"In this project, I built a reproducible sentiment-analysis pipeline on {dataset_rows} labeled records and achieved {accuracy:.3f} accuracy with {f1_weighted:.3f} weighted F1 using {training_mode}.",
        _build_evidence_line(profile),
        "My role was end-to-end: I shaped the data pipeline, trained and evaluated the model, and packaged the work into a FastAPI demo so non-technical reviewers can inspect results quickly.",
    ]

    if research_focus:
        answer.append(
            f"For research, I can point to {research_focus.title}, where I served as {research_focus.role}. The work mattered because {research_focus.impact}"
        )
    if poster_focus:
        answer.append(
            f"For dissemination, I presented {poster_focus.title} as {poster_focus.role}; that poster translated the technical work into a concise story for a broader audience and demonstrated {poster_focus.impact}"
        )
    if startup_focus:
        answer.append(
            f"For entrepreneurship, {startup_focus.title} shows how I approached execution as a builder rather than a spectator. I acted as {startup_focus.role}, and the impact was {startup_focus.impact}"
        )

    answer.append(
        "This combination of measurable technical output, public communication, and execution discipline is the most honest signal I can give an admissions committee about my readiness for rigorous graduate work."
    )

    text = " ".join(answer)
    words = text.split()
    if len(words) > 250:
        text = " ".join(words[:250]).rstrip() + "."
    return text


def _build_evidence_line(profile: PortfolioProfile) -> str:
    evidence_parts: list[str] = []
    if profile.publications:
        evidence_parts.append(f"Research evidence: {_compact_reference(profile.publications[0])}")
    if profile.posters:
        evidence_parts.append(f"Poster evidence: {_compact_reference(profile.posters[0])}")
    if profile.startups:
        evidence_parts.append(f"Entrepreneurship evidence: {_compact_reference(profile.startups[0])}")
    if not evidence_parts:
        return "Evidence: replace the template links with your own DOI, Google Scholar, ResearchGate, GitHub, or website references."
    return " ".join(evidence_parts)


def build_markdown_report(profile: PortfolioProfile, metrics: dict[str, Any]) -> str:
    answer = build_admissions_answer(profile, metrics)

    sections = [
        f"# Admissions Portfolio: {profile.full_name}",
        f"## Headline\n{profile.headline}",
        f"## 250-Word Response\n{answer}",
        "## Verified Project Metrics",
        f"- Dataset rows: {metrics.get('dataset_rows', 'n/a')}",
        f"- Accuracy: {metrics.get('accuracy', 'n/a')}",
        f"- Weighted F1: {metrics.get('f1_weighted', 'n/a')}",
        f"- Text column used: {metrics.get('text_column_used', 'n/a')}",
    ]

    if profile.publications:
        sections.append("## Publications")
        sections.extend(_render_item(item) for item in profile.publications)

    if profile.posters:
        sections.append("## Posters")
        sections.extend(_render_item(item) for item in profile.posters)

    if profile.startups:
        sections.append("## Entrepreneurship")
        sections.extend(_render_item(item) for item in profile.startups)

    if profile.notes:
        sections.append(f"## Notes\n{profile.notes}")

    return "\n\n".join(sections).strip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build an admissions-style portfolio summary")
    parser.add_argument(
        "--profile-path",
        type=Path,
        default=Path("examples/admissions_profile.example.json"),
        help="Path to a JSON profile containing publications, posters, and startup evidence.",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=Path("artifacts/metrics.json"),
        help="Path to the training metrics JSON file.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("artifacts/admissions_portfolio.md"),
        help="Where to write the generated markdown report.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    profile = load_profile(args.profile_path)
    metrics = json.loads(args.metrics_path.read_text(encoding="utf-8"))
    report = build_markdown_report(profile, metrics)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(report, encoding="utf-8")
    print(f"Portfolio report written to: {args.output_path}")


if __name__ == "__main__":
    main()