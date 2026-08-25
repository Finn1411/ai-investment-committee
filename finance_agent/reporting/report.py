"""
Report Formatter — Week 4.

Generates two output formats from a CommitteeResult:
1. Rich terminal report (colour-coded, structured)
2. Structured JSON export (for storage / downstream use)

Usage:
    from finance_agent.reporting.report import ReportFormatter
    fmt = ReportFormatter(result)
    fmt.print_terminal()
    json_path = fmt.export_json("reports/AAPL_2026-08-24.json")
"""

from __future__ import annotations

import json
import textwrap
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from finance_agent.agents.committee import CommitteeResult

# Score colour thresholds (ANSI terminal colours)
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_RED    = "\033[91m"
_YELLOW = "\033[93m"
_GREEN  = "\033[92m"
_CYAN   = "\033[96m"
_WHITE  = "\033[97m"
_DIM    = "\033[2m"


def _score_colour(score: float) -> str:
    if score >= 7.0:   return _GREEN
    if score >= 5.0:   return _YELLOW
    return _RED


def _rating_colour(rating: str) -> str:
    if rating == "BUY":   return _GREEN
    if rating == "AVOID": return _RED
    return _YELLOW


def _bar(score: float, width: int = 20) -> str:
    """ASCII progress bar for a 0-10 score."""
    filled = round((score / 10.0) * width)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


class ReportFormatter:
    """
    Formats a CommitteeResult into terminal output or JSON.

    Args:
        result: CommitteeResult from CommitteeEngine.run()
    """

    def __init__(self, result: "CommitteeResult") -> None:
        self.result = result
        self.verdict = result.verdict
        self.v = result.verdict

    # ── Terminal Report ───────────────────────────────────────────────────────

    def print_terminal(self, use_colour: bool = True) -> None:
        """Print a full formatted committee report to stdout."""
        lines = self._build_terminal_lines(use_colour=use_colour)
        print("\n".join(lines))

    def _build_terminal_lines(self, use_colour: bool = True) -> list[str]:
        v = self.v
        r = self.result

        def c(colour: str, text: str) -> str:
            return f"{colour}{text}{_RESET}" if use_colour else text

        def bold(text: str) -> str:
            return c(_BOLD, text)

        def section(title: str) -> str:
            bar = "=" * 70
            return f"\n{bar}\n  {bold(title)}\n{bar}"

        lines = []
        sep = "=" * 70

        # ── Header ────────────────────────────────────────────────────────────
        rating_str = c(_rating_colour(v.rating.value) + _BOLD, f"  {v.rating_label}  ")
        lines += [
            "",
            sep,
            c(_BOLD + _WHITE, f"  FINANCE AGENT COMMITTEE REPORT: {v.ticker}"),
            sep,
            f"  Date:       {v.analysis_date}  |  Horizon: {v.horizon.value}",
            f"  Rating:    {rating_str}",
            f"  Score:      {c(_score_colour(v.weighted_score), f'{v.weighted_score:.2f}/10')}  "
            f"{_bar(v.weighted_score)}",
            f"  Confidence: {v.confidence:.1%}",
        ]

        if v.disagreement_flag:
            lines += [
                "",
                c(_RED + _BOLD, f"  *** SPLIT COMMITTEE — High disagreement detected ***"),
                c(_YELLOW, f"  Disagreement score: {v.disagreement_score:.2f}  |  "
                          f"Conflicting agents: {', '.join(v.conflicting_agents)}"),
            ]

        # ── Agent Scores ──────────────────────────────────────────────────────
        lines.append(section("AGENT SCORES"))
        weights = {
            "FundamentalAnalyst": 0.20,
            "ValueAnalyst":        0.20,
            "GrowthAnalyst":       0.15,
            "EarningsAnalyst":     0.10,
            "RiskManager":         0.15,
            "BearAnalyst":         0.20,
        }
        for agent_name, score in sorted(r.agent_scores.items(), key=lambda x: -x[1]):
            conf = r.agent_confidences.get(agent_name, 0)
            weight = weights.get(agent_name, 0)
            bar = _bar(score, 15)
            score_str = c(_score_colour(score), f"{score:>5.2f}/10")
            conflict_tag = c(_RED, " [CONFLICT]") if agent_name in v.conflicting_agents else ""
            lines.append(
                f"  {agent_name:<25} {score_str}  {bar}  "
                f"conf={conf:.0%}  w={weight:.0%}{conflict_tag}"
            )

        if r.portfolio_analysis:
            pa = r.portfolio_analysis
            score_str = c(_score_colour(pa.score or 5.0), f"{pa.score:>5.2f}/10")
            lines.append(
                f"  {'PortfolioManager':<25} {score_str}  {_bar(pa.score or 5.0, 15)}  "
                f"(advisory — not in weighted score)"
            )

        # ── Scenarios ─────────────────────────────────────────────────────────
        lines.append(section("SCENARIO MODEL (Monte Carlo)"))
        sm = v.scenario_model
        ev = sm.expected_value
        ev_colour = _GREEN if ev > 0 else _RED
        lines += [
            f"  Bear ({sm.bear.probability:.0%}):  {c(_RED, f'{sm.bear.expected_return:+.1%}')}",
            f"  Base ({sm.base.probability:.0%}):  {sm.base.expected_return:+.1%}",
            f"  Bull ({sm.bull.probability:.0%}):  {c(_GREEN, f'{sm.bull.expected_return:+.1%}')}",
            f"  Expected Value:     {c(ev_colour + _BOLD, f'{ev:+.1%}')}",
            f"  P(Positive Return): {sm.prob_outperform:.0%}",
        ]

        # ── Bull / Bear Case ──────────────────────────────────────────────────
        if v.bull_case_summary or v.bear_case_summary:
            lines.append(section("BULL vs BEAR"))
            if v.bull_case_summary:
                lines.append(c(_GREEN, "  BULL CASE:"))
                for line in textwrap.wrap(v.bull_case_summary, 66):
                    lines.append(f"  {line}")
            if v.bear_case_summary:
                lines += [""]
                lines.append(c(_RED, "  BEAR CASE:"))
                for line in textwrap.wrap(v.bear_case_summary, 66):
                    lines.append(f"  {line}")

        # ── Investment Thesis ─────────────────────────────────────────────────
        lines.append(section("INVESTMENT THESIS"))
        for para in v.thesis.split("\n"):
            for line in textwrap.wrap(para.strip(), 66):
                lines.append(f"  {line}")
            if para.strip():
                lines.append("")

        # ── Invalidation Criteria ─────────────────────────────────────────────
        lines.append(section("INVALIDATION CRITERIA (exit triggers)"))
        for i, criterion in enumerate(v.invalidation_criteria, 1):
            lines.append(c(_YELLOW, f"  {i}. {criterion}"))

        # ── Key Findings per Agent ────────────────────────────────────────────
        lines.append(section("KEY FINDINGS BY AGENT"))
        for agent_name, analysis in self.result.agent_analyses.items():
            lines.append(f"\n  {bold(agent_name)} (score {analysis.score:.1f}/10):")
            lines.append(f"  {c(_DIM, analysis.summary)}")
            for finding in (analysis.key_findings or [])[:4]:
                wrapped = textwrap.wrap(finding, 62)
                if wrapped:
                    lines.append(f"    - {wrapped[0]}")
                    for extra in wrapped[1:]:
                        lines.append(f"      {extra}")

        # ── Portfolio Manager ─────────────────────────────────────────────────
        if r.portfolio_analysis:
            lines.append(section("PORTFOLIO MANAGER"))
            pa = r.portfolio_analysis
            lines.append(f"  {pa.summary}")
            for finding in (pa.key_findings or [])[-3:]:
                lines.append(f"    - {finding}")

        # ── Footer ────────────────────────────────────────────────────────────
        lines += [
            "",
            sep,
            c(_DIM, f"  Runtime: {r.total_runtime_seconds:.1f}s  |  "
                    f"Agents: {len(r.agent_analyses)}  |  "
                    f"Generated: {date.today()}"),
            sep,
            "",
        ]

        return lines

    # ── JSON Export ───────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialize the full committee result to a structured dict."""
        v = self.v
        r = self.result

        agent_outputs = {}
        for name, analysis in r.agent_analyses.items():
            agent_outputs[name] = {
                "score": analysis.score,
                "confidence": analysis.confidence,
                "summary": analysis.summary,
                "key_findings": analysis.key_findings,
            }

        if r.portfolio_analysis:
            agent_outputs["PortfolioManager"] = {
                "score": r.portfolio_analysis.score,
                "confidence": r.portfolio_analysis.confidence,
                "summary": r.portfolio_analysis.summary,
                "key_findings": r.portfolio_analysis.key_findings,
            }

        return {
            "ticker": v.ticker,
            "analysis_date": str(v.analysis_date),
            "horizon": v.horizon.value,
            "rating": v.rating.value,
            "rating_label": v.rating_label,
            "weighted_score": v.weighted_score,
            "confidence": v.confidence,
            "disagreement": {
                "score": v.disagreement_score,
                "flag": v.disagreement_flag,
                "conflicting_agents": v.conflicting_agents,
            },
            "scenarios": {
                "bear": {
                    "probability": v.scenario_model.bear.probability,
                    "return": v.scenario_model.bear.expected_return,
                    "narrative": v.scenario_model.bear.narrative,
                },
                "base": {
                    "probability": v.scenario_model.base.probability,
                    "return": v.scenario_model.base.expected_return,
                    "narrative": v.scenario_model.base.narrative,
                },
                "bull": {
                    "probability": v.scenario_model.bull.probability,
                    "return": v.scenario_model.bull.expected_return,
                    "narrative": v.scenario_model.bull.narrative,
                },
                "expected_value": v.scenario_model.expected_value,
                "prob_positive": v.scenario_model.prob_outperform,
            },
            "thesis": v.thesis,
            "bull_case_summary": v.bull_case_summary,
            "bear_case_summary": v.bear_case_summary,
            "invalidation_criteria": v.invalidation_criteria,
            "agent_scores": v.agent_scores,
            "agent_confidences": v.agent_confidences,
            "agent_outputs": agent_outputs,
            "runtime_seconds": r.total_runtime_seconds,
        }

    def export_json(self, path: str | Path | None = None) -> Path:
        """
        Export the full result to a JSON file.

        Args:
            path: Optional explicit path. Defaults to reports/{ticker}_{date}.json

        Returns:
            Path to the written file.
        """
        if path is None:
            reports_dir = Path("reports")
            reports_dir.mkdir(exist_ok=True)
            path = reports_dir / f"{self.v.ticker}_{self.v.analysis_date}.json"

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    def export_markdown(self, path: str | Path | None = None) -> Path:
        """
        Export the full report as a Markdown file (no ANSI colours).

        Returns:
            Path to the written file.
        """
        if path is None:
            reports_dir = Path("reports")
            reports_dir.mkdir(exist_ok=True)
            path = reports_dir / f"{self.v.ticker}_{self.v.analysis_date}.md"

        path = Path(path)
        lines = self._build_terminal_lines(use_colour=False)
        path.write_text("\n".join(lines), encoding="utf-8")
        return path
