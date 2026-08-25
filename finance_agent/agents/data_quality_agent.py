"""
Data Quality Agent — validates all data BEFORE any analyst sees it.
Implements Phase 1 data validation checks from the masterplan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from finance_agent.utils.logger import logger


@dataclass
class DataQualityReport:
    ticker: str
    as_of: date
    passed: bool = True
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    missing_fields: list[str] = field(default_factory=list)
    outliers_detected: list[str] = field(default_factory=list)
    estimated_fields: list[str] = field(default_factory=list)

    def fail(self, msg: str) -> None:
        self.passed = False
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)


class DataQualityAgent:
    """
    Validates structured data dicts before passing them to analysts.
    Does NOT call any LLM — all checks are deterministic.
    """

    # Fields that must exist for a minimum viable analysis
    REQUIRED_MARKET_FIELDS = ["close", "volume", "market_cap"]
    REQUIRED_FUNDAMENTAL_FIELDS = [
        "revenue", "gross_margin", "operating_margin", "net_margin",
        "free_cash_flow", "eps_diluted",
    ]
    REQUIRED_VALUATION_FIELDS = ["pe_ratio", "ev_to_ebitda", "price_to_sales"]

    # Sanity bounds for common ratios
    BOUNDS: dict[str, tuple[float, float]] = {
        "gross_margin": (-1.0, 1.0),
        "operating_margin": (-5.0, 1.0),
        "net_margin": (-10.0, 1.0),
        "pe_ratio": (-500.0, 3000.0),
        "ev_to_ebitda": (-100.0, 500.0),
        "beta": (-5.0, 10.0),
        "current_ratio": (0.0, 100.0),
        "net_debt_to_ebitda": (-50.0, 100.0),
    }

    def validate(self, ticker: str, data: dict[str, Any]) -> DataQualityReport:
        report = DataQualityReport(ticker=ticker, as_of=date.today())

        self._check_missing(report, data)
        self._check_bounds(report, data)
        self._check_staleness(report, data)
        self._check_consistency(report, data)
        self._flag_estimates(report, data)

        if report.errors:
            logger.warning(
                f"[DataQualityAgent] {ticker}: {len(report.errors)} error(s), "
                f"{len(report.warnings)} warning(s) — FAILED"
            )
        else:
            logger.info(
                f"[DataQualityAgent] {ticker}: passed with "
                f"{len(report.warnings)} warning(s)"
            )

        return report

    # ── Internal checks ───────────────────────────────────────────────────────

    def _check_missing(self, report: DataQualityReport, data: dict) -> None:
        for field_name in self.REQUIRED_MARKET_FIELDS + self.REQUIRED_FUNDAMENTAL_FIELDS:
            val = data.get(field_name)
            if val is None or (isinstance(val, float) and val != val):  # NaN
                report.missing_fields.append(field_name)
                report.warn(f"Missing field: {field_name}")

    def _check_bounds(self, report: DataQualityReport, data: dict) -> None:
        for field_name, (lo, hi) in self.BOUNDS.items():
            val = data.get(field_name)
            if val is None:
                continue
            try:
                fval = float(val)
            except (TypeError, ValueError):
                report.fail(f"Non-numeric value for {field_name}: {val!r}")
                continue
            if not (lo <= fval <= hi):
                report.outliers_detected.append(field_name)
                report.warn(
                    f"Outlier detected: {field_name}={fval:.4f} "
                    f"(expected [{lo}, {hi}])"
                )

    def _check_staleness(self, report: DataQualityReport, data: dict) -> None:
        """Warn if the last data update is more than 3 business days old."""
        last_updated = data.get("last_updated")
        if not last_updated:
            report.warn("No 'last_updated' timestamp in data.")
            return
        try:
            if isinstance(last_updated, str):
                last_updated = date.fromisoformat(last_updated)
            staleness = (date.today() - last_updated).days
            if staleness > 5:
                report.warn(f"Data is {staleness} days old (last_updated={last_updated})")
        except (ValueError, TypeError):
            report.warn(f"Could not parse 'last_updated': {last_updated!r}")

    def _check_consistency(self, report: DataQualityReport, data: dict) -> None:
        """Cross-field consistency checks."""
        # FCF yield should match FCF / market_cap if both present
        fcf = data.get("free_cash_flow")
        mc = data.get("market_cap")
        reported_yield = data.get("fcf_yield")
        if fcf and mc and reported_yield and mc > 0:
            derived = fcf / mc
            if abs(derived - reported_yield) > 0.05:
                report.warn(
                    f"FCF yield inconsistency: reported={reported_yield:.4f}, "
                    f"derived={derived:.4f}"
                )

        # Operating margin should be <= gross margin
        gm = data.get("gross_margin")
        om = data.get("operating_margin")
        if gm is not None and om is not None:
            if om > gm + 0.01:  # allow small rounding
                report.warn(
                    f"Operating margin ({om:.2%}) > Gross margin ({gm:.2%}) — check data."
                )

    def _flag_estimates(self, report: DataQualityReport, data: dict) -> None:
        """Flag any fields explicitly marked as estimated."""
        for key, val in data.items():
            if isinstance(key, str) and key.endswith("_estimated") and val:
                base = key.replace("_estimated", "")
                report.estimated_fields.append(base)
                report.warn(f"Field is estimated (not reported): {base}")
