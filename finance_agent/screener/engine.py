"""
Market Screener Engine.

Scans an entire index using parallel data fetching + deterministic
quant scoring. NO LLM calls — this is a pure quantitative filter.

Scoring model (0-10):
  - Expected Value (scenario model)   30%
  - Piotroski F-Score (quality)       25%
  - Overall Quality Score (composite)  25%
  - Momentum (1Y price return)        10%
  - Altman Z-Score (safety)           10%
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Generator, Any

import yfinance as yf

from finance_agent.data.fetcher import YFinanceFetcher
from finance_agent.quant.metrics import MetricsEngine
from finance_agent.quant.scenarios import ScenarioBuilder
from finance_agent.models.schemas import Horizon
from finance_agent.utils.logger import logger


@dataclass
class ScreenerResult:
    ticker: str
    name: str = ""
    sector: str = ""
    industry: str = ""
    screener_score: float = 0.0
    expected_value: float | None = None
    piotroski: int | None = None
    altman_z: float | None = None
    quality_score: float | None = None
    momentum_1y: float | None = None
    current_price: float | None = None
    market_cap: float | None = None
    pe_ratio: float | None = None
    forward_pe: float | None = None
    revenue_growth: float | None = None
    fcf_yield: float | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


class ScreenerEngine:
    """
    Parallel quantitative screener. No LLM calls.

    Usage:
        engine = ScreenerEngine(max_workers=10)
        for update in engine.scan_stream(tickers, top_n=20):
            # update is a dict with type='progress' or type='result'
            yield update
    """

    # Weights for composite screener score
    WEIGHTS = {
        "ev":      0.30,
        "piotroski": 0.25,
        "quality": 0.25,
        "momentum": 0.10,
        "altman":  0.10,
    }

    def __init__(self, max_workers: int = 12) -> None:
        self.max_workers = max_workers
        self._fetcher = YFinanceFetcher()
        self._metrics = MetricsEngine()

    def scan_stream(
        self,
        tickers: list[str],
        top_n: int = 20,
        horizon: Horizon = Horizon.TWELVE_MONTHS,
    ) -> Generator[dict[str, Any], None, None]:
        """
        Generator that yields progress + result events.
        Designed to be used with FastAPI StreamingResponse (SSE).
        """
        total = len(tickers)
        done = 0
        results: list[ScreenerResult] = []

        yield {"type": "start", "total": total}

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(self._score_ticker, t, horizon): t for t in tickers}

            for future in as_completed(futures):
                done += 1
                result = future.result()
                results.append(result)

                yield {
                    "type": "progress",
                    "done": done,
                    "total": total,
                    "ticker": result.ticker,
                    "ok": result.error is None,
                }

        # Sort by screener_score descending, filter errors
        ranked = sorted(
            [r for r in results if r.error is None],
            key=lambda r: r.screener_score,
            reverse=True
        )[:top_n]

        yield {"type": "results", "data": [r.to_dict() for r in ranked]}

    def _score_ticker(self, ticker: str, horizon: Horizon) -> ScreenerResult:
        result = ScreenerResult(ticker=ticker)
        try:
            # Fetch raw data
            raw = self._fetcher.fetch(ticker, price_period="2y")
            info = raw.info or {}

            result.name = info.get("shortName", ticker)
            result.sector = info.get("sector", "Unknown")
            result.industry = info.get("industry", "Unknown")

            # Compute metrics (no benchmark for speed)
            ms = self._metrics.compute(raw, benchmark_prices=None)

            result.current_price = ms.get("current_price")
            result.market_cap = ms.get("market_cap")
            result.pe_ratio = ms.get("pe_ratio")
            result.forward_pe = ms.get("forward_pe")
            result.revenue_growth = ms.get("revenue_growth_yoy")
            result.fcf_yield = ms.get("fcf_yield")
            result.piotroski = ms.get("piotroski_f_score")
            result.altman_z = ms.get("altman_z_score")
            result.quality_score = ms.get("overall_quality_score")
            result.momentum_1y = ms.get("return_365d") or ms.get("return_ytd")

            # Scenario model for Expected Value
            try:
                scenario_builder = ScenarioBuilder()
                scenario = scenario_builder.build(ms, horizon=horizon)
                result.expected_value = scenario.expected_value
            except Exception as se:
                logger.debug(f"[Screener] Scenario failed for {ticker}: {se}")
                result.expected_value = None

            # Compute composite screener score (0-10)
            result.screener_score = self._composite_score(result)

        except Exception as e:
            logger.warning(f"[Screener] Failed to score {ticker}: {e}")
            result.error = str(e)

        return result

    def _composite_score(self, r: ScreenerResult) -> float:
        """Combine individual signals into a single 0-10 screener score."""
        component_scores: dict[str, float] = {}

        # 1. Expected Value (EV): clamp to [-40%, +60%], map to 0-10
        if r.expected_value is not None:
            ev_pct = r.expected_value if abs(r.expected_value) <= 1 else r.expected_value / 100
            # map -40% → 0, 0% → 3.33, +60% → 10
            component_scores["ev"] = max(0.0, min(10.0, (ev_pct + 0.40) / 1.00 * 10))

        # 2. Piotroski F-Score (0-9) → normalize to 0-10
        if r.piotroski is not None:
            component_scores["piotroski"] = (r.piotroski / 9.0) * 10.0

        # 3. Overall Quality Score (already 0-10)
        if r.quality_score is not None:
            component_scores["quality"] = max(0.0, min(10.0, r.quality_score))

        # 4. Momentum 1Y: clamp [-50%, +100%], map to 0-10
        if r.momentum_1y is not None:
            mom = r.momentum_1y if abs(r.momentum_1y) <= 1 else r.momentum_1y / 100
            # Moderate positive momentum is good; excessive may mean overvalued
            # Sweet spot: 10-50% → high scores
            if mom >= 0:
                mom_score = min(10.0, mom * 15)      # 67% gain → 10
            else:
                mom_score = max(0.0, 5.0 + mom * 10)  # -50% → 0, 0% → 5
            component_scores["momentum"] = mom_score

        # 5. Altman Z-Score: >2.99=safe(10), 1.81-2.99=grey(5), <1.81=distress(0)
        if r.altman_z is not None:
            z = r.altman_z
            if z >= 2.99:
                component_scores["altman"] = 10.0
            elif z >= 1.81:
                component_scores["altman"] = 5.0 + (z - 1.81) / (2.99 - 1.81) * 5.0
            else:
                component_scores["altman"] = max(0.0, (z / 1.81) * 5.0)

        if not component_scores:
            return 0.0

        # Weighted average — only use weights for available components
        total_weight = sum(self.WEIGHTS[k] for k in component_scores)
        if total_weight == 0:
            return 0.0

        weighted_sum = sum(self.WEIGHTS[k] * v for k, v in component_scores.items())
        normalized = (weighted_sum / total_weight)
        return round(max(0.0, min(10.0, normalized)), 2)
