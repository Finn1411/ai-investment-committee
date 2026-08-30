"""
Market Screener Engine — Professional 7-Factor Model.

Architecture:
  - Pass 1: Parallel data fetch + raw factor computation for every ticker
  - Pass 2: Cross-sectional z-score normalisation within GICS sectors
             (fallback to global z-score when sector < MIN_SECTOR_SIZE stocks)
  - Pass 3: Weighted factor composite → mapped to [0, 10] score

Factor model (AQR / systematic-equity inspired):
  Quality      (QMJ-style)    25%
  Value        (EY + FCF + PB) 20%
  Momentum     (12-1 month)   20%
  Profitability (ROIC + margins) 15%
  Safety       (Piotroski + Altman) 10%
  Growth       (Rev CAGR + EPS)   5%
  Risk-Adjusted (Sharpe + low-vol) 5%

Enriched result fields include Beneish M-Score (fraud flag), Magic Formula
earnings yield, ROIC, RSI-14, Capital Allocation Score, and more.
"""

from __future__ import annotations

import math
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Generator, Optional

from finance_agent.data.fetcher import YFinanceFetcher
from finance_agent.models.schemas import Horizon
from finance_agent.quant.engine import QuantEngine
from finance_agent.quant.metrics import MetricsEngine
from finance_agent.utils.logger import logger


# ── Screener Result ────────────────────────────────────────────────────────────

@dataclass
class ScreenerResult:
    """Enriched per-ticker result with all factor data exposed for the UI."""

    ticker: str
    name:     str   = ""
    sector:   str   = ""
    industry: str   = ""

    # ── Core output ──────────────────────────────────────────────────────────
    screener_score: float = 0.0          # Final composite 0–10

    # ── Factor component scores (raw, pre-normalisation) ─────────────────────
    quality_score:       float | None = None   # QMJ-style composite  (0–10)
    capital_alloc_score: float | None = None   # ROIC/WACC + buybacks (0–10)

    # ── Value signals ─────────────────────────────────────────────────────────
    earnings_yield:  float | None = None   # EBIT / EV
    fcf_yield:       float | None = None
    price_to_book:   float | None = None

    # ── Momentum ──────────────────────────────────────────────────────────────
    momentum_12_1:   float | None = None   # 12-1 month AQR momentum
    return_1y:       float | None = None   # fallback
    rsi_14:          float | None = None   # Wilder RSI

    # ── Profitability ─────────────────────────────────────────────────────────
    roic:            float | None = None
    gross_margin:    float | None = None
    net_margin:      float | None = None

    # ── Safety / Quality flags ────────────────────────────────────────────────
    piotroski:       int   | None = None   # 0–9
    altman_z:        float | None = None
    beneish_m:       float | None = None   # >-1.78 = manipulation risk

    # ── Growth ────────────────────────────────────────────────────────────────
    rev_cagr_3y:     float | None = None
    eps_growth_yoy:  float | None = None

    # ── Risk ──────────────────────────────────────────────────────────────────
    sharpe_1y:       float | None = None
    volatility_90d:  float | None = None
    calmar_ratio:    float | None = None

    # ── Price / Market ────────────────────────────────────────────────────────
    current_price:   float | None = None
    market_cap:      float | None = None
    pe_ratio:        float | None = None
    forward_pe:      float | None = None

    # ── Error ─────────────────────────────────────────────────────────────────
    error: str | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


# ── Screener Engine ───────────────────────────────────────────────────────────

class ScreenerEngine:
    """
    Institutional-grade parallel quantitative screener.
    No LLM calls — pure deterministic factor model.
    """

    # 7-factor weights
    FACTOR_WEIGHTS: dict[str, float] = {
        "quality":       0.25,
        "value":         0.20,
        "momentum":      0.20,
        "profitability": 0.15,
        "safety":        0.10,
        "growth":        0.05,
        "risk_adj":      0.05,
    }

    # Minimum stocks per sector for within-sector z-scoring
    # (falls back to global z-score for smaller sectors)
    MIN_SECTOR_SIZE = 5

    def __init__(self, max_workers: int = 12) -> None:
        self.max_workers = max_workers
        self._fetcher = YFinanceFetcher()
        self._metrics = MetricsEngine()

    # ── Public API ────────────────────────────────────────────────────────────

    def scan_stream(
        self,
        tickers: list[str],
        top_n:   int     = 20,
        horizon: Horizon = Horizon.TWELVE_MONTHS,
    ) -> Generator[dict[str, Any], None, None]:
        """
        Generator that yields SSE-compatible progress and result events.

        Events:
          {"type": "start",    "total": N}
          {"type": "progress", "done": k, "total": N, "ticker": T, "ok": bool}
          {"type": "results",  "data": [...]}
        """
        total  = len(tickers)
        done   = 0
        # (ScreenerResult, raw_factors dict) tuples
        pairs: list[tuple[ScreenerResult, dict[str, float | None]]] = []

        yield {"type": "start", "total": total}

        # ── Pass 1: Parallel fetch + raw factor computation ───────────────────
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(self._score_ticker, t, horizon): t
                for t in tickers
            }

            for future in as_completed(futures):
                done  += 1
                result, raw_factors = future.result()
                pairs.append((result, raw_factors))

                yield {
                    "type":   "progress",
                    "done":   done,
                    "total":  total,
                    "ticker": result.ticker,
                    "ok":     result.error is None,
                }

        # ── Pass 2 + 3: Sector z-score + weighted composite ───────────────────
        valid_pairs = [(r, f) for r, f in pairs if r.error is None]
        error_pairs = [(r, f) for r, f in pairs if r.error is not None]

        self._apply_sector_normalisation(valid_pairs)

        all_results = [r for r, _ in valid_pairs] + [r for r, _ in error_pairs]
        ranked = sorted(
            [r for r in all_results if r.error is None],
            key=lambda r: r.screener_score,
            reverse=True,
        )[:top_n]

        yield {"type": "results", "data": [r.to_dict() for r in ranked]}

    # ── Pass 1: Per-ticker scoring ────────────────────────────────────────────

    def _score_ticker(
        self,
        ticker:  str,
        horizon: Horizon,
    ) -> tuple[ScreenerResult, dict[str, float | None]]:
        """Fetch data, compute MetricSet, extract all factor raw values."""
        result      = ScreenerResult(ticker=ticker)
        raw_factors: dict[str, float | None] = {}

        try:
            raw  = self._fetcher.fetch(ticker, price_period="2y")
            info = raw.info or {}

            result.name     = info.get("shortName", ticker)
            result.sector   = info.get("sector", "Unknown")
            result.industry = info.get("industry", "Unknown")

            # Full metric suite (no benchmark for speed)
            ms = self._metrics.compute(raw, benchmark_prices=None)

            # ── Populate result fields ────────────────────────────────────────
            result.current_price    = ms.get("current_price")
            result.market_cap       = ms.get("market_cap")
            result.pe_ratio         = ms.get("pe_ratio")
            result.forward_pe       = ms.get("forward_pe")
            result.piotroski        = ms.get("piotroski_f_score")
            result.altman_z         = ms.get("altman_z_score")
            result.beneish_m        = ms.get("beneish_m_score")
            result.roic             = ms.get("roic")
            result.gross_margin     = ms.get("gross_margin")
            result.net_margin       = ms.get("net_margin")
            result.fcf_yield        = ms.get("fcf_yield")
            result.earnings_yield   = ms.get("magic_formula_ey")
            result.price_to_book    = ms.get("price_to_book")
            result.momentum_12_1    = ms.get("momentum_12_1")
            result.return_1y        = ms.get("return_1y")
            result.rsi_14           = ms.get("rsi_14")
            result.rev_cagr_3y      = ms.get("revenue_cagr_3y")
            result.eps_growth_yoy   = ms.get("eps_growth_yoy")
            result.sharpe_1y        = ms.get("sharpe_ratio_1y")
            result.volatility_90d   = ms.get("volatility_90d")
            result.calmar_ratio     = ms.get("calmar_ratio")
            result.quality_score    = ms.get("quality_factor_score")
            result.capital_alloc_score = ms.get("capital_allocation_score")

            # ── Compute raw factor signals for z-scoring ──────────────────────
            raw_factors = self._extract_raw_factors(result, ms)

        except Exception as e:
            logger.warning(f"[Screener] Failed to score {ticker}: {e}")
            result.error = str(e)

        return result, raw_factors

    def _extract_raw_factors(
        self,
        r:  ScreenerResult,
        ms: Any,  # MetricSet
    ) -> dict[str, float | None]:
        """
        Convert MetricSet into 7 raw factor scores in their natural units.
        These will be z-scored cross-sectionally in Pass 2.
        """
        factors: dict[str, float | None] = {}

        # ── Quality (QMJ-style, already 0-10) ────────────────────────────────
        if r.quality_score is not None:
            factors["quality"] = r.quality_score

        # ── Value: avg of earnings yield, FCF yield, book-to-price ───────────
        value_components: list[float] = []
        if r.earnings_yield is not None:
            value_components.append(min(0.30, max(-0.10, r.earnings_yield)))
        if r.fcf_yield is not None:
            value_components.append(min(0.25, max(-0.05, r.fcf_yield)))
        if r.price_to_book and r.price_to_book > 0:
            bp = 1.0 / r.price_to_book          # book-to-price
            value_components.append(min(2.0, bp))
        if value_components:
            factors["value"] = sum(value_components) / len(value_components)

        # ── Momentum 12-1 (preferred) with return_1y fallback ─────────────────
        mom = r.momentum_12_1 if r.momentum_12_1 is not None else r.return_1y
        if mom is not None:
            factors["momentum"] = mom

        # ── Profitability: avg of ROIC, gross margin, net margin ──────────────
        prof_components: list[float] = []
        if r.roic is not None:
            prof_components.append(r.roic)
        if r.gross_margin is not None:
            prof_components.append(r.gross_margin)
        if r.net_margin is not None:
            prof_components.append(r.net_margin)
        if prof_components:
            factors["profitability"] = sum(prof_components) / len(prof_components)

        # ── Safety: Piotroski (normalised) + Altman (normalised) ─────────────
        safety_components: list[float] = []
        if r.piotroski is not None:
            safety_components.append(r.piotroski / 9.0)
        if r.altman_z is not None:
            z = r.altman_z
            if z >= 2.99:
                safety_components.append(1.0)
            elif z >= 1.81:
                safety_components.append(0.3 + (z - 1.81) / (2.99 - 1.81) * 0.7)
            else:
                safety_components.append(max(0.0, (z / 1.81) * 0.3))
        if safety_components:
            factors["safety"] = sum(safety_components) / len(safety_components)

        # ── Growth: avg of revenue CAGR 3Y, EPS growth, FCF growth ───────────
        growth_components: list[float] = []
        if r.rev_cagr_3y is not None:
            growth_components.append(r.rev_cagr_3y)
        if r.eps_growth_yoy is not None:
            growth_components.append(r.eps_growth_yoy)
        fcf_g = ms.get("fcf_growth_yoy")
        if fcf_g is not None:
            growth_components.append(fcf_g)
        if growth_components:
            factors["growth"] = sum(growth_components) / len(growth_components)

        # ── Risk-adjusted: Sharpe + low-vol bonus (BAB style) ─────────────────
        if r.sharpe_1y is not None:
            risk_adj = r.sharpe_1y
            vol = r.volatility_90d
            if vol is not None:
                # Low volatility premium: every 5% below 30% vol = +0.1 bonus
                bab_bonus = max(0.0, (0.30 - vol) * 2)
                risk_adj += bab_bonus
            factors["risk_adj"] = risk_adj

        return factors

    # ── Pass 2 + 3: Sector normalisation and composite scoring ────────────────

    def _apply_sector_normalisation(
        self,
        pairs: list[tuple[ScreenerResult, dict[str, float | None]]],
    ) -> None:
        """
        In-place update of ScreenerResult.screener_score for all valid tickers.

        Algorithm:
          1. Group ticker indices by sector.
          2. For each of the 7 factors, z-score within sector.
             If sector has < MIN_SECTOR_SIZE tickers with valid data,
             fall back to the global z-score for those tickers.
          3. Winsorise z-scores to [-3, +3].
          4. Weighted average of available z-scores.
          5. Map composite z ∈ [-3, +3] → score ∈ [0, 10].
        """
        n             = len(pairs)
        factors_list  = [f for _, f in pairs]
        results_list  = [r for r, _ in pairs]

        # Group indices by sector
        sector_indices: dict[str, list[int]] = defaultdict(list)
        for i, r in enumerate(results_list):
            sector_indices[r.sector].append(i)

        factor_names  = list(self.FACTOR_WEIGHTS.keys())
        normalised:   list[dict[str, float | None]] = [{} for _ in range(n)]

        for factor in factor_names:
            raw_values = [factors_list[i].get(factor) for i in range(n)]

            # Global z-scores (used as fallback for small sectors)
            global_z = QuantEngine.z_score_normalise(raw_values)

            for sector, indices in sector_indices.items():
                sector_vals = [raw_values[i] for i in indices]
                n_valid     = sum(1 for v in sector_vals if v is not None)

                if n_valid >= self.MIN_SECTOR_SIZE:
                    sector_z = QuantEngine.z_score_normalise(sector_vals)
                    for j, i in enumerate(indices):
                        normalised[i][factor] = sector_z[j]
                else:
                    # Fallback: use global z-score for this factor in small sectors
                    for i in indices:
                        normalised[i][factor] = global_z[i]

        # Combine with weights → composite z → 0-10 score
        for i, result in enumerate(results_list):
            z_dict = normalised[i]
            total_weight  = 0.0
            weighted_sum  = 0.0

            for factor, weight in self.FACTOR_WEIGHTS.items():
                z = z_dict.get(factor)
                if z is not None and not (isinstance(z, float) and math.isnan(z)):
                    z_clamped     = max(-3.0, min(3.0, z))   # Winsorise
                    weighted_sum += weight * z_clamped
                    total_weight += weight

            if total_weight > 0:
                composite_z       = weighted_sum / total_weight
                # Linear map: z=-3 → 0, z=0 → 5, z=+3 → 10
                score             = 5.0 + (composite_z / 3.0) * 5.0
                result.screener_score = round(max(0.0, min(10.0, score)), 2)
            else:
                result.screener_score = 0.0
