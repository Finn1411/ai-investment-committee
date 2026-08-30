"""
Data Pipeline -- the central orchestrator for Week 2.

Flow:
  1. Fetch raw data (YFinanceFetcher)
  2. Validate data quality (DataQualityAgent)
  3. Compute all 45 metrics (MetricsEngine)
  4. Build valuation model (ValuationEngine)
  5. Build scenarios (ScenarioBuilder)
  6. Persist to database
  7. Return AnalysisContext for agents

Usage:
    ctx = DataPipeline().run("AAPL")
    print(ctx.metrics.pe_ratio)
    print(ctx.scenarios.bear.expected_return)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

from finance_agent.agents.data_quality_agent import DataQualityAgent, DataQualityReport
from finance_agent.data.fetcher import RawTickerData, YFinanceFetcher
from finance_agent.database.db import (
    FundamentalsORM,
    MarketDataORM,
    StockORM,
    ValuationORM,
    get_session,
    init_db,
)
from finance_agent.models.schemas import Horizon
from finance_agent.quant.metrics import MetricSet, MetricsEngine
from finance_agent.quant.scenarios import ScenarioBuildResult, build_scenarios_from_metrics
from finance_agent.quant.valuation import ValuationEngine, ValuationSummary
from finance_agent.rag.ingestor import NewsIngestor
from finance_agent.rag.engine import RAGEngine
from finance_agent.data.macro import MacroEngine, MacroRegime
from finance_agent.data.peers import PeerEngine
from finance_agent.utils.config import settings
from finance_agent.utils.logger import logger


# -- Context container ---------------------------------------------------------

@dataclass
class AnalysisContext:
    """
    Everything agents need to do their analysis.
    Passed to all agents; they do NOT touch raw data directly.
    """
    ticker: str
    company_name: str
    sector: str
    industry: str
    currency: str
    exchange: str
    analysis_date: date

    raw: RawTickerData
    quality_report: DataQualityReport
    metrics: MetricSet
    valuation: Optional[ValuationSummary]
    scenarios: Optional[ScenarioBuildResult]
    
    benchmark_ticker: str
    
    macro_regime_summary: str = ""
    peer_context: str = ""
    
    # RAG Claims (Phase 8)
    claims: list[dict] = field(default_factory=list)

    # Quick-access summary dict for LLM prompts
    _summary_cache: Optional[dict] = field(default=None, repr=False)

    @property
    def data_passed_quality_check(self) -> bool:
        return self.quality_report.passed

    def to_prompt_dict(self) -> dict:
        """Returns a clean dict suitable for injecting into LLM prompts."""
        if self._summary_cache:
            return self._summary_cache

        m = self.metrics.metrics
        d = {
            "ticker": self.ticker,
            "company": self.company_name,
            "sector": self.sector,
            "analysis_date": str(self.analysis_date),
            # Market
            "current_price": m.get("current_price"),
            "market_cap_bn": round(m.get("market_cap", 0) / 1e9, 2) if m.get("market_cap") else None,
            "return_ytd": m.get("return_ytd"),
            "return_1y": m.get("return_1y"),
            "beta": m.get("beta"),
            "volatility_annual": m.get("volatility_90d"),
            "max_drawdown_1y": m.get("max_drawdown_1y"),
            "sharpe_ratio_1y": m.get("sharpe_ratio_1y"),
            # Profitability
            "gross_margin": m.get("gross_margin"),
            "operating_margin": m.get("operating_margin"),
            "net_margin": m.get("net_margin"),
            "roic": m.get("roic"),
            "roe": m.get("roe"),
            "fcf_margin": m.get("fcf_margin"),
            "fcf_yield": m.get("fcf_yield"),
            "cash_conversion": m.get("cash_conversion"),
            # Balance sheet
            "net_debt_to_ebitda": m.get("net_debt_to_ebitda"),
            "interest_coverage": m.get("interest_coverage"),
            "current_ratio": m.get("current_ratio"),
            # Growth
            "revenue_growth_yoy": m.get("revenue_growth_yoy"),
            "eps_growth_yoy": m.get("eps_growth_yoy"),
            "fcf_growth_yoy": m.get("fcf_growth_yoy"),
            "revenue_cagr_3y": m.get("revenue_cagr_3y"),
            # Valuation
            "pe_ratio": m.get("pe_ratio"),
            "forward_pe": m.get("forward_pe"),
            "ev_to_ebitda": m.get("ev_to_ebitda"),
            "price_to_sales": m.get("price_to_sales"),
            "pe_percentile_5y": m.get("pe_percentile_5y"),
            "fcf_yield_pct": round(m.get("fcf_yield", 0) * 100, 2) if m.get("fcf_yield") else None,
            # Composite scores
            "piotroski_f_score": m.get("piotroski_f_score"),
            "altman_z_score": m.get("altman_z_score"),
            "earnings_quality_score": m.get("earnings_quality_score"),
            "overall_quality_score": m.get("overall_quality_score"),
        }

        # Valuation summary
        if self.valuation:
            d["valuation_label"] = self.valuation.valuation_label
            d["valuation_score"] = self.valuation.composite_valuation_score
            if self.valuation.dcf:
                d["dcf_intrinsic_per_share"] = self.valuation.dcf.intrinsic_value_per_share
                d["dcf_margin_of_safety"] = self.valuation.dcf.margin_of_safety
                d["dcf_upside"] = self.valuation.dcf.upside_downside
            if self.valuation.reverse_dcf:
                d["implied_fcf_growth"] = self.valuation.reverse_dcf.implied_fcf_growth
                d["reverse_dcf_narrative"] = self.valuation.reverse_dcf.narrative

        # Scenario summary
        if self.scenarios:
            sm = self.scenarios.model
            d["scenario_bear_prob"] = sm.bear.probability
            d["scenario_bear_return"] = sm.bear.expected_return
            d["scenario_base_prob"] = sm.base.probability
            d["scenario_base_return"] = sm.base.expected_return
            d["scenario_bull_prob"] = sm.bull.probability
            d["scenario_bull_return"] = sm.bull.expected_return
            d["scenario_expected_value"] = round(sm.expected_value, 4)
            d["scenario_prob_positive"] = sm.prob_outperform

        # Quality warnings
        d["data_quality_warnings"] = self.quality_report.warnings
        d["estimated_fields"] = self.quality_report.estimated_fields

        self._summary_cache = d
        return d


# -- Pipeline ------------------------------------------------------------------

class DataPipeline:
    """
    Central orchestrator. Call .run(ticker) to get an AnalysisContext.
    """

    def __init__(
        self,
        benchmark_ticker: str = "^GSPC",
        horizon: Horizon = Horizon.TWELVE_MONTHS,
        persist_to_db: bool = True,
    ) -> None:
        self.benchmark_ticker = benchmark_ticker
        self.horizon = horizon
        self.persist_to_db = persist_to_db

        self._fetcher = YFinanceFetcher()
        self._dqa = DataQualityAgent()
        self._metrics = MetricsEngine(benchmark_ticker=benchmark_ticker)
        self._valuation = ValuationEngine()

        if persist_to_db:
            init_db()

    def run(self, ticker: str, price_period: str = "2y") -> AnalysisContext:
        """
        Full pipeline run for a single ticker.

        Returns:
            AnalysisContext with metrics, valuation, scenarios, and quality report.
        """
        ticker = ticker.upper()
        logger.info(f"[Pipeline] ====== Starting pipeline for {ticker} ======")

        # -- 1. Fetch raw data -------------------------------------------------
        raw = self._fetcher.fetch(ticker, price_period=price_period)

        # Also fetch benchmark (shared to avoid double API call in metrics)
        logger.info(f"[Pipeline] Fetching benchmark {self.benchmark_ticker}")
        bm_raw = self._fetcher.fetch(self.benchmark_ticker, price_period=price_period)
        benchmark_prices = bm_raw.price_history

        # -- 2. Data Quality Validation ----------------------------------------
        logger.info(f"[Pipeline] Running data quality checks for {ticker}")
        flat_data = self._flatten_for_dqa(raw)
        quality_report = self._dqa.validate(ticker, flat_data)

        if not quality_report.passed:
            logger.warning(
                f"[Pipeline] {ticker} data quality FAILED -- "
                f"{len(quality_report.errors)} error(s). Proceeding with caution."
            )

        # -- 3. Compute metrics ------------------------------------------------
        logger.info(f"[Pipeline] Computing metrics for {ticker}")
        metrics = self._metrics.compute(raw, benchmark_prices=benchmark_prices)

        # -- 4. Build valuation model ------------------------------------------
        valuation = self._build_valuation(ticker, metrics, raw)

        # -- 5. Build scenarios ------------------------------------------------
        scenarios = None
        # 5. Build Scenarios
        try:
            scenarios = build_scenarios_from_metrics(
                ticker=ticker,
                metrics=metrics,
                horizon=self.horizon,
            )
        except Exception as e:
            logger.warning(f"[Pipeline] Scenario build failed: {e}")
        
        # 6. Fetch Macro & Peer Data
        logger.info(f"[Pipeline] Fetching Macro & Peer Context for {ticker}")
        macro_engine = MacroEngine()
        regime = macro_engine.fetch_regime()
        macro_summary = f"Risk Environment: {regime.risk_environment} | VIX: {regime.vix_level:.2f} | 10Y Yield: {regime.treasury_10y_yield:.2f}%"
        
        peer_engine = PeerEngine()
        peer_context = peer_engine.fetch_peer_context(ticker)

        # -- 6. Persist to DB --------------------------------------------------
        if self.persist_to_db:
            self._persist(ticker, raw, metrics)

        # -- 7. RAG Ingestion & Retrieval (Phase 8) ----------------------------
        logger.info(f"[Pipeline] Running RAG ingestion for {ticker}...")
        ingestor = NewsIngestor()
        ingestor.ingest_ticker_news(ticker)
        
        rag_engine = RAGEngine()
        # Retrieve top claims to provide context
        claims = rag_engine.query_claims(query=f"Recent developments, financials, and news for {ticker}", ticker=ticker, n_results=10)

        # -- 7. Assemble context -----------------------------------------------
        info = raw.info
        ctx = AnalysisContext(
            ticker=ticker,
            company_name=info.get("longName") or info.get("shortName") or ticker,
            sector=info.get("sector") or "Unknown",
            industry=info.get("industry") or "Unknown",
            currency=info.get("currency") or "USD",
            exchange=info.get("exchange") or "Unknown",
            analysis_date=date.today(),
            raw=raw,
            quality_report=quality_report,
            metrics=metrics,
            valuation=valuation,
            scenarios=scenarios,
            benchmark_ticker=self.benchmark_ticker,
            claims=claims,
            macro_regime_summary=macro_summary,
            peer_context=peer_context,
        )

        ev_str = f"{scenarios.model.expected_value:.1%}" if scenarios else "N/A"
        logger.info(
            f"[Pipeline] --- {ticker} complete --- "
            f"metrics={metrics.available_count} | "
            f"quality={'OK' if quality_report.passed else 'WARN'} | "
            f"valuation={valuation.valuation_label if valuation else 'N/A'} | "
            f"EV={ev_str}"
        )

        return ctx

    def run_batch(self, tickers: list[str], **kwargs) -> dict[str, AnalysisContext]:
        """Run the pipeline for multiple tickers."""
        results = {}
        for ticker in tickers:
            try:
                results[ticker] = self.run(ticker, **kwargs)
            except Exception as e:
                logger.error(f"[Pipeline] {ticker} failed: {e}")
        return results

    # -- Private helpers -------------------------------------------------------

    def _flatten_for_dqa(self, raw: RawTickerData) -> dict:
        """Flatten raw data into a single dict for DataQualityAgent."""
        info = raw.info
        get = YFinanceFetcher.get_statement_row
        latest = YFinanceFetcher.latest_value

        revenue = latest(get(raw.income_stmt, "Total Revenue"))
        gross_profit = latest(get(raw.income_stmt, "Gross Profit"))
        ebit = latest(get(raw.income_stmt, "EBIT", "Operating Income"))
        net_income = latest(get(raw.income_stmt, "Net Income"))
        op_cf = latest(get(raw.cash_flow, "Operating Cash Flow", "Total Cash From Operating Activities"))
        capex = latest(get(raw.cash_flow, "Capital Expenditure", "Capital Expenditures"))
        close = raw.price_history["Close"].iloc[-1] if raw.price_history is not None and not raw.price_history.empty else None

        flat = {
            "close": close,
            "volume": raw.price_history["Volume"].iloc[-1] if raw.price_history is not None and not raw.price_history.empty else None,
            "market_cap": info.get("marketCap"),
            "revenue": revenue,
            "gross_profit": gross_profit,
            "gross_margin": (gross_profit / revenue) if gross_profit and revenue else info.get("grossMargins"),
            "operating_margin": (ebit / revenue) if ebit and revenue else info.get("operatingMargins"),
            "net_margin": (net_income / revenue) if net_income and revenue else info.get("profitMargins"),
            "free_cash_flow": (op_cf + capex) if op_cf and capex else info.get("freeCashflow"),
            "fcf_yield": info.get("freeCashflow", 0) / info.get("marketCap", 1) if info.get("marketCap") else None,
            "eps_diluted": info.get("trailingEps"),
            "pe_ratio": info.get("trailingPE"),
            "beta": info.get("beta"),
            "current_ratio": info.get("currentRatio"),
            "net_debt_to_ebitda": None,  # computed in metrics
            "last_updated": str(date.today()),
        }
        return flat

    def _build_valuation(
        self, ticker: str, metrics: MetricSet, raw: RawTickerData
    ) -> Optional[ValuationSummary]:
        try:
            fcf = metrics.get("free_cash_flow")
            shares = metrics.get("shares_outstanding")
            ev = metrics.get("enterprise_value")
            price = metrics.get("current_price")
            net_debt = metrics.get("net_debt")

            if not all([fcf, shares, ev, price]) or fcf <= 0 or shares <= 0:
                logger.warning(f"[Pipeline] {ticker}: insufficient data for valuation model")
                return None

            return self._valuation.build(
                ticker=ticker,
                current_price=price,
                fcf=fcf,
                shares_outstanding=shares,
                enterprise_value=ev,
                net_debt=net_debt or 0,
                pe_percentile=metrics.get("pe_percentile_5y"),
                ps_percentile=metrics.get("ps_percentile_2y"),
            )
        except Exception as e:
            logger.warning(f"[Pipeline] {ticker} valuation build failed: {e}")
            return None

    def _persist(self, ticker: str, raw: RawTickerData, metrics: MetricSet) -> None:
        """Save/update stock info, latest market data row, and latest fundamentals to DB."""
        try:
            with get_session() as session:
                # -- Stock metadata (only update if already in watchlist) -------
                info = raw.info
                stock = session.query(StockORM).filter_by(ticker=ticker).first()
                if stock:
                    # Update metadata for existing watchlist entries only
                    stock.name = info.get("longName") or info.get("shortName") or stock.name
                    stock.sector = info.get("sector") or stock.sector
                    stock.industry = info.get("industry") or stock.industry
                    stock.country = info.get("country") or stock.country
                    stock.currency = info.get("currency") or stock.currency
                    stock.exchange = info.get("exchange") or stock.exchange
                # NOTE: do NOT create a new StockORM here — that would silently
                # add every analyzed stock to the watchlist.

                # -- Latest price row ------------------------------------------
                if raw.price_history is not None and not raw.price_history.empty:
                    latest_row = raw.price_history.iloc[-1]
                    latest_date = str(raw.price_history.index[-1].date())
                    existing = session.query(MarketDataORM).filter_by(
                        ticker=ticker, date=latest_date
                    ).first()
                    if not existing:
                        session.add(MarketDataORM(
                            ticker=ticker,
                            date=latest_date,
                            open=float(latest_row.get("Open", 0)),
                            high=float(latest_row.get("High", 0)),
                            low=float(latest_row.get("Low", 0)),
                            close=float(latest_row.get("Close", 0)),
                            adj_close=float(latest_row.get("Close", 0)),
                            volume=int(latest_row.get("Volume", 0)),
                        ))

                # -- Fundamentals snapshot -------------------------------------
                session.add(FundamentalsORM(
                    ticker=ticker,
                    fiscal_year=date.today().year,
                    report_date=str(date.today()),
                    gross_margin=metrics.get("gross_margin"),
                    operating_margin=metrics.get("operating_margin"),
                    net_margin=metrics.get("net_margin"),
                    roic=metrics.get("roic"),
                    roe=metrics.get("roe"),
                    free_cash_flow=metrics.get("free_cash_flow"),
                    fcf_margin=metrics.get("fcf_margin"),
                    net_debt_to_ebitda=metrics.get("net_debt_to_ebitda"),
                    interest_coverage=metrics.get("interest_coverage"),
                    current_ratio=metrics.get("current_ratio"),
                    revenue=YFinanceFetcher.latest_value(
                        YFinanceFetcher.get_statement_row(raw.income_stmt, "Total Revenue")
                    ),
                    net_income=YFinanceFetcher.latest_value(
                        YFinanceFetcher.get_statement_row(raw.income_stmt, "Net Income")
                    ),
                    eps_diluted=info.get("trailingEps"),
                    revenue_growth_yoy=metrics.get("revenue_growth_yoy"),
                    eps_growth_yoy=metrics.get("eps_growth_yoy"),
                ))

                # -- Valuation snapshot ----------------------------------------
                session.add(ValuationORM(
                    ticker=ticker,
                    as_of=str(date.today()),
                    pe_ratio=metrics.get("pe_ratio"),
                    forward_pe=metrics.get("forward_pe"),
                    ev_to_ebit=metrics.get("ev_to_ebit"),
                    ev_to_ebitda=metrics.get("ev_to_ebitda"),
                    price_to_sales=metrics.get("price_to_sales"),
                    price_to_book=metrics.get("price_to_book"),
                    fcf_yield=metrics.get("fcf_yield"),
                    enterprise_value=metrics.get("enterprise_value"),
                    market_cap=metrics.get("market_cap"),
                ))

                session.commit()
                logger.debug(f"[Pipeline] {ticker} persisted to DB")
        except Exception as e:
            logger.error(f"[Pipeline] DB persist failed for {ticker}: {e}")

