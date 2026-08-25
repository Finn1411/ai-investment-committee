"""
Finance Agent CLI — Week 4.

Usage:
    python -m finance_agent analyze AAPL
    python -m finance_agent analyze AAPL --horizon 3M
    python -m finance_agent analyze AAPL MSFT NVDA --horizon 12M --json
    python -m finance_agent analyze AAPL --no-persist
    python -m finance_agent journal           # show prediction journal stats
    python -m finance_agent journal pending   # show predictions due for review
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _cmd_analyze(args: argparse.Namespace) -> int:
    """Run full pipeline + committee for one or more tickers."""
    from finance_agent.agents.committee import CommitteeEngine
    from finance_agent.agents.portfolio_manager import PortfolioContext
    from finance_agent.data.pipeline import DataPipeline
    from finance_agent.models.schemas import Horizon
    from finance_agent.reporting.report import ReportFormatter
    from finance_agent.utils.logger import logger, setup_logger

    setup_logger()

    horizon_map = {
        "3M":   Horizon.THREE_MONTHS,
        "12M":  Horizon.TWELVE_MONTHS,
        "3-5Y": Horizon.THREE_FIVE_YEARS,
    }
    horizon = horizon_map.get(args.horizon, Horizon.TWELVE_MONTHS)

    portfolio = PortfolioContext()

    pipeline = DataPipeline(
        horizon=horizon,
        persist_to_db=args.persist,
    )
    committee = CommitteeEngine(
        portfolio=portfolio,
        persist_to_db=args.persist,
        log_to_journal=args.persist,
    )

    exit_code = 0
    for ticker in args.tickers:
        print(f"\n[*] Fetching data for {ticker.upper()}...")
        try:
            context = pipeline.run(ticker)
        except Exception as e:
            print(f"[ERROR] Failed to fetch data for {ticker}: {e}")
            exit_code = 1
            continue

        print(f"[*] Running committee analysis ({len(args.tickers)} tickers)...")
        try:
            result = committee.run(context, horizon=horizon)
        except Exception as e:
            print(f"[ERROR] Committee failed for {ticker}: {e}")
            exit_code = 1
            continue

        formatter = ReportFormatter(result)

        if args.json:
            # JSON-only mode
            import json
            print(json.dumps(formatter.to_dict(), indent=2))
        else:
            formatter.print_terminal()

        if args.export_json:
            path = formatter.export_json()
            print(f"[+] JSON exported → {path}")

        if args.export_md:
            path = formatter.export_markdown()
            print(f"[+] Markdown exported → {path}")

    return exit_code


def _cmd_journal(args: argparse.Namespace) -> int:
    """Show prediction journal statistics."""
    from finance_agent.evaluation.journal import PredictionJournal
    from finance_agent.utils.logger import setup_logger

    setup_logger()
    journal = PredictionJournal()

    if args.subcommand == "pending":
        pending = journal.get_pending_reviews()
        if not pending:
            print("No predictions pending review.")
        else:
            print(f"\n{'='*60}")
            print(f"  PREDICTIONS PENDING REVIEW ({len(pending)} total)")
            print(f"{'='*60}")
            for p in pending:
                print(
                    f"  #{p['seq']:>3}  {p['ticker']:<8}  "
                    f"analysed {p['analysis_date']}  "
                    f"({p['days_elapsed']}d ago)"
                )
        return 0

    stats = journal.summary_stats()
    print(f"\n{'='*60}")
    print(f"  PREDICTION JOURNAL STATS")
    print(f"{'='*60}")
    print(f"  Total predictions:  {stats['total_predictions']}")
    print(f"  Reviewed:           {stats['total_reviews']}")
    print(f"  Correct calls:      {stats['correct_calls']}")
    hit = stats['hit_rate']
    print(f"  Hit rate:           {f'{hit:.1%}' if hit is not None else 'N/A (no reviews yet)'}")
    print()
    return 0


def _cmd_backtest(args: argparse.Namespace) -> int:
    """Run backtesting and calibration."""
    from finance_agent.evaluation.backtest import BacktestEngine
    from finance_agent.utils.logger import setup_logger

    setup_logger()
    engine = BacktestEngine()

    if args.resolve:
        try:
            review = engine.resolve_prediction(args.resolve, actual_return=args.actual_return)
            print(f"[+] Successfully resolved {args.resolve}")
            print(f"    Actual Return: {review.actual_return:+.2%}")
            print(f"    Alpha vs Benchmark: {review.alpha:+.2%}")
            print(f"    Rating Correct: {review.rating_correct}")
        except Exception as e:
            print(f"[ERROR] Failed to resolve: {e}")
            return 1
        return 0

    if args.stats:
        stats = engine.get_stats()
        if stats.get("total_resolved", 0) == 0:
            print("No resolved predictions yet.")
            return 0
            
        print(f"\n{'='*60}")
        print(f"  BACKTESTING & CALIBRATION STATS")
        print(f"{'='*60}")
        print(f"  Total Resolved: {stats['total_resolved']}")
        print(f"  Overall Hit Rate: {stats['hit_rate']:.1%}")
        print(f"  Brier Score (Calibration): {stats['brier_score']:.4f}")
        print("\n  [Hit Rate by Rating]")
        for rating, r_stats in stats['by_rating'].items():
            hit = r_stats['hit_rate']
            total = r_stats['total']
            if total > 0:
                print(f"    {rating:<5}: {hit:.1%} ({r_stats['correct']}/{total})")
        print()
        return 0
        
    print("Please specify --resolve <id> or --stats")
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="finance_agent",
        description="Finance Agent — Professional Investment Research System",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── analyze ──────────────────────────────────────────────────────────────
    analyze = subparsers.add_parser("analyze", help="Run full analysis on one or more tickers")
    analyze.add_argument(
        "tickers", nargs="+", type=str,
        help="Ticker symbol(s) to analyse, e.g. AAPL MSFT NVDA"
    )
    analyze.add_argument(
        "--horizon", choices=["3M", "12M", "3-5Y"], default="12M",
        help="Investment horizon (default: 12M)"
    )
    analyze.add_argument(
        "--json", action="store_true",
        help="Print raw JSON output instead of terminal report"
    )
    analyze.add_argument(
        "--export-json", action="store_true",
        help="Export report as JSON to reports/ directory"
    )
    analyze.add_argument(
        "--export-md", action="store_true",
        help="Export report as Markdown to reports/ directory"
    )
    analyze.add_argument(
        "--no-persist", dest="persist", action="store_false", default=True,
        help="Do not persist results to DB or prediction journal"
    )

    # ── journal ───────────────────────────────────────────────────────────────
    journal = subparsers.add_parser("journal", help="Prediction journal management")
    journal.add_argument(
        "subcommand", nargs="?", default="stats",
        choices=["stats", "pending"],
        help="'stats' (default) or 'pending' to see predictions due for review"
    )

    # ── backtest ──────────────────────────────────────────────────────────────
    backtest = subparsers.add_parser("backtest", help="Evaluate predictions against real data")
    backtest.add_argument("--resolve", type=str, help="Resolve a prediction by ID or ticker")
    backtest.add_argument("--actual-return", type=float, help="Explicit actual return (e.g., 0.12 for 12%%) if not using yfinance")
    backtest.add_argument("--stats", action="store_true", help="Show system calibration stats")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "analyze":
        return _cmd_analyze(args)
    elif args.command == "journal":
        return _cmd_journal(args)
    elif args.command == "backtest":
        return _cmd_backtest(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
