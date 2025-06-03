#!/usr/bin/env python3
"""
Ecosystem Mapping Module Demo

This script demonstrates the ecosystem mapping functionality by analyzing
peer relationships and generating ecosystem-based trading signals.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

import json

from rich import print as rprint
from rich.console import Console
from rich.table import Table

from buffetbot.analysis.ecosystem import EcosystemAnalyzer
from buffetbot.data.peer_fetcher import PeerFetchError, get_peer_info, get_peers
from buffetbot.utils.logger import get_logger, setup_logger

# Setup logging
setup_logger()
logger = get_logger(__name__)
console = Console()


def demo_peer_fetching():
    """Demonstrate peer fetching functionality."""
    console.print("\n[bold blue]🔍 Peer Fetching Demo[/bold blue]")

    test_tickers = ["NVDA", "AAPL", "TSLA", "JPM", "UNKNOWN_TICKER"]

    for ticker in test_tickers:
        try:
            console.print(f"\n[yellow]Getting peers for {ticker}...[/yellow]")
            peers = get_peers(ticker)

            console.print(f"[green]✅ Found {len(peers)} peers for {ticker}:[/green]")
            rprint(f"  {', '.join(peers)}")

            # Get detailed peer info for first few peers
            if peers:
                console.print(
                    f"\n[yellow]Getting detailed info for {ticker} peers...[/yellow]"
                )
                peer_info = get_peer_info(ticker)

                for i, peer in enumerate(peer_info[:3]):  # Show first 3
                    company_name = peer.name or "N/A"
                    sector = peer.sector or "N/A"
                    console.print(f"  {peer.ticker}: {company_name} ({sector})")

        except PeerFetchError as e:
            console.print(f"[red]❌ Error for {ticker}: {str(e)}[/red]")
        except Exception as e:
            console.print(f"[red]❌ Unexpected error for {ticker}: {str(e)}[/red]")


def demo_ecosystem_analysis():
    """Demonstrate ecosystem analysis functionality."""
    console.print("\n[bold blue]📊 Ecosystem Analysis Demo[/bold blue]")

    # Initialize analyzer
    analyzer = EcosystemAnalyzer()

    # Test with popular tickers
    test_tickers = ["NVDA", "AAPL", "TSLA"]

    for ticker in test_tickers:
        try:
            console.print(f"\n[yellow]Analyzing ecosystem for {ticker}...[/yellow]")

            # Perform analysis
            analysis = analyzer.analyze_ecosystem(ticker)

            # Display results
            console.print(f"[green]✅ Analysis complete for {ticker}[/green]")

            # Create results table
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            summary = analyzer.get_ecosystem_summary(analysis)

            table.add_row("Ticker", summary["ticker"])
            table.add_row("Peer Count", str(summary["peer_count"]))
            table.add_row("Ecosystem Score", f"{summary['ecosystem_score']:.3f}")
            table.add_row("Avg Correlation", f"{summary['avg_correlation']:.3f}")
            table.add_row("Momentum Signal", summary["momentum_signal"])
            table.add_row("Avg RSI", f"{summary['avg_rsi']:.1f}")
            table.add_row("Volatility Regime", summary["volatility_regime"])
            table.add_row("Recommendation", summary["recommendation"])
            table.add_row("Signal Strength", f"{summary['signal_strength']:.3f}")
            table.add_row("Confidence", f"{summary['confidence']:.3f}")
            table.add_row("Strongest Correlation", summary["strongest_correlation"])

            console.print(table)

            # Show top correlations
            correlations = analysis.individual_correlations
            if correlations:
                console.print(f"\n[bold]Top Correlations with {ticker}:[/bold]")
                sorted_corrs = sorted(
                    correlations.items(),
                    key=lambda x: abs(x[1].correlation),
                    reverse=True,
                )

                for peer, corr_result in sorted_corrs[:5]:  # Top 5
                    strength = corr_result.strength_category
                    significance = "✓" if corr_result.is_significant else "✗"
                    console.print(
                        f"  {peer}: {corr_result.correlation:+.3f} "
                        f"({strength}, sig: {significance})"
                    )

        except Exception as e:
            console.print(f"[red]❌ Analysis failed for {ticker}: {str(e)}[/red]")
            logger.error(f"Analysis failed for {ticker}", exc_info=True)


def demo_custom_peers():
    """Demonstrate custom peer analysis."""
    console.print("\n[bold blue]🎯 Custom Peers Demo[/bold blue]")

    analyzer = EcosystemAnalyzer()

    # Define custom peer groups
    custom_scenarios = {
        "NVDA": ["AMD", "INTC", "TSM", "AVGO"],  # GPU/Semiconductor focus
        "TSLA": ["RIVN", "LCID", "F", "GM"],  # EV focus
        "AAPL": ["MSFT", "GOOGL", "META"],  # Big Tech focus
    }

    for ticker, custom_peers in custom_scenarios.items():
        try:
            console.print(
                f"\n[yellow]Analyzing {ticker} with custom peers: {custom_peers}[/yellow]"
            )

            # Analyze with custom peers
            analysis = analyzer.analyze_ecosystem(ticker, custom_peers=custom_peers)

            console.print(f"[green]✅ Custom analysis complete for {ticker}[/green]")
            console.print(f"  Recommendation: {analysis.recommendation}")
            console.print(f"  Signal Strength: {analysis.signal_strength:.3f}")
            console.print(f"  Confidence: {analysis.confidence:.3f}")
            console.print(
                f"  Ecosystem Score: {analysis.ecosystem_score.normalized_score:.3f}"
            )

        except Exception as e:
            console.print(f"[red]❌ Custom analysis failed for {ticker}: {str(e)}[/red]")


def demo_multiple_analysis():
    """Demonstrate batch analysis of multiple tickers."""
    console.print("\n[bold blue]📈 Batch Analysis Demo[/bold blue]")

    analyzer = EcosystemAnalyzer()
    tickers = ["NVDA", "AMD", "INTC"]

    console.print(f"[yellow]Analyzing multiple tickers: {tickers}[/yellow]")

    try:
        results = analyzer.analyze_multiple_tickers(tickers)

        console.print(
            f"[green]✅ Batch analysis complete for {len(results)} tickers[/green]"
        )

        # Create comparison table
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Ticker", style="cyan")
        table.add_column("Recommendation", style="green")
        table.add_column("Score", style="yellow")
        table.add_column("Confidence", style="blue")
        table.add_column("Momentum", style="red")

        for ticker, analysis in results.items():
            table.add_row(
                ticker,
                analysis.recommendation,
                f"{analysis.signal_strength:.3f}",
                f"{analysis.confidence:.3f}",
                analysis.ecosystem_metrics.momentum_signal,
            )

        console.print(table)

    except Exception as e:
        console.print(f"[red]❌ Batch analysis failed: {str(e)}[/red]")


def main():
    """Run all ecosystem mapping demos."""
    console.print("[bold green]🌐 Ecosystem Mapping Module Demonstration[/bold green]")
    console.print(
        "This demo showcases the peer fetching and ecosystem analysis capabilities."
    )

    try:
        # Run demos
        demo_peer_fetching()
        demo_ecosystem_analysis()
        demo_custom_peers()
        demo_multiple_analysis()

        console.print("\n[bold green]🎉 Demo completed successfully![/bold green]")
        console.print(
            "\nThe ecosystem mapping module is ready to enhance your options recommendations"
        )
        console.print("by considering how related stocks are performing.")

    except KeyboardInterrupt:
        console.print("\n[yellow]Demo interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Demo failed with error: {str(e)}[/red]")
        logger.error("Demo failed", exc_info=True)


if __name__ == "__main__":
    main()
