#!/usr/bin/env python3
"""
BuffetBot CLI - Command Line Interface for Financial Analysis

This module provides a command-line interface for the BuffetBot financial analysis
tools, enabling users to access recommendation features directly from the terminal.
"""

import csv
import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from buffetbot.analysis.options_advisor import (
    CalculationError,
    InsufficientDataError,
    OptionsAdvisorError,
    recommend_long_calls,
)
from buffetbot.utils.logger import setup_logger
from buffetbot.utils.validators import validate_ticker

# Initialize CLI components
app = typer.Typer(
    name="buffetbot",
    help="BuffetBot CLI - Financial Analysis Tools",
    add_completion=False,
)
console = Console()
logger = setup_logger(__name__, "logs/cli.log")


def validate_ticker_format(ticker: str) -> str:
    """
    Validate and format ticker symbol.

    Args:
        ticker: Raw ticker input from user

    Returns:
        str: Formatted ticker symbol

    Raises:
        typer.BadParameter: If ticker format is invalid
    """
    if not ticker:
        raise typer.BadParameter("Ticker cannot be empty")

    ticker = ticker.upper().strip()

    if not validate_ticker(ticker):
        raise typer.BadParameter(
            f"Invalid ticker format '{ticker}'. "
            "Ticker must be alphanumeric and 1-10 characters long."
        )

    return ticker


def validate_positive_int(value: int, param_name: str) -> int:
    """
    Validate that an integer parameter is positive.

    Args:
        value: Integer value to validate
        param_name: Name of the parameter for error messages

    Returns:
        int: Validated value

    Raises:
        typer.BadParameter: If value is not positive
    """
    if value <= 0:
        raise typer.BadParameter(f"{param_name} must be positive, got {value}")
    return value


def format_recommendations_table(df: pd.DataFrame) -> Table:
    """
    Format recommendations DataFrame as a Rich table for display.

    Args:
        df: Recommendations DataFrame from recommend_long_calls

    Returns:
        Table: Rich table ready for display
    """
    table = Table(
        title="📈 Long Call Options Recommendations",
        show_header=True,
        header_style="bold magenta",
    )

    # Add columns
    table.add_column("Rank", style="dim", width=4)
    table.add_column("Ticker", style="bold cyan")
    table.add_column("Strike ($)", justify="right", style="green")
    table.add_column("Expiry", style="blue")
    table.add_column("Last Price ($)", justify="right", style="yellow")
    table.add_column("IV (%)", justify="right")
    table.add_column("RSI", justify="right")
    table.add_column("Beta", justify="right")
    table.add_column("Momentum", justify="right")
    table.add_column("Score", justify="right", style="bold red")

    # Add rows
    for idx, row in df.iterrows():
        table.add_row(
            str(idx + 1),
            row["ticker"],
            f"{row['strike']:.2f}",
            row["expiry"].strftime("%Y-%m-%d")
            if hasattr(row["expiry"], "strftime")
            else str(row["expiry"]),
            f"{row['lastPrice']:.2f}",
            f"{row['IV']:.2%}",
            f"{row['RSI']:.1f}",
            f"{row['Beta']:.2f}",
            f"{row['Momentum']:.3f}",
            f"{row['CompositeScore']:.3f}",
        )

    return table


def export_to_csv(df: pd.DataFrame, filename: str, ticker: str) -> None:
    """
    Export recommendations DataFrame to CSV file.

    Args:
        df: Recommendations DataFrame
        filename: Output CSV filename
        ticker: Stock ticker for logging

    Raises:
        typer.Exit: If CSV export fails
    """
    try:
        # Ensure the directory exists
        filepath = Path(filename)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Format expiry column for CSV
        df_export = df.copy()
        if "expiry" in df_export.columns:
            df_export["expiry"] = df_export["expiry"].astype(str)

        # Export to CSV
        df_export.to_csv(filepath, index=False, float_format="%.4f")

        console.print(
            f"✅ Results exported to: {filepath.absolute()}", style="bold green"
        )
        logger.info(
            f"Successfully exported {len(df_export)} recommendations to {filepath}"
        )

    except Exception as e:
        error_msg = f"Failed to export CSV: {str(e)}"
        console.print(f"❌ {error_msg}", style="bold red")
        logger.error(f"CSV export failed for {ticker}: {error_msg}")
        raise typer.Exit(1)


@app.command()
def recommend(
    ticker: str = typer.Argument(..., help="Stock ticker symbol (e.g., AAPL, MSFT)"),
    min_days: int = typer.Option(
        180, "--min-days", help="Minimum days to expiry for options"
    ),
    top_n: int = typer.Option(
        5, "--top-n", help="Number of top recommendations to return"
    ),
    output: str
    | None = typer.Option(None, "--output", help="Export results to CSV file"),
) -> None:
    """
    Recommend long-dated call options using comprehensive technical analysis.

    This command analyzes long-dated call options for the specified stock ticker and
    provides ranked recommendations based on a composite scoring system that combines:
    - RSI (Relative Strength Index)
    - Beta coefficient vs SPY
    - Price momentum indicators
    - Implied volatility analysis

    The scoring system identifies the most attractive option contracts by balancing
    technical strength of the underlying stock with option-specific factors like
    implied volatility and time to expiration.

    Args:
        ticker: Stock ticker symbol (will be validated and normalized)
        min_days: Minimum days to expiry for options to consider (default: 180)
        top_n: Number of top recommendations to return (default: 5)
        output: Optional CSV filename to export results

    Examples:
        # Get top 5 long-term call recommendations for Apple
        python cli.py recommend AAPL

        # Get top 3 recommendations with minimum 1 year to expiry
        python cli.py recommend MSFT --min-days 365 --top-n 3

        # Export results to CSV
        python cli.py recommend AAPL --output results/aapl_calls.csv
    """
    # Log the CLI action
    logger.info(
        f"CLI command started: recommend {ticker} --min-days {min_days} --top-n {top_n}"
    )

    try:
        # Validate and normalize inputs
        ticker = validate_ticker_format(ticker)
        min_days = validate_positive_int(min_days, "min_days")
        top_n = validate_positive_int(top_n, "top_n")

        # Display analysis start message
        with console.status(f"[bold green]Analyzing long call options for {ticker}..."):
            console.print(
                f"\n🔍 Starting options analysis for [bold cyan]{ticker}[/bold cyan]"
            )
            console.print(f"   • Minimum days to expiry: [yellow]{min_days}[/yellow]")
            console.print(f"   • Top recommendations: [yellow]{top_n}[/yellow]")

            # Call the recommendation function
            recommendations = recommend_long_calls(
                ticker=ticker, min_days=min_days, top_n=top_n
            )

        # Check if we got any recommendations
        if recommendations.empty:
            console.print(
                Panel(
                    f"No long call options found for {ticker} with minimum {min_days} days to expiry.",
                    title="⚠️  No Results",
                    border_style="yellow",
                )
            )
            logger.warning(
                f"No recommendations returned for {ticker} (min_days={min_days})"
            )
            raise typer.Exit(1)

        # Display results
        console.print("\n")
        table = format_recommendations_table(recommendations)
        console.print(table)

        # Display summary statistics
        avg_score = recommendations["CompositeScore"].mean()
        best_score = recommendations["CompositeScore"].max()
        console.print(
            f"\n📊 [bold]Summary:[/bold] Found [green]{len(recommendations)}[/green] recommendations"
        )
        console.print(f"   • Best score: [bold red]{best_score:.3f}[/bold red]")
        console.print(f"   • Average score: [yellow]{avg_score:.3f}[/yellow]")

        # Export to CSV if requested
        if output:
            export_to_csv(recommendations, output, ticker)

        logger.info(
            f"Successfully completed analysis for {ticker}. Returned {len(recommendations)} recommendations."
        )

    except typer.BadParameter as e:
        console.print(f"❌ Input validation error: {e}", style="bold red")
        logger.error(f"Input validation failed: {e}")
        raise typer.Exit(1)

    except OptionsAdvisorError as e:
        console.print(f"❌ Options analysis error: {e}", style="bold red")
        logger.error(f"Options advisor error for {ticker}: {e}")
        raise typer.Exit(1)

    except InsufficientDataError as e:
        console.print(f"⚠️  Insufficient data: {e}", style="bold yellow")
        logger.warning(f"Insufficient data for {ticker}: {e}")
        raise typer.Exit(1)

    except CalculationError as e:
        console.print(f"❌ Calculation error: {e}", style="bold red")
        logger.error(f"Calculation error for {ticker}: {e}")
        raise typer.Exit(1)

    except KeyboardInterrupt:
        console.print("\n⏹️  Analysis interrupted by user", style="bold yellow")
        logger.info("Analysis interrupted by user (Ctrl+C)")
        raise typer.Exit(130)  # Standard exit code for SIGINT

    except Exception as e:
        console.print(f"💥 Unexpected error: {e}", style="bold red")
        logger.error(f"Unexpected error in CLI for {ticker}: {e}", exc_info=True)
        raise typer.Exit(1)


@app.command()
def version() -> None:
    """Display BuffetBot CLI version information."""
    console.print("📈 [bold]BuffetBot CLI[/bold] v1.0.0")
    console.print("Financial Analysis Tools - Options Advisor")


if __name__ == "__main__":
    app()
