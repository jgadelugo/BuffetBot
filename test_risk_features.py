#!/usr/bin/env python3
"""
Test Risk Metrics Features

Comprehensive testing of Phase 3 Task 3.3 risk metrics features
including VaR analysis, drawdown metrics, correlation analysis,
and risk-adjusted return calculations.

Author: BuffetBot Development Team
Date: 2024
"""

import logging
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Add project root to path
sys.path.append(
    "/Users/josealvarezdelugo/Documents/projects/Agents/Data Engineer/BuffetBot"
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

try:
    from buffetbot.features.risk.correlation_metrics import CorrelationMetrics
    from buffetbot.features.risk.drawdown_analysis import DrawdownAnalysis
    from buffetbot.features.risk.risk_adjusted_returns import RiskAdjustedReturns
    from buffetbot.features.risk.var_metrics import VaRMetrics

    logger.info("✓ Successfully imported risk metrics features")
except ImportError as e:
    logger.error(f"✗ Import error: {e}")
    sys.exit(1)


def create_test_data(length: int = 252) -> dict:
    """Create realistic financial data for risk testing."""

    # Create date index
    dates = pd.date_range(start="2023-01-01", periods=length, freq="D")

    np.random.seed(42)

    # Create realistic return patterns
    # Stock returns: higher volatility, some negative skew
    stock_drift = 0.0008  # ~20% annual return
    stock_vol = 0.02  # ~32% annual volatility

    # Add volatility clustering (GARCH-like effects)
    vol_persistence = 0.9
    vol_shock = 0.1
    current_vol = stock_vol

    stock_returns = []
    market_returns = []

    for i in range(length):
        # Volatility clustering
        vol_innovation = np.random.normal(0, vol_shock)
        current_vol = (
            vol_persistence * current_vol
            + (1 - vol_persistence) * stock_vol
            + vol_innovation
        )
        current_vol = max(0.005, min(0.05, current_vol))  # Bound volatility

        # Generate correlated returns
        market_return = np.random.normal(stock_drift * 0.8, current_vol * 0.7)

        # Stock return correlated with market (correlation ~0.7)
        stock_return = (
            np.random.normal(stock_drift, current_vol) * 0.7 + market_return * 0.3
        )

        # Add occasional extreme events
        if np.random.random() < 0.02:  # 2% chance of extreme event
            extreme_shock = np.random.normal(-0.05, 0.02)  # Average -5% shock
            stock_return += extreme_shock
            market_return += extreme_shock * 0.8

        stock_returns.append(stock_return)
        market_returns.append(market_return)

    # Convert to pandas Series
    stock_returns = pd.Series(stock_returns, index=dates)
    market_returns = pd.Series(market_returns, index=dates)

    # Create price series
    stock_prices = (1 + stock_returns).cumprod() * 100
    market_prices = (1 + market_returns).cumprod() * 100

    return {
        "dates": dates,
        "stock_returns": stock_returns,
        "market_returns": market_returns,
        "stock_prices": stock_prices,
        "market_prices": market_prices,
    }


def test_var_metrics():
    """Test VaR and Expected Shortfall calculations."""
    logger.info("📉 Testing VaR Metrics...")

    data = create_test_data(252)
    returns = data["stock_returns"]

    try:
        # Test historical VaR
        historical_var = VaRMetrics.historical_var(returns)
        logger.info(
            f"  ✓ Historical VaR calculated: {len(historical_var)} confidence levels"
        )

        if not historical_var["var_95"].empty:
            current_var_95 = historical_var["var_95"].iloc[-1]
            current_var_99 = historical_var["var_99"].iloc[-1]
            logger.info(
                f"    Current 95% VaR: {current_var_95:.3f} ({current_var_95*100:.1f}%)"
            )
            logger.info(
                f"    Current 99% VaR: {current_var_99:.3f} ({current_var_99*100:.1f}%)"
            )

        # Test parametric VaR
        parametric_var = VaRMetrics.parametric_var(returns, distribution="normal")
        logger.info(f"  ✓ Parametric VaR (normal) calculated")

        # Test t-distribution VaR
        t_var = VaRMetrics.parametric_var(returns, distribution="t")
        logger.info(f"  ✓ Parametric VaR (t-distribution) calculated")

        # Test Expected Shortfall
        expected_shortfall = VaRMetrics.expected_shortfall(returns)
        logger.info(f"  ✓ Expected Shortfall calculated")

        if not expected_shortfall["es_95"].empty:
            current_es_95 = expected_shortfall["es_95"].iloc[-1]
            current_es_99 = expected_shortfall["es_99"].iloc[-1]
            logger.info(
                f"    Current 95% ES: {current_es_95:.3f} ({current_es_95*100:.1f}%)"
            )
            logger.info(
                f"    Current 99% ES: {current_es_99:.3f} ({current_es_99*100:.1f}%)"
            )

        # Test VaR breach analysis
        if not historical_var["var_95"].empty:
            breach_analysis = VaRMetrics.var_breach_analysis(
                returns, historical_var["var_95"]
            )
            logger.info(f"  ✓ VaR breach analysis:")
            logger.info(f"    Breach rate: {breach_analysis['breach_rate']:.1%}")
            logger.info(
                f"    Expected rate: {breach_analysis['expected_breach_rate']:.1%}"
            )
            logger.info(f"    Total breaches: {breach_analysis['total_breaches']}")

        # Test tail risk features
        tail_features = VaRMetrics.tail_risk_features(returns)
        logger.info(f"  ✓ Tail risk features: {len(tail_features)} metrics")

        if not tail_features["skewness"].empty:
            current_skew = tail_features["skewness"].iloc[-1]
            current_kurt = tail_features["kurtosis"].iloc[-1]
            logger.info(f"    Current skewness: {current_skew:.2f}")
            logger.info(f"    Current kurtosis: {current_kurt:.2f}")

        return True

    except Exception as e:
        logger.error(f"  ✗ VaR metrics test failed: {e}")
        return False


def test_drawdown_analysis():
    """Test drawdown calculation and analysis."""
    logger.info("📊 Testing Drawdown Analysis...")

    data = create_test_data(252)
    prices = data["stock_prices"]

    try:
        # Test basic drawdown calculation
        dd_metrics = DrawdownAnalysis.calculate_drawdowns(prices)
        logger.info(f"  ✓ Basic drawdown metrics calculated")

        if not dd_metrics["drawdown"].empty:
            current_dd = dd_metrics["drawdown"].iloc[-1]
            max_dd = dd_metrics["drawdown"].min()
            underwater_pct = (
                dd_metrics["underwater"].sum() / len(dd_metrics["underwater"])
            ) * 100
            logger.info(f"    Current drawdown: {current_dd:.1f}%")
            logger.info(f"    Maximum drawdown: {max_dd:.1f}%")
            logger.info(f"    Underwater periods: {underwater_pct:.1f}%")

        # Test maximum drawdown analysis
        max_dd_analysis = DrawdownAnalysis.maximum_drawdown_analysis(prices)
        logger.info(f"  ✓ Maximum drawdown analysis:")
        logger.info(f"    Worst drawdown: {max_dd_analysis['max_drawdown']:.1f}%")
        logger.info(
            f"    Drawdown duration: {max_dd_analysis['drawdown_duration']} days"
        )
        logger.info(
            f"    Recovery duration: {max_dd_analysis['recovery_duration']} days"
        )

        # Test rolling maximum drawdown
        rolling_dd = DrawdownAnalysis.rolling_max_drawdown(prices, window=60)
        logger.info(f"  ✓ Rolling maximum drawdown calculated")

        if not rolling_dd.empty:
            recent_worst = rolling_dd.iloc[-1]
            logger.info(f"    Recent worst (60-day): {recent_worst:.1f}%")

        # Test drawdown clusters
        clusters = DrawdownAnalysis.drawdown_clusters(prices)
        logger.info(f"  ✓ Drawdown cluster analysis:")
        logger.info(f"    Number of clusters: {clusters['cluster_count']}")
        logger.info(
            f"    Average duration: {clusters['avg_cluster_duration']:.1f} days"
        )
        logger.info(f"    Average depth: {clusters['avg_cluster_depth']:.1f}%")

        # Test recovery analysis
        recovery = DrawdownAnalysis.recovery_analysis(prices)
        logger.info(f"  ✓ Recovery analysis:")
        logger.info(
            f"    Average recovery time: {recovery['avg_recovery_time']:.1f} days"
        )

        # Test risk features extraction
        dd_features = DrawdownAnalysis.drawdown_risk_features(prices)
        logger.info(f"  ✓ Drawdown risk features: {len(dd_features)} features")
        logger.info(f"    Risk score: {dd_features['drawdown_risk_score']:.1f}/100")

        return True

    except Exception as e:
        logger.error(f"  ✗ Drawdown analysis test failed: {e}")
        return False


def test_correlation_metrics():
    """Test correlation and beta analysis."""
    logger.info("🔗 Testing Correlation Metrics...")

    data = create_test_data(252)
    stock_returns = data["stock_returns"]
    market_returns = data["market_returns"]

    try:
        # Test rolling correlation
        rolling_corr = CorrelationMetrics.rolling_correlation(
            stock_returns, market_returns
        )
        logger.info(f"  ✓ Rolling correlation calculated")

        if not rolling_corr.empty:
            current_corr = rolling_corr.iloc[-1]
            avg_corr = rolling_corr.mean()
            logger.info(f"    Current correlation: {current_corr:.3f}")
            logger.info(f"    Average correlation: {avg_corr:.3f}")

        # Test rolling beta
        beta_metrics = CorrelationMetrics.rolling_beta(stock_returns, market_returns)
        logger.info(f"  ✓ Rolling beta analysis:")

        if not beta_metrics["beta"].empty:
            current_beta = beta_metrics["beta"].iloc[-1]
            current_alpha = beta_metrics["alpha"].iloc[-1]
            current_r2 = beta_metrics["r_squared"].iloc[-1]
            tracking_error = beta_metrics["tracking_error"].iloc[-1]

            logger.info(f"    Current beta: {current_beta:.3f}")
            logger.info(f"    Current alpha: {current_alpha:.4f}")
            logger.info(f"    Current R²: {current_r2:.3f}")
            logger.info(f"    Tracking error: {tracking_error:.4f}")

        # Test correlation stability
        stability = CorrelationMetrics.correlation_stability(
            stock_returns, market_returns
        )
        logger.info(f"  ✓ Correlation stability analysis:")

        if not stability["stability_score"].empty:
            stability_score = stability["stability_score"].iloc[-1]
            corr_vol = stability["correlation_volatility"].iloc[-1]
            logger.info(f"    Stability score: {stability_score:.1f}/100")
            logger.info(f"    Correlation volatility: {corr_vol:.3f}")

        # Test correlation breakdown analysis
        breakdown = CorrelationMetrics.correlation_breakdown_risk(
            stock_returns, market_returns
        )
        logger.info(f"  ✓ Correlation breakdown analysis:")
        logger.info(f"    Normal correlation: {breakdown['normal_correlation']:.3f}")
        logger.info(f"    Stress correlation: {breakdown['stress_correlation']:.3f}")
        logger.info(f"    Breakdown risk: {breakdown['correlation_breakdown']:.3f}")

        # Test multi-asset correlation (create additional synthetic assets)
        returns_dict = {
            "Asset1": stock_returns,
            "Market": market_returns,
            "Asset2": stock_returns * 0.8
            + np.random.normal(0, 0.01, len(stock_returns)),
        }

        multi_corr = CorrelationMetrics.multi_asset_correlation_matrix(returns_dict)
        logger.info(f"  ✓ Multi-asset correlation analysis:")
        logger.info(
            f"    Correlation matrices: {len(multi_corr['correlation_matrices'])}"
        )

        if not multi_corr["avg_correlation"].empty:
            avg_correlation = multi_corr["avg_correlation"].iloc[-1]
            logger.info(f"    Average correlation: {avg_correlation:.3f}")

        # Test correlation risk features
        corr_features = CorrelationMetrics.correlation_risk_features(
            stock_returns, market_returns
        )
        logger.info(f"  ✓ Correlation risk features: {len(corr_features)} features")
        logger.info(f"    Risk level: {corr_features['correlation_risk_level']}")
        logger.info(
            f"    Diversification benefit: {corr_features['diversification_benefit']:.1f}%"
        )

        return True

    except Exception as e:
        logger.error(f"  ✗ Correlation metrics test failed: {e}")
        return False


def test_risk_adjusted_returns():
    """Test risk-adjusted return calculations."""
    logger.info("📈 Testing Risk-Adjusted Returns...")

    data = create_test_data(252)
    returns = data["stock_returns"]
    market_returns = data["market_returns"]

    try:
        # Test Sharpe ratio
        sharpe = RiskAdjustedReturns.sharpe_ratio(returns, risk_free_rate=0.02)
        logger.info(f"  ✓ Sharpe ratio calculated")

        if not sharpe.empty:
            current_sharpe = sharpe.iloc[-1]
            avg_sharpe = sharpe.mean()
            logger.info(f"    Current Sharpe: {current_sharpe:.3f}")
            logger.info(f"    Average Sharpe: {avg_sharpe:.3f}")

        # Test Sortino ratio
        sortino = RiskAdjustedReturns.sortino_ratio(returns, risk_free_rate=0.02)
        logger.info(f"  ✓ Sortino ratio calculated")

        if not sortino.empty:
            current_sortino = sortino.iloc[-1]
            logger.info(f"    Current Sortino: {current_sortino:.3f}")

        # Test Calmar ratio
        calmar = RiskAdjustedReturns.calmar_ratio(returns)
        logger.info(f"  ✓ Calmar ratio calculated")

        if not calmar.empty:
            current_calmar = calmar.iloc[-1]
            logger.info(f"    Current Calmar: {current_calmar:.3f}")

        # Test Information ratio
        info_ratio = RiskAdjustedReturns.information_ratio(returns, market_returns)
        logger.info(f"  ✓ Information ratio calculated")

        if not info_ratio.empty:
            current_info = info_ratio.iloc[-1]
            logger.info(f"    Current Information ratio: {current_info:.3f}")

        # Test Omega ratio
        omega = RiskAdjustedReturns.omega_ratio(returns, threshold=0.0)
        logger.info(f"  ✓ Omega ratio calculated")

        if not omega.empty:
            current_omega = omega.iloc[-1]
            logger.info(f"    Current Omega: {current_omega:.3f}")

        # Test Sterling ratio
        sterling = RiskAdjustedReturns.sterling_ratio(returns)
        logger.info(f"  ✓ Sterling ratio calculated")

        if not sterling.empty:
            current_sterling = sterling.iloc[-1]
            logger.info(f"    Current Sterling: {current_sterling:.3f}")

        # Test comprehensive metrics
        comprehensive = RiskAdjustedReturns.comprehensive_risk_adjusted_metrics(
            returns, market_returns, risk_free_rate=0.02
        )
        logger.info(f"  ✓ Comprehensive risk-adjusted metrics:")
        logger.info(f"    Performance class: {comprehensive['performance_class']}")
        logger.info(
            f"    Risk efficiency score: {comprehensive['risk_efficiency_score']:.1f}/100"
        )

        return True

    except Exception as e:
        logger.error(f"  ✗ Risk-adjusted returns test failed: {e}")
        return False


def test_performance():
    """Test performance with larger datasets."""
    logger.info("⚡ Testing Performance...")

    # Test with larger dataset
    large_data = create_test_data(1000)  # ~4 years of data

    try:
        import time

        # Test VaR performance
        start_time = time.time()
        var_result = VaRMetrics.historical_var(large_data["stock_returns"])
        var_time = time.time() - start_time
        logger.info(f"  ✓ VaR analysis (1000 days): {var_time:.3f}s")

        # Test drawdown performance
        start_time = time.time()
        dd_result = DrawdownAnalysis.calculate_drawdowns(large_data["stock_prices"])
        dd_time = time.time() - start_time
        logger.info(f"  ✓ Drawdown analysis (1000 days): {dd_time:.3f}s")

        # Test correlation performance
        start_time = time.time()
        corr_result = CorrelationMetrics.rolling_correlation(
            large_data["stock_returns"], large_data["market_returns"]
        )
        corr_time = time.time() - start_time
        logger.info(f"  ✓ Correlation analysis (1000 days): {corr_time:.3f}s")

        # Test risk-adjusted returns performance
        start_time = time.time()
        risk_adj_result = RiskAdjustedReturns.sharpe_ratio(large_data["stock_returns"])
        risk_adj_time = time.time() - start_time
        logger.info(f"  ✓ Risk-adjusted returns (1000 days): {risk_adj_time:.3f}s")

        total_time = var_time + dd_time + corr_time + risk_adj_time
        logger.info(f"  ✓ Total processing time: {total_time:.3f}s")

        # Performance target: sub-2-second for 1000 days
        if total_time < 2.0:
            logger.info(f"  ✓ Performance target met: {total_time:.3f}s < 2.0s")
            return True
        else:
            logger.warning(f"  ⚠ Performance target missed: {total_time:.3f}s > 2.0s")
            return False

    except Exception as e:
        logger.error(f"  ✗ Performance test failed: {e}")
        return False


def main():
    """Run comprehensive risk metrics features test."""
    logger.info("🚀 Testing Phase 3 Task 3.3: Risk Metrics Features")
    logger.info("=" * 60)

    # Track test results
    test_results = []

    # Run all tests
    test_results.append(("VaR Metrics", test_var_metrics()))
    test_results.append(("Drawdown Analysis", test_drawdown_analysis()))
    test_results.append(("Correlation Metrics", test_correlation_metrics()))
    test_results.append(("Risk-Adjusted Returns", test_risk_adjusted_returns()))
    test_results.append(("Performance", test_performance()))

    # Summary
    logger.info("=" * 60)
    logger.info("📋 Test Results Summary:")

    passed = 0
    for test_name, result in test_results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"  {status} {test_name}")
        if result:
            passed += 1

    logger.info(f"\n🎯 Overall: {passed}/{len(test_results)} tests passed")

    if passed == len(test_results):
        logger.info("🎉 Phase 3 Task 3.3 - Risk Metrics Features: COMPLETE")
        logger.info("💰 Cost: $0.00 (local processing)")
        logger.info("🚀 Ready to enhance ML models with comprehensive risk features!")
        return True
    else:
        logger.error("❌ Some tests failed. Please review and fix issues.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
