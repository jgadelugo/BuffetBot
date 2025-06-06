#!/usr/bin/env python3
"""
Test Market Structure Features

Comprehensive testing of Phase 3 Task 3.2 market structure features
including gap analysis, support/resistance levels, and market regime detection.

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
    from buffetbot.features.market.gaps import GapAnalysis, analyze_gap_patterns
    from buffetbot.features.market.regime import (
        MarketRegimeDetector,
        detect_market_regime,
    )
    from buffetbot.features.market.support_resistance import (
        SupportResistance,
        identify_key_levels,
    )

    logger.info("✓ Successfully imported market structure features")
except ImportError as e:
    logger.error(f"✗ Import error: {e}")
    sys.exit(1)


def create_test_data(length: int = 100) -> dict:
    """Create realistic market data for testing."""

    # Create date index
    dates = pd.date_range(start="2023-01-01", periods=length, freq="D")

    # Create realistic price movements with gaps
    np.random.seed(42)

    # Base price trend with some volatility
    base_price = 100.0
    price_trend = np.cumsum(
        np.random.normal(0.001, 0.02, length)
    )  # 0.1% daily drift, 2% volatility

    # Add some gaps (sudden jumps)
    gap_days = [20, 45, 70]  # Days with gaps
    gap_sizes = [0.05, -0.03, 0.04]  # Gap sizes (5%, -3%, 4%)

    prices = [base_price]

    for i in range(1, length):
        # Normal price movement
        price_change = price_trend[i] - price_trend[i - 1]
        new_price = prices[-1] * (1 + price_change)

        # Add gaps on specific days
        if i in gap_days:
            gap_idx = gap_days.index(i)
            gap_size = gap_sizes[gap_idx]
            new_price = new_price * (1 + gap_size)
            logger.info(f"Added gap on day {i}: {gap_size:.1%}")

        prices.append(new_price)

    # Create OHLC data
    prices = np.array(prices)

    # High = close + random positive amount
    high = prices * (1 + np.abs(np.random.normal(0, 0.01, length)))

    # Low = close - random positive amount
    low = prices * (1 - np.abs(np.random.normal(0, 0.01, length)))

    # Open = previous close + small random amount (except for gap days)
    open_prices = np.zeros(length)
    open_prices[0] = prices[0]

    for i in range(1, length):
        if i in gap_days:
            # For gap days, open significantly different from previous close
            gap_idx = gap_days.index(i)
            gap_size = gap_sizes[gap_idx]
            open_prices[i] = prices[i - 1] * (1 + gap_size)
        else:
            # Normal opening near previous close
            open_prices[i] = prices[i - 1] * (1 + np.random.normal(0, 0.003))

    # Volume with realistic patterns
    base_volume = 1000000
    volume = np.random.lognormal(np.log(base_volume), 0.5, length).astype(int)

    # Higher volume on gap days
    for gap_day in gap_days:
        volume[gap_day] = int(volume[gap_day] * 2.5)

    return {
        "dates": dates,
        "open": pd.Series(open_prices, index=dates),
        "high": pd.Series(high, index=dates),
        "low": pd.Series(low, index=dates),
        "close": pd.Series(prices, index=dates),
        "volume": pd.Series(volume, index=dates),
    }


def test_gap_analysis():
    """Test gap detection and analysis features."""
    logger.info("🔍 Testing Gap Analysis...")

    # Create test data with known gaps
    data = create_test_data(100)

    try:
        # Test gap detection
        gaps_df = GapAnalysis.detect_gaps(
            data["open"], data["high"], data["low"], data["close"]
        )

        logger.info(f"  ✓ Detected {len(gaps_df)} gaps")

        if not gaps_df.empty:
            # Display gap information
            for _, gap in gaps_df.iterrows():
                logger.info(
                    f"    Gap on {gap['date'].date()}: "
                    f"{gap['gap_direction']} {gap['gap_percent']:.2f}% "
                    f"({gap['gap_type']}) - Filled: {gap['is_filled']}"
                )

        # Test gap statistics
        stats = GapAnalysis.gap_statistics(gaps_df)
        logger.info(f"  ✓ Gap statistics calculated: {len(stats)} metrics")
        logger.info(f"    Fill rate: {stats.get('gap_fill_rate', 0):.1f}%")
        logger.info(f"    Average gap size: {stats.get('avg_gap_size', 0):.2f}%")

        # Test recent gap features
        recent_features = GapAnalysis.recent_gap_features(gaps_df)
        logger.info(f"  ✓ Recent gap features: {len(recent_features)} features")
        logger.info(
            f"    Recent gaps count: {recent_features.get('recent_gaps_count', 0)}"
        )

        # Test comprehensive analysis
        analysis = analyze_gap_patterns(
            data["open"], data["high"], data["low"], data["close"]
        )
        logger.info(f"  ✓ Comprehensive gap analysis completed")

        return True

    except Exception as e:
        logger.error(f"  ✗ Gap analysis failed: {e}")
        return False


def test_support_resistance():
    """Test support and resistance level identification."""
    logger.info("🎯 Testing Support & Resistance Analysis...")

    data = create_test_data(100)

    try:
        # Test pivot point levels
        pivot_levels = SupportResistance.pivot_point_levels(
            data["high"], data["low"], data["close"]
        )
        logger.info(f"  ✓ Pivot points calculated: {len(pivot_levels)} levels")

        current_price = data["close"].iloc[-1]
        if not pivot_levels["resistance1"].empty:
            r1 = pivot_levels["resistance1"].iloc[-1]
            s1 = pivot_levels["support1"].iloc[-1]
            logger.info(f"    Current price: ${current_price:.2f}")
            logger.info(
                f"    Resistance 1: ${r1:.2f} (+{((r1/current_price)-1)*100:.1f}%)"
            )
            logger.info(f"    Support 1: ${s1:.2f} ({((s1/current_price)-1)*100:.1f}%)")

        # Test psychological levels
        psych_levels = SupportResistance.psychological_levels(data["close"])
        logger.info(
            f"  ✓ Psychological levels: {len(psych_levels['nearby_levels'])} nearby levels"
        )

        if psych_levels["nearby_levels"]:
            logger.info(f"    Nearby levels: {psych_levels['nearby_levels'][:5]}")

        # Test peak/trough analysis
        peak_trough = SupportResistance.peak_trough_levels(
            data["high"], data["low"], data["close"]
        )
        logger.info(f"  ✓ Peak/trough analysis:")
        logger.info(
            f"    Resistance levels found: {len(peak_trough['resistance_levels'])}"
        )
        logger.info(f"    Support levels found: {len(peak_trough['support_levels'])}")

        # Test volume profile levels
        volume_levels = SupportResistance.volume_profile_levels(
            data["close"], data["volume"]
        )
        logger.info(f"  ✓ Volume profile analysis:")
        logger.info(f"    High-volume levels: {len(volume_levels['volume_levels'])}")

        # Test level strength scoring
        if not peak_trough["resistance_levels"].empty:
            test_level = peak_trough["resistance_levels"].iloc[0]["price"]
            strength = SupportResistance.level_strength_score(
                test_level, data["high"], data["low"], data["close"]
            )
            logger.info(
                f"  ✓ Level strength test: {strength['strength_score']:.1f}/100"
            )

        # Test comprehensive level identification
        all_levels = identify_key_levels(
            data["high"], data["low"], data["close"], data["volume"]
        )
        logger.info(
            f"  ✓ Comprehensive analysis: {len(all_levels)} level types identified"
        )

        return True

    except Exception as e:
        logger.error(f"  ✗ Support/Resistance analysis failed: {e}")
        return False


def test_market_regime():
    """Test market regime detection."""
    logger.info("📊 Testing Market Regime Detection...")

    data = create_test_data(100)

    try:
        # Test ADX regime
        adx_regime = MarketRegimeDetector.adx_regime(
            data["high"], data["low"], data["close"]
        )
        logger.info(f"  ✓ ADX regime analysis completed")

        if not adx_regime["regime"].empty:
            current_adx_regime = adx_regime["regime"].iloc[-1]
            current_trend = adx_regime["trend_direction"].iloc[-1]
            current_adx = adx_regime["adx"].iloc[-1]
            logger.info(f"    Current ADX regime: {current_adx_regime}")
            logger.info(f"    Trend direction: {current_trend}")
            logger.info(f"    ADX value: {current_adx:.1f}")

        # Test volatility regime
        vol_regime = MarketRegimeDetector.volatility_regime(data["close"])
        logger.info(f"  ✓ Volatility regime analysis completed")

        if not vol_regime["regime"].empty:
            current_vol_regime = vol_regime["regime"].iloc[-1]
            current_vol = vol_regime["volatility"].iloc[-1]
            vol_percentile = vol_regime["volatility_percentile"].iloc[-1]
            logger.info(f"    Current volatility regime: {current_vol_regime}")
            logger.info(f"    Volatility: {current_vol:.1f}%")
            logger.info(f"    Volatility percentile: {vol_percentile:.1f}%")

        # Test price action regime
        price_regime = MarketRegimeDetector.price_action_regime(
            data["high"], data["low"], data["close"]
        )
        logger.info(f"  ✓ Price action regime analysis completed")

        if not price_regime["regime"].empty:
            current_price_regime = price_regime["regime"].iloc[-1]
            trend_consistency = price_regime["trend_consistency"].iloc[-1]
            range_position = price_regime["range_position"].iloc[-1]
            logger.info(f"    Current price regime: {current_price_regime}")
            logger.info(f"    Trend consistency: {trend_consistency:.1f}%")
            logger.info(f"    Range position: {range_position:.1f}%")

        # Test composite regime
        composite = MarketRegimeDetector.composite_regime(
            data["high"], data["low"], data["close"]
        )
        logger.info(f"  ✓ Composite regime analysis completed")

        if not composite["composite_regime"].empty:
            current_composite = composite["composite_regime"].iloc[-1]
            confidence = composite["regime_confidence"].iloc[-1]
            logger.info(f"    Composite regime: {current_composite}")
            logger.info(f"    Confidence: {confidence:.1f}%")

        # Test regime transitions
        if not composite["composite_regime"].empty:
            transitions = MarketRegimeDetector.regime_transitions(
                composite["composite_regime"]
            )
            logger.info(f"  ✓ Regime transition analysis:")
            logger.info(
                f"    Current regime duration: {transitions['regime_duration']} days"
            )
            logger.info(f"    Stability score: {transitions['stability_score']:.1f}%")

        # Test comprehensive regime detection
        regime_info = detect_market_regime(data["high"], data["low"], data["close"])
        logger.info(f"  ✓ Comprehensive regime detection:")
        logger.info(f"    Current regime: {regime_info['current_regime']}")
        logger.info(f"    Confidence: {regime_info['regime_confidence']:.1f}%")
        logger.info(f"    ML features: {len(regime_info['regime_features'])} features")

        return True

    except Exception as e:
        logger.error(f"  ✗ Market regime detection failed: {e}")
        return False


def test_performance():
    """Test performance with larger datasets."""
    logger.info("⚡ Testing Performance...")

    # Test with larger dataset
    large_data = create_test_data(500)  # 500 days

    try:
        import time

        # Test gap analysis performance
        start_time = time.time()
        gaps_analysis = analyze_gap_patterns(
            large_data["open"],
            large_data["high"],
            large_data["low"],
            large_data["close"],
        )
        gap_time = time.time() - start_time
        logger.info(f"  ✓ Gap analysis (500 days): {gap_time:.3f}s")

        # Test support/resistance performance
        start_time = time.time()
        levels_analysis = identify_key_levels(
            large_data["high"],
            large_data["low"],
            large_data["close"],
            large_data["volume"],
        )
        levels_time = time.time() - start_time
        logger.info(f"  ✓ Support/Resistance (500 days): {levels_time:.3f}s")

        # Test regime detection performance
        start_time = time.time()
        regime_analysis = detect_market_regime(
            large_data["high"], large_data["low"], large_data["close"]
        )
        regime_time = time.time() - start_time
        logger.info(f"  ✓ Regime detection (500 days): {regime_time:.3f}s")

        total_time = gap_time + levels_time + regime_time
        logger.info(f"  ✓ Total processing time: {total_time:.3f}s")

        # Performance target: sub-second for 500 days
        if total_time < 1.0:
            logger.info(f"  ✓ Performance target met: {total_time:.3f}s < 1.0s")
            return True
        else:
            logger.warning(f"  ⚠ Performance target missed: {total_time:.3f}s > 1.0s")
            return False

    except Exception as e:
        logger.error(f"  ✗ Performance test failed: {e}")
        return False


def main():
    """Run comprehensive market structure features test."""
    logger.info("🚀 Testing Phase 3 Task 3.2: Market Structure Features")
    logger.info("=" * 60)

    # Track test results
    test_results = []

    # Run all tests
    test_results.append(("Gap Analysis", test_gap_analysis()))
    test_results.append(("Support & Resistance", test_support_resistance()))
    test_results.append(("Market Regime Detection", test_market_regime()))
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
        logger.info("🎉 Phase 3 Task 3.2 - Market Structure Features: COMPLETE")
        logger.info("💰 Cost: $0.00 (local processing)")
        logger.info("🚀 Ready to enhance ML models with market structure features!")
        return True
    else:
        logger.error("❌ Some tests failed. Please review and fix issues.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
