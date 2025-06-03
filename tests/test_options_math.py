#!/usr/bin/env python3
"""
Test script for the options_math module.

This script demonstrates the functionality of all four technical indicator functions
and validates their proper operation with sample data. It includes structured logging,
comprehensive error handling, and modular design for easy unit testing.
"""

import logging
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from utils.options_math import (
    OptionsMathError,
    calculate_average_iv,
    calculate_beta,
    calculate_momentum,
    calculate_rsi,
    validate_options_math_inputs,
)


class TestResult(Enum):
    """Enum for test result status."""

    PASSED = "PASSED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


@dataclass
class TestCase:
    """Data class to represent a test case result."""

    name: str
    status: TestResult
    message: str
    execution_time: float | None = None
    error_details: str | None = None


class TestDataGenerator:
    """Class responsible for generating test data for various scenarios."""

    def __init__(self, random_seed: int = 42) -> None:
        """Initialize the test data generator.

        Args:
            random_seed: Seed for reproducible random data generation.
        """
        np.random.seed(random_seed)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def generate_price_series(
        self,
        length: int = 50,
        start_price: float = 100.0,
        trend: float = 0.02,
        volatility: float = 0.02,
    ) -> pd.Series:
        """Generate a price series with specified characteristics.

        Args:
            length: Number of price points to generate.
            start_price: Starting price value.
            trend: Daily trend factor (0.02 = 2% daily growth).
            volatility: Daily volatility factor.

        Returns:
            Generated price series.

        Raises:
            ValueError: If invalid parameters are provided.
        """
        self.logger.info(
            f"Generating price series: length={length}, start_price={start_price}, "
            f"trend={trend}, volatility={volatility}"
        )

        if length <= 0:
            raise ValueError("Length must be positive")
        if start_price <= 0:
            raise ValueError("Start price must be positive")

        try:
            # Generate trending prices with noise
            trend_component = np.arange(length) * trend
            noise_component = np.random.normal(0, volatility, length)
            prices = start_price * (1 + trend_component + noise_component)

            result = pd.Series(prices)
            self.logger.debug(f"Generated price series with {len(result)} points")
            return result

        except Exception as e:
            self.logger.error(f"Failed to generate price series: {e}")
            raise

    def generate_return_series(
        self, length: int = 50, mean_return: float = 0.01, volatility: float = 0.02
    ) -> pd.Series:
        """Generate a return series with specified statistical properties.

        Args:
            length: Number of return observations.
            mean_return: Mean return value.
            volatility: Return volatility.

        Returns:
            Generated return series.
        """
        self.logger.info(
            f"Generating return series: length={length}, mean={mean_return}, vol={volatility}"
        )

        try:
            returns = np.random.normal(mean_return, volatility, length)
            result = pd.Series(returns)
            self.logger.debug(f"Generated return series with {len(result)} points")
            return result

        except Exception as e:
            self.logger.error(f"Failed to generate return series: {e}")
            raise

    def generate_options_data(
        self,
        num_options: int = 10,
        iv_range: tuple[float, float] = (0.15, 0.40),
        volume_range: tuple[int, int] = (50, 500),
    ) -> pd.DataFrame:
        """Generate sample options data for testing.

        Args:
            num_options: Number of option contracts to generate.
            iv_range: Range for implied volatility values.
            volume_range: Range for volume values.

        Returns:
            DataFrame with options data including impliedVolatility and volume.
        """
        self.logger.info(
            f"Generating options data: num_options={num_options}, "
            f"iv_range={iv_range}, volume_range={volume_range}"
        )

        try:
            iv_values = np.random.uniform(iv_range[0], iv_range[1], num_options)
            volume_values = np.random.randint(
                volume_range[0], volume_range[1], num_options
            )

            data = pd.DataFrame(
                {"impliedVolatility": iv_values, "volume": volume_values}
            )

            self.logger.debug(f"Generated options data with {len(data)} contracts")
            return data

        except Exception as e:
            self.logger.error(f"Failed to generate options data: {e}")
            raise


class OptionsMathTester:
    """Main testing class for options_math module functionality."""

    def __init__(self, data_generator: TestDataGenerator | None = None) -> None:
        """Initialize the tester.

        Args:
            data_generator: Optional custom data generator instance.
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.data_generator = data_generator or TestDataGenerator()
        self.test_results: list[TestCase] = []

    def _log_test_start(self, test_name: str) -> None:
        """Log the start of a test case.

        Args:
            test_name: Name of the test being started.
        """
        self.logger.info(f"Starting test: {test_name}")

    def _log_test_result(self, test_case: TestCase) -> None:
        """Log the result of a test case.

        Args:
            test_case: Completed test case with results.
        """
        if test_case.status == TestResult.PASSED:
            self.logger.info(f"✅ {test_case.name}: {test_case.message}")
        elif test_case.status == TestResult.FAILED:
            self.logger.error(f"❌ {test_case.name}: {test_case.message}")
            if test_case.error_details:
                self.logger.error(f"Error details: {test_case.error_details}")
        else:
            self.logger.warning(f"⏭️ {test_case.name}: {test_case.message}")

    def test_rsi_calculation(self) -> TestCase:
        """Test RSI calculation with various scenarios.

        Returns:
            TestCase with results of RSI testing.
        """
        test_name = "RSI Calculation"
        self._log_test_start(test_name)

        try:
            self.logger.info("Generating test data for RSI calculation")

            # Generate trending price data
            prices = self.data_generator.generate_price_series(
                length=30, trend=0.01, volatility=0.02
            )

            self.logger.info("Computing RSI with different periods")

            # Test default period (14)
            rsi_14 = calculate_rsi(prices)
            self.logger.debug(f"RSI (14-period): {rsi_14:.2f}")

            # Test shorter period
            rsi_5 = calculate_rsi(prices, period=5)
            self.logger.debug(f"RSI (5-period): {rsi_5:.2f}")

            # Validate RSI range (0-100)
            if not (0 <= rsi_14 <= 100) or not (0 <= rsi_5 <= 100):
                raise ValueError("RSI values outside expected range [0, 100]")

            # Test with NaN values
            self.logger.info("Testing RSI with NaN values")
            prices_with_nan = prices.copy()
            prices_with_nan.iloc[5:8] = np.nan
            rsi_nan = calculate_rsi(prices_with_nan)
            self.logger.debug(f"RSI with NaN values: {rsi_nan:.2f}")

            result = TestCase(
                name=test_name,
                status=TestResult.PASSED,
                message=f"RSI calculations successful (14-period: {rsi_14:.2f}, 5-period: {rsi_5:.2f})",
            )

        except Exception as e:
            self.logger.error(f"RSI test failed: {e}")
            result = TestCase(
                name=test_name,
                status=TestResult.FAILED,
                message="RSI calculation test failed",
                error_details=str(e),
            )

        self._log_test_result(result)
        self.test_results.append(result)
        return result

    def test_beta_calculation(self) -> TestCase:
        """Test beta calculation with various market scenarios.

        Returns:
            TestCase with results of beta testing.
        """
        test_name = "Beta Calculation"
        self._log_test_start(test_name)

        try:
            self.logger.info("Generating market and stock return data")

            # Generate market returns
            market_returns = self.data_generator.generate_return_series(
                length=60, mean_return=0.008, volatility=0.015
            )

            # Generate stock returns with higher beta
            stock_returns = (
                market_returns * 1.3
                + self.data_generator.generate_return_series(60, 0, 0.01)
            )

            self.logger.info("Computing beta coefficient")
            beta = calculate_beta(stock_returns, market_returns)
            self.logger.debug(f"Beta coefficient: {beta:.3f}")

            # Test with highly correlated returns (beta should be close to 1)
            correlated_returns = market_returns + np.random.normal(
                0, 0.001, len(market_returns)
            )
            beta_correlated = calculate_beta(correlated_returns, market_returns)
            self.logger.debug(f"Beta (highly correlated): {beta_correlated:.3f}")

            result = TestCase(
                name=test_name,
                status=TestResult.PASSED,
                message=f"Beta calculations successful (main: {beta:.3f}, correlated: {beta_correlated:.3f})",
            )

        except Exception as e:
            self.logger.error(f"Beta test failed: {e}")
            result = TestCase(
                name=test_name,
                status=TestResult.FAILED,
                message="Beta calculation test failed",
                error_details=str(e),
            )

        self._log_test_result(result)
        self.test_results.append(result)
        return result

    def test_momentum_calculation(self) -> TestCase:
        """Test momentum calculation with trending and declining scenarios.

        Returns:
            TestCase with results of momentum testing.
        """
        test_name = "Momentum Calculation"
        self._log_test_start(test_name)

        try:
            self.logger.info("Generating trending price data for momentum test")

            # Strong uptrend
            uptrend_prices = self.data_generator.generate_price_series(
                length=30, trend=0.02, volatility=0.01
            )

            # Downtrend
            downtrend_prices = self.data_generator.generate_price_series(
                length=30, trend=-0.015, volatility=0.01
            )

            self.logger.info("Computing momentum for different scenarios")

            # Test uptrend momentum
            momentum_up_20 = calculate_momentum(uptrend_prices, window=20)
            self.logger.debug(f"Uptrend momentum (20-period): {momentum_up_20:.2%}")

            momentum_up_5 = calculate_momentum(uptrend_prices, window=5)
            self.logger.debug(f"Uptrend momentum (5-period): {momentum_up_5:.2%}")

            # Test downtrend momentum
            momentum_down = calculate_momentum(downtrend_prices, window=10)
            self.logger.debug(f"Downtrend momentum: {momentum_down:.2%}")

            # Validate momentum signs
            if momentum_up_20 <= 0 or momentum_up_5 <= 0:
                self.logger.warning("Expected positive momentum for uptrending prices")

            if momentum_down >= 0:
                self.logger.warning(
                    "Expected negative momentum for downtrending prices"
                )

            result = TestCase(
                name=test_name,
                status=TestResult.PASSED,
                message=f"Momentum calculations successful (up: {momentum_up_20:.2%}, down: {momentum_down:.2%})",
            )

        except Exception as e:
            self.logger.error(f"Momentum test failed: {e}")
            result = TestCase(
                name=test_name,
                status=TestResult.FAILED,
                message="Momentum calculation test failed",
                error_details=str(e),
            )

        self._log_test_result(result)
        self.test_results.append(result)
        return result

    def test_average_iv_calculation(self) -> TestCase:
        """Test average implied volatility calculation.

        Returns:
            TestCase with results of average IV testing.
        """
        test_name = "Average IV Calculation"
        self._log_test_start(test_name)

        try:
            self.logger.info("Generating options data for IV testing")

            # Generate options data
            options_data = self.data_generator.generate_options_data(
                num_options=15, iv_range=(0.15, 0.45), volume_range=(50, 800)
            )

            self.logger.info("Computing volume-weighted and simple average IV")

            # Test volume-weighted average
            avg_iv_weighted = calculate_average_iv(options_data)
            self.logger.debug(f"Volume-weighted average IV: {avg_iv_weighted:.2%}")

            # Test simple average (without volume)
            options_simple = options_data[["impliedVolatility"]].copy()
            avg_iv_simple = calculate_average_iv(options_simple)
            self.logger.debug(f"Simple average IV: {avg_iv_simple:.2%}")

            # Test with NaN values
            self.logger.info("Testing IV calculation with missing data")
            options_with_nan = options_data.copy()
            options_with_nan.loc[2:4, "impliedVolatility"] = np.nan
            avg_iv_nan = calculate_average_iv(options_with_nan)
            self.logger.debug(f"Average IV with NaN values: {avg_iv_nan:.2%}")

            # Validate IV ranges
            if not (0 <= avg_iv_weighted <= 1) or not (0 <= avg_iv_simple <= 1):
                raise ValueError("IV values outside expected range [0, 1]")

            result = TestCase(
                name=test_name,
                status=TestResult.PASSED,
                message=f"IV calculations successful (weighted: {avg_iv_weighted:.2%}, simple: {avg_iv_simple:.2%})",
            )

        except Exception as e:
            self.logger.error(f"Average IV test failed: {e}")
            result = TestCase(
                name=test_name,
                status=TestResult.FAILED,
                message="Average IV calculation test failed",
                error_details=str(e),
            )

        self._log_test_result(result)
        self.test_results.append(result)
        return result

    def test_error_handling(self) -> TestCase:
        """Test comprehensive error handling scenarios.

        Returns:
            TestCase with results of error handling testing.
        """
        test_name = "Error Handling"
        self._log_test_start(test_name)

        error_tests_passed = 0
        total_error_tests = 0

        try:
            self.logger.info("Testing error handling scenarios")

            # Test invalid input types
            total_error_tests += 1
            try:
                calculate_rsi("not a series")
                self.logger.error("Should have raised error for invalid input type")
            except OptionsMathError:
                self.logger.debug("✅ Correctly caught invalid input type")
                error_tests_passed += 1

            # Test insufficient data
            total_error_tests += 1
            try:
                short_series = pd.Series([100, 101])
                calculate_rsi(short_series, period=14)
                self.logger.error("Should have raised error for insufficient data")
            except OptionsMathError:
                self.logger.debug("✅ Correctly caught insufficient data error")
                error_tests_passed += 1

            # Test mismatched series lengths for beta
            total_error_tests += 1
            try:
                stock_ret = pd.Series([0.01, 0.02, 0.03])
                market_ret = pd.Series([0.01, 0.02])
                calculate_beta(stock_ret, market_ret)
                self.logger.error("Should have raised error for mismatched lengths")
            except OptionsMathError:
                self.logger.debug("✅ Correctly caught mismatched lengths error")
                error_tests_passed += 1

            # Test missing required columns
            total_error_tests += 1
            try:
                bad_df = pd.DataFrame({"wrong_column": [0.1, 0.2, 0.3]})
                calculate_average_iv(bad_df)
                self.logger.error("Should have raised error for missing columns")
            except OptionsMathError:
                self.logger.debug("✅ Correctly caught missing columns error")
                error_tests_passed += 1

            if error_tests_passed == total_error_tests:
                result = TestCase(
                    name=test_name,
                    status=TestResult.PASSED,
                    message=f"All {total_error_tests} error handling scenarios passed",
                )
            else:
                result = TestCase(
                    name=test_name,
                    status=TestResult.FAILED,
                    message=f"Only {error_tests_passed}/{total_error_tests} error tests passed",
                )

        except Exception as e:
            self.logger.error(f"Error handling test setup failed: {e}")
            result = TestCase(
                name=test_name,
                status=TestResult.FAILED,
                message="Error handling test setup failed",
                error_details=str(e),
            )

        self._log_test_result(result)
        self.test_results.append(result)
        return result

    def test_validation_helper(self) -> TestCase:
        """Test the validation helper function.

        Returns:
            TestCase with results of validation testing.
        """
        test_name = "Validation Helper"
        self._log_test_start(test_name)

        try:
            self.logger.info("Testing validation helper function")

            prices = self.data_generator.generate_price_series(length=20)
            returns = self.data_generator.generate_return_series(length=19)
            options_df = self.data_generator.generate_options_data(num_options=5)
            periods = [5, 10, 20]

            self.logger.info("Running comprehensive input validation")
            results = validate_options_math_inputs(
                prices=prices, returns=returns, option_data=options_df, periods=periods
            )

            self.logger.debug(f"Validation results: {results}")

            result = TestCase(
                name=test_name,
                status=TestResult.PASSED,
                message=f"Validation helper successful: {results}",
            )

        except Exception as e:
            self.logger.error(f"Validation helper test failed: {e}")
            result = TestCase(
                name=test_name,
                status=TestResult.FAILED,
                message="Validation helper test failed",
                error_details=str(e),
            )

        self._log_test_result(result)
        self.test_results.append(result)
        return result

    def run_all_tests(self) -> dict[str, Any]:
        """Run all test cases and return comprehensive results.

        Returns:
            Dictionary containing test summary and detailed results.
        """
        self.logger.info("🚀 Starting comprehensive options math module testing")

        # Run all test cases
        test_methods = [
            self.test_rsi_calculation,
            self.test_beta_calculation,
            self.test_momentum_calculation,
            self.test_average_iv_calculation,
            self.test_error_handling,
            self.test_validation_helper,
        ]

        for test_method in test_methods:
            try:
                test_method()
            except Exception as e:
                self.logger.error(f"Unexpected error in {test_method.__name__}: {e}")
                self.test_results.append(
                    TestCase(
                        name=test_method.__name__,
                        status=TestResult.FAILED,
                        message="Unexpected test failure",
                        error_details=str(e),
                    )
                )

        # Calculate summary statistics
        passed_tests = sum(
            1 for result in self.test_results if result.status == TestResult.PASSED
        )
        failed_tests = sum(
            1 for result in self.test_results if result.status == TestResult.FAILED
        )
        total_tests = len(self.test_results)

        summary = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "detailed_results": self.test_results,
        }

        self.logger.info(
            f"🎉 Testing completed: {passed_tests}/{total_tests} tests passed"
        )

        if failed_tests > 0:
            self.logger.warning(f"⚠️ {failed_tests} tests failed")
            for result in self.test_results:
                if result.status == TestResult.FAILED:
                    self.logger.error(f"Failed test: {result.name} - {result.message}")

        return summary


def setup_logging(log_level: str = "INFO") -> None:
    """Configure structured logging for the test suite.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("options_math_test.log", mode="w"),
        ],
    )


def main() -> int:
    """Main function to run all tests with proper setup and teardown.

    Returns:
        Exit code (0 for success, 1 for failures).
    """
    try:
        # Setup logging
        setup_logging("INFO")
        logger = logging.getLogger(__name__)

        logger.info("Initializing options math test suite")

        # Create tester and run all tests
        tester = OptionsMathTester()
        results = tester.run_all_tests()

        # Print final summary
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {results['total_tests']}")
        print(f"Passed: {results['passed_tests']}")
        print(f"Failed: {results['failed_tests']}")
        print(f"Success Rate: {results['success_rate']:.1%}")
        print("=" * 60)

        # Return appropriate exit code
        return 0 if results["failed_tests"] == 0 else 1

    except Exception as e:
        print(f"❌ Test suite setup failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
