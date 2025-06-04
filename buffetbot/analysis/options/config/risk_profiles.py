"""
Risk profiles configuration for different risk tolerance levels.

This module provides risk profile configurations that define filtering
and selection criteria for different risk tolerance levels.
"""

from dataclasses import dataclass
from typing import Dict

from ..core.domain_models import RiskProfile, RiskTolerance


@dataclass
class RiskProfilesConfig:
    """Configuration for risk profiles across different tolerance levels."""

    profiles: dict[RiskTolerance, RiskProfile]

    def get_profile(self, risk_tolerance: RiskTolerance) -> RiskProfile:
        """Get risk profile for a given tolerance level."""
        return self.profiles[risk_tolerance]

    def validate(self) -> None:
        """Validate all risk profiles."""
        # Risk profiles are validated in RiskProfile __post_init__
        pass


# Default risk profiles configuration
DEFAULT_RISK_PROFILES_CONFIG = RiskProfilesConfig(
    profiles={
        RiskTolerance.CONSERVATIVE: RiskProfile(
            max_delta_threshold=0.30,  # Lower delta limit
            min_days_to_expiry=60,  # Longer timeframes
            max_moneyness_range=0.15,  # Closer to ATM
            volume_threshold=50,  # Higher volume requirement
            open_interest_threshold=100,  # Higher OI requirement
            max_bid_ask_spread=0.05,  # Tighter spreads
        ),
        RiskTolerance.MODERATE: RiskProfile(
            max_delta_threshold=0.50,  # Moderate delta limit
            min_days_to_expiry=30,  # Medium timeframes
            max_moneyness_range=0.25,  # Moderate moneyness range
            volume_threshold=25,  # Moderate volume requirement
            open_interest_threshold=50,  # Moderate OI requirement
            max_bid_ask_spread=0.10,  # Moderate spreads
        ),
        RiskTolerance.AGGRESSIVE: RiskProfile(
            max_delta_threshold=0.70,  # Higher delta allowed
            min_days_to_expiry=15,  # Shorter timeframes allowed
            max_moneyness_range=0.40,  # Wider moneyness range
            volume_threshold=10,  # Lower volume requirement
            open_interest_threshold=25,  # Lower OI requirement
            max_bid_ask_spread=0.20,  # Wider spreads allowed
        ),
    }
)


def get_risk_profile(risk_tolerance: RiskTolerance) -> RiskProfile:
    """
    Get risk profile for a given risk tolerance level.

    Args:
        risk_tolerance: Risk tolerance level

    Returns:
        RiskProfile: Risk profile configuration
    """
    return DEFAULT_RISK_PROFILES_CONFIG.get_profile(risk_tolerance)


def get_ecosystem_scoring_weights() -> dict[str, float]:
    """
    Get ecosystem scoring multipliers.

    Returns:
        Dict[str, float]: Ecosystem scoring weights
    """
    return {
        "confirm": 1.1,  # Boost score by 10% for ecosystem confirmation
        "neutral": 1.0,  # No adjustment for neutral ecosystem signal
        "veto": 0.9,  # Reduce score by 10% for ecosystem veto
    }
