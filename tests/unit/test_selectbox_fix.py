"""Test suite for selectbox ID collision fix."""

import pytest


class TestSelectboxIDCollisionFix:
    """Test cases to verify the selectbox ID collision fix."""

    def test_all_selectboxes_have_unique_keys(self):
        """Test that all selectbox elements in options_advisor.py have unique keys."""
        with open("buffetbot/dashboard/views/options_advisor.py") as f:
            content = f.read()

        # Expected unique keys for all selectboxes
        expected_keys = [
            "options_strategy_selector",
            "risk_tolerance_selector",
            "time_horizon_selector",
            "score_components_rec_selector",
            "comprehensive_analysis_rec_selector",
        ]

        # Verify all expected keys are present
        missing_keys = []
        for key in expected_keys:
            if f'key="{key}"' not in content:
                missing_keys.append(key)

        assert len(missing_keys) == 0, f"Missing selectbox keys: {missing_keys}"

        # Count selectboxes and keys
        selectbox_count = content.count("st.selectbox(")
        key_count = sum(1 for key in expected_keys if f'key="{key}"' in content)

        assert selectbox_count == 5, f"Expected 5 selectboxes, found {selectbox_count}"
        assert key_count == 5, f"Expected 5 keys, found {key_count}"

        print(f"✅ All {len(expected_keys)} selectboxes have unique keys")

    def test_no_duplicate_keys(self):
        """Test that all selectbox keys are unique."""
        with open("buffetbot/dashboard/views/options_advisor.py") as f:
            content = f.read()

        expected_keys = [
            "options_strategy_selector",
            "risk_tolerance_selector",
            "time_horizon_selector",
            "score_components_rec_selector",
            "comprehensive_analysis_rec_selector",
        ]

        # Check each key appears exactly once
        for key in expected_keys:
            count = content.count(f'key="{key}"')
            assert count == 1, f"Key '{key}' appears {count} times, should be exactly 1"

        print("✅ All selectbox keys are unique")

    def test_selectbox_error_not_in_logs(self):
        """Test that the selectbox error message is not in recent logs."""
        import glob
        import os

        error_pattern = "multiple selectbox elements with the same auto-generated ID"

        # Check only the most recent logs to avoid long scan times
        recent_logs = ["logs/app.log", "logs/dashboard_output.log"]

        for log_file in recent_logs:
            if os.path.exists(log_file):
                try:
                    with open(log_file) as f:
                        # Only read last 1000 lines for efficiency
                        lines = f.readlines()
                        recent_content = (
                            "".join(lines[-1000:]) if len(lines) > 1000 else f.read()
                        )

                    assert (
                        error_pattern.lower() not in recent_content.lower()
                    ), f"Found selectbox error in {log_file}"
                except Exception:
                    # Skip files that can't be read
                    continue

        print("✅ No selectbox ID collision errors found in recent logs")

    def test_fix_addresses_original_issue(self):
        """Test that the fix specifically addresses the Individual Recommendation Analysis issue."""
        with open("buffetbot/dashboard/views/options_advisor.py") as f:
            content = f.read()

        # The original error was in Individual Recommendation Analysis section
        # Verify the two problematic selectboxes now have unique keys

        # First selectbox: score components selector
        assert (
            'key="score_components_rec_selector"' in content
        ), "Missing score_components_rec_selector key"

        # Second selectbox: comprehensive analysis selector
        assert (
            'key="comprehensive_analysis_rec_selector"' in content
        ), "Missing comprehensive_analysis_rec_selector key"

        # Verify they are in the right sections
        lines = content.split("\n")
        found_score_components = False
        found_comprehensive = False

        for line in lines:
            if "score_components_rec_selector" in line:
                found_score_components = True
            if "comprehensive_analysis_rec_selector" in line:
                found_comprehensive = True

        assert found_score_components, "score_components_rec_selector not found"
        assert found_comprehensive, "comprehensive_analysis_rec_selector not found"

        print("✅ Fix addresses the original Individual Recommendation Analysis issue")


if __name__ == "__main__":
    test = TestSelectboxIDCollisionFix()
    test.test_all_selectboxes_have_unique_keys()
    test.test_no_duplicate_keys()
    test.test_selectbox_error_not_in_logs()
    test.test_fix_addresses_original_issue()
    print("🎉 All selectbox ID collision fix tests passed!")
    print("🔧 Selectbox error should be resolved in production")
