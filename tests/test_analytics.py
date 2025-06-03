"""
Simple test script to verify Google Analytics integration.

Run this with: streamlit run test_analytics.py
"""

import streamlit as st

from buffetbot.dashboard.components.analytics import (
    initialize_analytics,
    track_custom_event,
    track_page_view,
    track_ticker_analysis,
)


def main():
    st.title("Google Analytics Integration Test")

    # Initialize analytics
    initialize_analytics(environment="production")

    st.success("✅ Google Analytics initialized!")

    # Test page view tracking
    if st.button("Test Page View Tracking"):
        track_page_view("Test Page")
        st.info("📊 Page view tracked!")

    # Test custom event tracking
    if st.button("Test Custom Event"):
        track_custom_event("test_event", {"test_parameter": "test_value"})
        st.info("🎯 Custom event tracked!")

    # Test ticker analysis tracking
    ticker = st.text_input("Enter a ticker to test tracking:", value="AAPL")
    if st.button("Test Ticker Analysis Tracking"):
        track_ticker_analysis(ticker, "test_analysis")
        st.info(f"📈 Ticker analysis tracked for {ticker}!")

    st.markdown(
        """
    ## How to Verify

    1. **Open browser developer tools** (F12)
    2. **Go to Console tab**
    3. **Click the test buttons above**
    4. **Look for Google Analytics events** in the console
    5. **Check for `gtag` function calls**

    ## Expected Console Output
    ```
    Google Analytics loaded
    gtag('config', 'G-ZCCK6W5VEF');
    gtag('event', 'page_view', ...);
    ```

    ## In Google Analytics
    - Go to **Realtime** → **Events**
    - You should see the custom events appear
    - May take a few minutes to show up
    """
    )


if __name__ == "__main__":
    main()
