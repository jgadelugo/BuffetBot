import React, { useState, useRef, useEffect, memo } from 'react';
import PropTypes from 'prop-types';
import './MetricTooltip.css';

// Import glossary data - adjust path based on your build setup
// This assumes the glossary data is available as a module or passed as props
const GLOSSARY = window.GLOSSARY || {};

/**
 * MetricTooltip Component
 *
 * A reusable tooltip component that displays metric information from the glossary.
 * Features:
 * - Accessible (ARIA attributes, keyboard navigation)
 * - Responsive positioning
 * - Memoized for performance
 * - Customizable styling
 */
const MetricTooltip = memo(({
  metricKey,
  children,
  className = '',
  position = 'top',
  showIcon = true,
  iconContent = 'ℹ️',
  glossaryData = GLOSSARY
}) => {
  const [isVisible, setIsVisible] = useState(false);
  const [tooltipPosition, setTooltipPosition] = useState(position);
  const tooltipRef = useRef(null);
  const triggerRef = useRef(null);

  // Get metric information from glossary
  const metricInfo = glossaryData[metricKey] || {
    name: metricKey,
    description: 'No description available',
    category: 'unknown'
  };

  // Calculate tooltip position to ensure it stays within viewport
  useEffect(() => {
    if (isVisible && tooltipRef.current && triggerRef.current) {
      const tooltipRect = tooltipRef.current.getBoundingClientRect();
      const triggerRect = triggerRef.current.getBoundingClientRect();
      const viewportWidth = window.innerWidth;
      const viewportHeight = window.innerHeight;

      let newPosition = position;

      // Check if tooltip goes outside viewport and adjust position
      if (position === 'top' && triggerRect.top - tooltipRect.height < 0) {
        newPosition = 'bottom';
      } else if (position === 'bottom' && triggerRect.bottom + tooltipRect.height > viewportHeight) {
        newPosition = 'top';
      } else if (position === 'left' && triggerRect.left - tooltipRect.width < 0) {
        newPosition = 'right';
      } else if (position === 'right' && triggerRect.right + tooltipRect.width > viewportWidth) {
        newPosition = 'left';
      }

      // Additional check for horizontal overflow
      if ((newPosition === 'top' || newPosition === 'bottom') &&
          triggerRect.left + tooltipRect.width / 2 > viewportWidth) {
        tooltipRef.current.style.left = 'auto';
        tooltipRef.current.style.right = '0';
      }

      setTooltipPosition(newPosition);
    }
  }, [isVisible, position]);

  // Handle keyboard navigation
  const handleKeyDown = (e) => {
    if (e.key === 'Escape' && isVisible) {
      setIsVisible(false);
      triggerRef.current?.focus();
    }
  };

  // Show tooltip
  const showTooltip = () => setIsVisible(true);

  // Hide tooltip
  const hideTooltip = () => setIsVisible(false);

  return (
    <span
      className={`metric-tooltip-container ${className}`}
      onMouseEnter={showTooltip}
      onMouseLeave={hideTooltip}
      onFocus={showTooltip}
      onBlur={hideTooltip}
    >
      <span
        ref={triggerRef}
        className="metric-tooltip-trigger"
        role="button"
        tabIndex={0}
        aria-label={`Information about ${metricInfo.name}`}
        aria-describedby={isVisible ? `tooltip-${metricKey}` : undefined}
        onKeyDown={handleKeyDown}
      >
        {children || metricInfo.name}
        {showIcon && <span className="metric-tooltip-icon" aria-hidden="true">{iconContent}</span>}
      </span>

      {isVisible && (
        <div
          ref={tooltipRef}
          id={`tooltip-${metricKey}`}
          role="tooltip"
          className={`metric-tooltip metric-tooltip-${tooltipPosition} metric-tooltip-${metricInfo.category}`}
          aria-live="polite"
        >
          <div className="metric-tooltip-content">
            <h4 className="metric-tooltip-title">{metricInfo.name}</h4>
            <p className="metric-tooltip-description">{metricInfo.description}</p>
            {metricInfo.formula && (
              <div className="metric-tooltip-formula">
                <strong>Formula:</strong> <code>{metricInfo.formula}</code>
              </div>
            )}
            {metricInfo.good_range && (
              <div className="metric-tooltip-range">
                <strong>Good range:</strong> {metricInfo.good_range}
              </div>
            )}
          </div>
          <div className={`metric-tooltip-arrow metric-tooltip-arrow-${tooltipPosition}`} />
        </div>
      )}
    </span>
  );
});

MetricTooltip.displayName = 'MetricTooltip';

MetricTooltip.propTypes = {
  metricKey: PropTypes.string.isRequired,
  children: PropTypes.node,
  className: PropTypes.string,
  position: PropTypes.oneOf(['top', 'bottom', 'left', 'right']),
  showIcon: PropTypes.bool,
  iconContent: PropTypes.node,
  glossaryData: PropTypes.object
};

export default MetricTooltip;

// Export a higher-order component for easy integration
export const withMetricTooltip = (Component) => {
  const WrappedComponent = ({ metricKey, ...props }) => (
    <MetricTooltip metricKey={metricKey}>
      <Component {...props} />
    </MetricTooltip>
  );

  WrappedComponent.displayName = `withMetricTooltip(${Component.displayName || Component.name || 'Component'})`;

  return WrappedComponent;
};
