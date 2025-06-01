import React from 'react';
import MetricTooltip, { withMetricTooltip } from './MetricTooltip';

/**
 * MetricTooltip Usage Examples
 * 
 * This file demonstrates various ways to use the MetricTooltip component
 * in your dashboards and metric tables.
 */

// Example 1: Basic Usage in a Metric Table
export const MetricTableExample = () => {
  const metrics = [
    { key: 'pe_ratio', value: 18.5 },
    { key: 'pb_ratio', value: 3.2 },
    { key: 'return_on_equity', value: 0.22 },
    { key: 'debt_to_equity', value: 0.6 }
  ];

  return (
    <table className="metric-table">
      <thead>
        <tr>
          <th>Metric</th>
          <th>Value</th>
        </tr>
      </thead>
      <tbody>
        {metrics.map(({ key, value }) => (
          <tr key={key}>
            <td>
              <MetricTooltip metricKey={key} />
            </td>
            <td>{value}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
};

// Example 2: Custom Styling and Positioning
export const DashboardCardExample = () => {
  return (
    <div className="dashboard-card">
      <h3>
        <MetricTooltip 
          metricKey="revenue_growth" 
          className="metric-header"
          position="right"
        >
          Revenue Growth
        </MetricTooltip>
      </h3>
      <div className="metric-value">15.2%</div>
      
      <div className="metric-detail">
        <MetricTooltip 
          metricKey="gross_margin" 
          showIcon={false}
          className="subtle-metric"
        >
          <span className="metric-label">Gross Margin: </span>
        </MetricTooltip>
        <span className="metric-value">35.8%</span>
      </div>
    </div>
  );
};

// Example 3: Using the Higher-Order Component
const MetricLabel = ({ children, value }) => (
  <div className="metric-label-container">
    <span className="label">{children}</span>
    <span className="value">{value}</span>
  </div>
);

const EnhancedMetricLabel = withMetricTooltip(MetricLabel);

export const HOCExample = () => {
  return (
    <div className="metric-grid">
      <EnhancedMetricLabel metricKey="current_ratio" value="1.8">
        Current Ratio
      </EnhancedMetricLabel>
      
      <EnhancedMetricLabel metricKey="beta" value="1.2">
        Beta
      </EnhancedMetricLabel>
      
      <EnhancedMetricLabel metricKey="operating_margin" value="0.18">
        Operating Margin
      </EnhancedMetricLabel>
    </div>
  );
};

// Example 4: Custom Icon and Dynamic Glossary Data
export const CustomIconExample = () => {
  // You might fetch this from an API or context
  const customGlossary = {
    custom_metric: {
      name: "Custom Metric",
      description: "This is a custom metric specific to our analysis",
      category: "custom",
      formula: "Custom calculation"
    }
  };

  return (
    <div className="custom-metrics">
      <MetricTooltip 
        metricKey="custom_metric"
        iconContent="❓"
        glossaryData={customGlossary}
      >
        Custom Analysis Score
      </MetricTooltip>
      
      <MetricTooltip 
        metricKey="volatility"
        iconContent={<span style={{ fontSize: '0.8em' }}>📊</span>}
      >
        Volatility Index
      </MetricTooltip>
    </div>
  );
};

// Example 5: Integration with Chart Libraries
export const ChartIntegrationExample = () => {
  return (
    <div className="chart-container">
      <div className="chart-header">
        <MetricTooltip metricKey="price_to_earnings_growth">
          PEG Ratio Trend
        </MetricTooltip>
      </div>
      {/* Your chart component here */}
      <div className="chart-placeholder">
        [Chart Component]
      </div>
      <div className="chart-legend">
        <MetricTooltip 
          metricKey="pe_ratio" 
          position="bottom"
          className="legend-item"
        >
          <span className="legend-color" style={{ backgroundColor: '#4a90e2' }}></span>
          P/E Ratio
        </MetricTooltip>
        <MetricTooltip 
          metricKey="earnings_growth" 
          position="bottom"
          className="legend-item"
        >
          <span className="legend-color" style={{ backgroundColor: '#7cb342' }}></span>
          Earnings Growth
        </MetricTooltip>
      </div>
    </div>
  );
};

// Example 6: Responsive Mobile View
export const MobileResponsiveExample = () => {
  return (
    <div className="mobile-metric-list">
      <div className="metric-row">
        <MetricTooltip 
          metricKey="market_cap"
          position="bottom"  // Better for mobile
        >
          Market Cap
        </MetricTooltip>
        <span className="metric-value">$2.5T</span>
      </div>
      
      <div className="metric-row">
        <MetricTooltip 
          metricKey="free_cash_flow"
          position="bottom"
        >
          FCF
        </MetricTooltip>
        <span className="metric-value">$95.2B</span>
      </div>
    </div>
  );
};

// Example styles for the examples (would typically be in a separate CSS file)
const exampleStyles = `
  .metric-table {
    width: 100%;
    border-collapse: collapse;
  }
  
  .metric-table th,
  .metric-table td {
    padding: 12px;
    text-align: left;
    border-bottom: 1px solid #e0e0e0;
  }
  
  .dashboard-card {
    padding: 20px;
    border-radius: 8px;
    background: #f5f5f5;
  }
  
  .metric-value {
    font-size: 24px;
    font-weight: bold;
    color: #333;
  }
  
  .metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 20px;
  }
  
  .metric-label-container {
    display: flex;
    justify-content: space-between;
    padding: 10px;
    background: white;
    border-radius: 4px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  }
  
  .legend-item {
    display: inline-flex;
    align-items: center;
    margin-right: 20px;
  }
  
  .legend-color {
    width: 12px;
    height: 12px;
    border-radius: 2px;
    margin-right: 8px;
  }
  
  .mobile-metric-list .metric-row {
    display: flex;
    justify-content: space-between;
    padding: 16px 0;
    border-bottom: 1px solid #e0e0e0;
  }
`; 