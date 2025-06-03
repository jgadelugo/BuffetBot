import React, { useState, useMemo } from 'react';
import './GlossaryComponent.css'; // CSS file for styling

// Import glossary data - in a real app this would come from an API or module
import { GLOSSARY } from './glossaryData';

const GlossaryComponent = () => {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [expandedCategories, setExpandedCategories] = useState({
    growth: true,
    value: true,
    health: true,
    risk: true
  });

  // Category configuration
  const categoryConfig = {
    growth: { color: '#10b981', bgColor: '#d1fae5', icon: '📈', label: 'Growth' },
    value: { color: '#3b82f6', bgColor: '#dbeafe', icon: '💰', label: 'Value' },
    health: { color: '#f59e0b', bgColor: '#fef3c7', icon: '💪', label: 'Health' },
    risk: { color: '#ef4444', bgColor: '#fee2e2', icon: '⚠️', label: 'Risk' }
  };

  // Filter metrics based on search and category
  const filteredMetrics = useMemo(() => {
    let filtered = Object.entries(GLOSSARY);

    // Category filter
    if (selectedCategory !== 'all') {
      filtered = filtered.filter(([_, metric]) => metric.category === selectedCategory);
    }

    // Search filter
    if (searchTerm) {
      const searchLower = searchTerm.toLowerCase();
      filtered = filtered.filter(([_, metric]) =>
        metric.name.toLowerCase().includes(searchLower) ||
        metric.description.toLowerCase().includes(searchLower)
      );
    }

    return filtered;
  }, [searchTerm, selectedCategory]);

  // Group metrics by category
  const groupedMetrics = useMemo(() => {
    const groups = { growth: [], value: [], health: [], risk: [] };

    filteredMetrics.forEach(([key, metric]) => {
      if (groups[metric.category]) {
        groups[metric.category].push({ key, ...metric });
      }
    });

    return groups;
  }, [filteredMetrics]);

  // Calculate statistics
  const stats = useMemo(() => {
    const total = Object.keys(GLOSSARY).length;
    const byCategory = Object.values(GLOSSARY).reduce((acc, metric) => {
      acc[metric.category] = (acc[metric.category] || 0) + 1;
      return acc;
    }, {});

    return { total, byCategory };
  }, []);

  const toggleCategory = (category) => {
    setExpandedCategories(prev => ({
      ...prev,
      [category]: !prev[category]
    }));
  };

  const highlightText = (text, highlight) => {
    if (!highlight) return text;

    const parts = text.split(new RegExp(`(${highlight})`, 'gi'));
    return parts.map((part, index) =>
      part.toLowerCase() === highlight.toLowerCase()
        ? <mark key={index} className="highlight">{part}</mark>
        : part
    );
  };

  const MetricCard = ({ metric }) => {
    const config = categoryConfig[metric.category];

    return (
      <div
        className="metric-card"
        style={{ borderLeftColor: config.color }}
      >
        <div className="metric-header">
          <h3 className="metric-name">
            {highlightText(metric.name, searchTerm)}
          </h3>
          <span
            className="category-badge"
            style={{
              backgroundColor: config.bgColor,
              color: config.color
            }}
          >
            {config.icon} {metric.category.toUpperCase()}
          </span>
        </div>
        <p className="metric-description">
          {highlightText(metric.description, searchTerm)}
        </p>
        <div className="metric-formula">
          <strong>Formula:</strong>
          <code>{metric.formula}</code>
        </div>
      </div>
    );
  };

  const exportToCSV = () => {
    const rows = [['Key', 'Name', 'Category', 'Description', 'Formula']];

    Object.entries(GLOSSARY).forEach(([key, metric]) => {
      rows.push([key, metric.name, metric.category, metric.description, metric.formula]);
    });

    const csv = rows.map(row => row.map(cell => `"${cell}"`).join(',')).join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'financial_metrics_glossary.csv';
    a.click();
    URL.revokeObjectURL(url);
  };

  const exportToJSON = () => {
    const json = JSON.stringify(GLOSSARY, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'financial_metrics_glossary.json';
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="glossary-container">
      {/* Header */}
      <header className="glossary-header">
        <h1>📊 Financial Metrics Glossary</h1>
        <p>Comprehensive guide to financial metrics and KPIs used in value investing analysis</p>
      </header>

      <div className="glossary-content">
        {/* Sidebar */}
        <aside className="glossary-sidebar">
          {/* Search */}
          <div className="sidebar-section">
            <h3>🔍 Search</h3>
            <input
              type="text"
              placeholder="Search metrics..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="search-input"
            />
          </div>

          {/* Category Filter */}
          <div className="sidebar-section">
            <h3>🏷️ Categories</h3>
            <div className="category-filters">
              <label className="category-option">
                <input
                  type="radio"
                  name="category"
                  value="all"
                  checked={selectedCategory === 'all'}
                  onChange={(e) => setSelectedCategory(e.target.value)}
                />
                <span>All Categories</span>
              </label>
              {Object.entries(categoryConfig).map(([key, config]) => (
                <label key={key} className="category-option">
                  <input
                    type="radio"
                    name="category"
                    value={key}
                    checked={selectedCategory === key}
                    onChange={(e) => setSelectedCategory(e.target.value)}
                  />
                  <span style={{ color: config.color }}>
                    {config.icon} {config.label}
                  </span>
                </label>
              ))}
            </div>
          </div>

          {/* Statistics */}
          <div className="sidebar-section">
            <h3>📈 Statistics</h3>
            <div className="stats">
              <div className="stat-item">
                <span>Total Metrics:</span>
                <strong>{stats.total}</strong>
              </div>
              {Object.entries(stats.byCategory).map(([category, count]) => (
                <div key={category} className="stat-item">
                  <span>{categoryConfig[category].label}:</span>
                  <strong>{count}</strong>
                </div>
              ))}
            </div>
          </div>

          {/* Export */}
          <div className="sidebar-section">
            <h3>📥 Export</h3>
            <button onClick={exportToCSV} className="export-button csv">
              Export as CSV
            </button>
            <button onClick={exportToJSON} className="export-button json">
              Export as JSON
            </button>
          </div>
        </aside>

        {/* Main Content */}
        <main className="glossary-main">
          {searchTerm && (
            <p className="search-results">
              Found {filteredMetrics.length} metrics matching "{searchTerm}"
            </p>
          )}

          {selectedCategory === 'all' && !searchTerm ? (
            // Grouped view
            Object.entries(groupedMetrics).map(([category, metrics]) => {
              if (metrics.length === 0) return null;

              const config = categoryConfig[category];
              const isExpanded = expandedCategories[category];

              return (
                <div key={category} className="category-group">
                  <h2
                    className="category-header"
                    onClick={() => toggleCategory(category)}
                  >
                    <span>
                      {config.icon} {category.toUpperCase()} METRICS
                      <span className="metric-count">({metrics.length})</span>
                    </span>
                    <span className="expand-icon">
                      {isExpanded ? '▼' : '▶'}
                    </span>
                  </h2>
                  {isExpanded && (
                    <div className="metrics-grid">
                      {metrics.map(metric => (
                        <MetricCard key={metric.key} metric={metric} />
                      ))}
                    </div>
                  )}
                </div>
              );
            })
          ) : (
            // Flat view for search/filter results
            <div className="metrics-grid">
              {filteredMetrics.map(([key, metric]) => (
                <MetricCard key={key} metric={{ key, ...metric }} />
              ))}
            </div>
          )}

          {filteredMetrics.length === 0 && (
            <p className="no-results">No metrics found matching your criteria.</p>
          )}
        </main>
      </div>

      {/* Footer */}
      <footer className="glossary-footer">
        <p>Financial Metrics Glossary v1.0</p>
        <p>💡 Tip: Use the search bar to find specific metrics or filter by category</p>
      </footer>
    </div>
  );
};

export default GlossaryComponent;
