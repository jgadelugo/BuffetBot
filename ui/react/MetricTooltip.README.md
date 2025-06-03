# MetricTooltip Component

A reusable, accessible, and responsive tooltip component for displaying financial metric information from the BuffetBot glossary.

## Features

- **Accessible**: Full keyboard navigation and ARIA attributes
- **Responsive**: Automatic positioning adjustment to stay within viewport
- **Performant**: Memoized with React.memo for optimal performance
- **Customizable**: Flexible styling and positioning options
- **Category-aware**: Color-coded based on metric categories (growth, value, health, risk)
- **Dark mode support**: Automatically adapts to system preferences
- **Mobile-friendly**: Optimized for touch devices

## Installation

The component is already included in the BuffetBot UI package. To use it in your project:

```jsx
import MetricTooltip from '@buffetbot/ui/react/MetricTooltip';
// or
import { MetricTooltip } from '@buffetbot/ui';
```

## Basic Usage

```jsx
// Simple usage - displays metric name with info icon
<MetricTooltip metricKey="pe_ratio" />

// With custom text
<MetricTooltip metricKey="revenue_growth">
  Revenue Growth Rate
</MetricTooltip>

// Without icon
<MetricTooltip metricKey="debt_to_equity" showIcon={false}>
  D/E Ratio
</MetricTooltip>
```

## Props

| Prop | Type | Default | Description |
|------|------|---------|-------------|
| `metricKey` | `string` | **required** | Key from the glossary data |
| `children` | `ReactNode` | metric name | Content to display as the trigger |
| `className` | `string` | `''` | Additional CSS classes |
| `position` | `'top' \| 'bottom' \| 'left' \| 'right'` | `'top'` | Preferred tooltip position |
| `showIcon` | `boolean` | `true` | Whether to show the info icon |
| `iconContent` | `ReactNode` | `'ℹ️'` | Custom icon content |
| `glossaryData` | `object` | `GLOSSARY` | Custom glossary data object |

## Advanced Usage

### Higher-Order Component

For wrapping existing components with tooltip functionality:

```jsx
import { withMetricTooltip } from '@buffetbot/ui/react/MetricTooltip';

const MyMetricDisplay = ({ value, children }) => (
  <div className="metric">
    <span>{children}</span>
    <span>{value}</span>
  </div>
);

const EnhancedMetricDisplay = withMetricTooltip(MyMetricDisplay);

// Usage
<EnhancedMetricDisplay metricKey="current_ratio" value="1.8">
  Current Ratio
</EnhancedMetricDisplay>
```

### Custom Glossary Data

```jsx
const customGlossary = {
  custom_metric: {
    name: "Custom Metric",
    description: "Description of the custom metric",
    category: "custom",
    formula: "Custom formula"
  }
};

<MetricTooltip
  metricKey="custom_metric"
  glossaryData={customGlossary}
>
  My Custom Metric
</MetricTooltip>
```

### Custom Icons

```jsx
// Text icon
<MetricTooltip metricKey="volatility" iconContent="📊">
  Volatility
</MetricTooltip>

// React component icon
<MetricTooltip
  metricKey="beta"
  iconContent={<InfoIcon size={14} />}
>
  Beta
</MetricTooltip>
```

## Styling

The component comes with default styles but can be customized:

### CSS Classes

- `.metric-tooltip-container` - Main container
- `.metric-tooltip-trigger` - Clickable/hoverable element
- `.metric-tooltip` - Tooltip popup
- `.metric-tooltip-{position}` - Position-specific styles
- `.metric-tooltip-{category}` - Category-specific styles (growth, value, health, risk)

### Custom Styling Example

```css
/* Custom trigger styling */
.my-custom-tooltip .metric-tooltip-trigger {
  color: #007bff;
  border-bottom-style: solid;
}

/* Custom tooltip styling */
.my-custom-tooltip .metric-tooltip {
  background: #f8f9fa;
  max-width: 400px;
}

/* Category-specific colors */
.metric-tooltip-growth {
  border-left-color: #28a745;
}
```

## Accessibility

The component follows WAI-ARIA guidelines:

- Keyboard navigable (Tab, Escape)
- Screen reader announcements
- Focus management
- High contrast mode support
- Reduced motion support

## Performance Considerations

1. **Memoization**: The component is wrapped in `React.memo` to prevent unnecessary re-renders
2. **Lazy positioning**: Viewport calculations only occur when tooltip is visible
3. **Event delegation**: Uses native browser events for optimal performance

## Browser Support

- Modern browsers (Chrome, Firefox, Safari, Edge)
- IE11 (with polyfills)
- Mobile browsers (iOS Safari, Chrome Android)

## Examples

See `MetricTooltip.examples.jsx` for comprehensive usage examples including:

1. Basic metric tables
2. Dashboard cards
3. Chart integrations
4. Mobile-responsive layouts
5. Custom styling
6. HOC usage

## Integration with BuffetBot

The component automatically integrates with the BuffetBot glossary data structure:

```javascript
{
  "metric_key": {
    "name": "Display Name",
    "description": "Detailed description",
    "formula": "Calculation formula",
    "category": "growth|value|health|risk",
    "good_range": "Optimal value range",
    "related_metrics": ["related1", "related2"]
  }
}
```

## Troubleshooting

### Tooltip not showing
- Ensure `metricKey` exists in glossary data
- Check z-index conflicts with other elements
- Verify JavaScript is enabled

### Position issues
- Component automatically adjusts position to stay in viewport
- Use `position` prop for preferred placement
- Consider mobile viewport constraints

### Performance issues
- Ensure component is properly memoized
- Avoid passing new object references as props
- Use production React build

## Future Enhancements

- [ ] Keyboard shortcut to show all tooltips
- [ ] Tooltip delay configuration
- [ ] Animation customization
- [ ] Multi-language support
- [ ] Tooltip pinning/persistent mode
