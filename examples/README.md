# BuffetBot Examples

This directory contains example scripts demonstrating how to use the BuffetBot financial analysis toolkit.

## Available Examples

### 1. `example_integration.py`
Demonstrates how to integrate the glossary module with analysis results to provide enhanced context and interpretations.

**Features demonstrated:**
- Enhancing raw analysis results with metric definitions
- Adding interpretations based on metric values
- Generating formatted reports with explanations
- Combining multiple analysis modules

**To run:**
```bash
# From the BuffetBot root directory
python examples/example_integration.py
```

## Example Output

```
=== Enhanced Analysis Output ===

Metric Information:

Revenue Growth Rate: 0.15
  Interpretation: Strong growth

Current Ratio: 1.80
  Interpretation: Good liquidity

...
```

## Creating New Examples

When adding new example scripts:

1. Name files descriptively (e.g., `example_dashboard.py`, `example_api_integration.py`)
2. Include a module docstring explaining what the example demonstrates
3. Add comments explaining key concepts
4. Use realistic sample data
5. Show both basic and advanced usage patterns
6. Include error handling examples where appropriate

## Best Practices for Examples

1. **Self-contained**: Examples should run without external dependencies beyond the BuffetBot modules
2. **Well-documented**: Include comments explaining what each section does
3. **Realistic**: Use data that resembles real financial metrics
4. **Progressive**: Start with simple usage and build to more complex scenarios
5. **Output-focused**: Show clear output so users can see what to expect

## Integration Patterns

Common patterns demonstrated in examples:

- **Data Enhancement**: Adding context to raw numerical results
- **Report Generation**: Creating human-readable reports from analysis data
- **Metric Interpretation**: Providing business meaning to calculated values
- **Cross-module Integration**: Combining results from multiple analysis modules

## Future Examples

Planned examples to be added:
- Web API integration example
- Dashboard visualization example
- Batch processing example
- Custom metric definition example
- Real-time monitoring example
