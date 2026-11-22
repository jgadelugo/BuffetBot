# 📊 BuffetBot – Value Investing Stock Analysis Dashboard

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

BuffetBot is a Streamlit dashboard for analysing a stock using Warren Buffett-style value investing principles. Enter a ticker and it pulls live price history, financial statements and options data from Yahoo Finance (via `yfinance`) and shows growth, valuation, financial health, risk and options analysis.

No API keys or database are needed.

---

## 🧭 What the dashboard shows

| Tab | What you get |
| --- | --- |
| Overview | Price and daily change, market cap, P/E, annualized volatility, RSI, data-quality score |
| Price Analysis | Intrinsic value estimate, margin of safety, over/undervalued verdict, technical chart |
| Financial Health | Piotroski F-Score, Altman Z-Score, liquidity / leverage / profitability ratios |
| Growth Metrics | Revenue and earnings growth |
| Risk Analysis | Market, financial and business risk scores |
| Options Advisor | Ranked options strategies scored against a risk profile |
| 🔮 Analyst Forecast | Price targets, recommendations and consensus |
| 📚 Glossary | Definitions for every metric |

---

## 🚀 Quick Start

Requires Python 3.10 or later.

**Windows (PowerShell)**

```powershell
python -m venv venv
venv\Scripts\activate
pip install -e .
$env:STREAMLIT_ENV = "development"; streamlit run main.py
```

**macOS / Linux**

```bash
python -m venv venv
source venv/bin/activate
pip install -e .
./run_dashboard.sh
```

Then open http://localhost:8501.

`pip install -e .` installs everything the dashboard and CLI need. If you only want the dashboard dependencies, `pip install -r requirements.txt` works too.

Other ways to start it:

- `buffetbot-dashboard` (after `pip install -e .`)
- `python scripts/run_dashboard.py`

---

## ⚙️ Configuration

Nothing is required. Optional environment variables:

| Variable | Effect |
| --- | --- |
| `STREAMLIT_ENV` | `development` turns Google Analytics off; `production` (the default when unset) turns it on. The run scripts and `buffetbot-dashboard` set `development` for you. |
| `YFINANCE_TIMEOUT`, `YFINANCE_RETRIES` | Yahoo Finance request tuning |
| `DEFAULT_TICKER`, `DEFAULT_GROWTH_RATE`, `DEFAULT_DISCOUNT_RATE` | Analysis defaults |
| `CACHE_ENABLED`, `CACHE_EXPIRATION` | Data caching |

Streamlit server and theme settings live in `.streamlit/config.toml`.

---

## 🧭 CLI

```bash
buffetbot recommend AAPL --min-days 180 --top-n 5 --output picks.csv
buffetbot version
```

`recommend` ranks long-dated call options for a ticker using RSI, beta, momentum and implied volatility.

---

## 📂 Project Structure

```
main.py                     # Streamlit entry point (also used by Streamlit Cloud)
buffetbot/
├── dashboard/              # Streamlit app: app.py, views/ (one per tab), components/
├── analysis/               # growth, value, health, risk and options analysis
├── data/                   # Yahoo Finance fetchers and data cleaning
├── recommend/              # recommendation engine
├── utils/                  # logging, caching, validation, maths helpers
└── cli.py                  # `buffetbot` command line tool
requirements/               # base / prod / dev / test dependency sets
scripts/                    # run and test helper scripts
tests/                      # pytest suite (unit + integration)
docs/                       # setup, deployment, architecture and feature docs
```

---

## 🎨 UI Components

The project includes several UI implementations for visualizing financial metrics:

### React Components

#### MetricTooltip
A reusable, accessible tooltip component for displaying metric information inline.

```jsx
import MetricTooltip from './ui/react/MetricTooltip';

// Basic usage
<MetricTooltip metricKey="pe_ratio" />

// Custom text with tooltip
<MetricTooltip metricKey="revenue_growth">
  Revenue Growth Rate
</MetricTooltip>
```

**Features:**
- Fully accessible (ARIA attributes, keyboard navigation)
- Responsive positioning
- Dark mode support
- Memoized for performance
- Category-aware styling

#### GlossaryComponent
A comprehensive glossary viewer with search and filtering capabilities.

### Other UI Implementations
- **Streamlit App** (`ui/streamlit/glossary_app.py`): Interactive Python web app
- **Standalone HTML** (`ui/web/glossary.html`): No-server-required web page
- **React Glossary** (`ui/react/GlossaryComponent.jsx`): Full glossary viewer

See `/docs/glossary/ui_implementations.md` for detailed UI documentation.

---

## ✅ Running Tests

```bash
pip install -e . -r requirements/test.txt
pytest
```

---

## 🤝 Contributing

1. Create a feature branch (`git checkout -b feature/my-feature`)
2. Commit your changes
3. Push the branch and open a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

---

## ⚠ Known Issues

- Data comes from Yahoo Finance through `yfinance`, which is unofficial and can rate-limit or change without notice.
- Some balance-sheet metrics show as missing when Yahoo doesn't report them for a ticker.

---

## 📄 License

MIT License. See [LICENSE](LICENSE).

---

## ⚠ Disclaimer

BuffetBot is an educational tool. Nothing it shows is investment advice.
