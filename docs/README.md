# BuffetBot Documentation

This directory contains comprehensive documentation for the BuffetBot project, organized by type and purpose.

## 📁 Directory Structure

### `/architecture/`
Major architectural decisions, system design changes, and structural improvements.

- [`DASHBOARD_RESTRUCTURE_SUMMARY.md`](architecture/DASHBOARD_RESTRUCTURE_SUMMARY.md) - Complete dashboard reorganization: Options Advisor & Analyst Forecast separation

### `/fixes/`
Bug fixes, error resolutions, and corrective implementations.

- [`FORECAST_INTEGRATION_FIX.md`](fixes/FORECAST_INTEGRATION_FIX.md) - Restoring forecast data integration in options scoring algorithm

### `/features/`
New feature implementations, enhancements, and capability additions.

*Future feature documentation will be added here*

### `/glossary/`
Metric definitions, terminology, and reference materials.

## 📋 Existing Root-Level Documentation

### Core Project Files
- [`README.md`](../README.md) - Main project overview and setup
- [`CONTRIBUTING.md`](../CONTRIBUTING.md) - Contribution guidelines
- [`CHANGELOG.md`](../CHANGELOG.md) - Version history and changes
- [`LICENSE`](../LICENSE) - Project license

### Deployment & Configuration
- [`DEPLOYMENT.md`](../DEPLOYMENT.md) - Deployment instructions
- [`RUNNING_THE_APP.md`](../RUNNING_THE_APP.md) - Application execution guide
- [`CONFIGURATION_FILES.md`](../CONFIGURATION_FILES.md) - Configuration documentation

### Technical Documentation
- [`PROJECT_STRUCTURE.md`](../PROJECT_STRUCTURE.md) - Project organization
- [`TEST_SUMMARY.md`](../TEST_SUMMARY.md) - Testing overview
- [`ARCHITECTURE_CONSOLIDATION.md`](../ARCHITECTURE_CONSOLIDATION.md) - Architecture decisions

### Legacy & Historical
- [`REFACTORING_SUMMARY.md`](../REFACTORING_SUMMARY.md) - Past refactoring work
- [`TICKER_SYNCHRONIZATION_FIX.md`](../TICKER_SYNCHRONIZATION_FIX.md) - Ticker sync improvements
- [`TICKER_CHANGE_FIX.md`](../TICKER_CHANGE_FIX.md) - Ticker change handling
- [`GOOGLE_ANALYTICS_INTEGRATION.md`](../GOOGLE_ANALYTICS_INTEGRATION.md) - Analytics setup
- [`FINAL_VERIFICATION.md`](../FINAL_VERIFICATION.md) - Final verification steps
- [`FILE_CLEANUP_PLAN.md`](../FILE_CLEANUP_PLAN.md) - File organization plan

## 📝 Documentation Guidelines

### When to Add Documentation

1. **Architecture Changes** → `/architecture/`
   - Major structural changes
   - Design decisions
   - System reorganization

2. **Bug Fixes** → `/fixes/`
   - Error resolutions
   - Data integration fixes
   - Performance improvements

3. **New Features** → `/features/`
   - Feature implementations
   - Enhancement descriptions
   - Capability additions

### Naming Conventions

- Use `UPPER_CASE_WITH_UNDERSCORES.md` for consistency
- Include date prefix for time-sensitive docs: `2024_12_03_FEATURE_NAME.md`
- Use descriptive names that clearly indicate content

### Content Structure

Each documentation file should include:
1. **Clear title and purpose**
2. **Problem/context description**
3. **Solution implementation**
4. **Technical details**
5. **Benefits and outcomes**
6. **Future considerations**

## 🔄 Maintenance

This documentation structure should be:
- **Committed to version control** (important decisions and fixes)
- **Updated with major changes**
- **Referenced in pull requests**
- **Reviewed during code reviews**

---

*This documentation index was created to improve organization and discoverability of project documentation.*
