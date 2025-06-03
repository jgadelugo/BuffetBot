# Changelog

All notable changes to BuffetBot will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive financial metrics glossary module (`glossary_data.py`)
- Multiple UI implementations for the glossary:
  - Streamlit interactive web app (`glossary_app.py`)
  - Standalone HTML/JavaScript version (`glossary_web.html`)
  - React component (`GlossaryComponent.jsx`)
- Test suite with pytest configuration
- Examples directory with integration demonstrations
- Development tooling:
  - Pre-commit hooks configuration
  - Makefile for common tasks
  - EditorConfig for consistent code style
  - Modern Python packaging with `pyproject.toml`

### Changed
- Reorganized project structure following Python best practices
- Moved tests to dedicated `tests/` directory
- Moved examples to dedicated `examples/` directory

### Fixed
- Import paths in tests and examples for new directory structure

## [1.0.0] - 2024-01-XX

### Added
- Initial release of BuffetBot financial analysis toolkit
- Core analysis modules:
  - Growth analysis
  - Value analysis
  - Health analysis
  - Risk analysis
- Financial metrics calculations
- Data validation and error handling
- Logging system
- Caching functionality

### Security
- Input validation for all financial data
- Secure handling of API credentials

## Notes

### Types of changes
- `Added` for new features
- `Changed` for changes in existing functionality
- `Deprecated` for soon-to-be removed features
- `Removed` for now removed features
- `Fixed` for any bug fixes
- `Security` in case of vulnerabilities

[Unreleased]: https://github.com/your-username/buffetbot/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/your-username/buffetbot/releases/tag/v1.0.0
