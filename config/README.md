# Configuration Directory

This directory contains tool-specific configuration files that can be centrally managed.

## Files

- **`.editorconfig`** - Editor configuration for consistent coding style across IDEs
- **`.flake8`** - Python linting configuration for flake8

## Usage

These configuration files are automatically detected by their respective tools:

### EditorConfig
Automatically applied by compatible editors (VS Code, Sublime, Vim, etc.) to maintain consistent formatting.

### Flake8
Used by flake8 for Python linting configuration. Also integrated into the Makefile:
```bash
make lint  # Uses config/.flake8
```

## Backward Compatibility

Root-level symlinks are maintained for tool compatibility:
- `.editorconfig` → `config/.editorconfig`
- `.flake8` → `config/.flake8`

## Adding New Configuration

When adding new tool configurations:
1. Place the config file in this directory
2. Create a symlink in the root if needed for tool compatibility
3. Update this README with the new file description
