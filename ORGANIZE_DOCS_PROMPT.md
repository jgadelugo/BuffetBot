# Documentation Organization Prompt

Act as a staff engineer and architect. Please help me organize all markdown (.md) files in this BuffetBot project according to established best practices. The project currently has many markdown files scattered throughout the root directory and some in subdirectories that need proper organization.

## Current Documentation Structure

I've established a documentation structure in the `docs/` directory with the following subdirectories:

```
docs/
├── README.md                 # Documentation index (already exists)
├── architecture/             # Major design decisions & structural changes
├── fixes/                   # Bug fixes, error resolutions, corrective implementations
├── features/                # New feature implementations & enhancements
└── glossary/                # Metric definitions, terminology, reference materials
```

## Categorization Guidelines

Please categorize and move ALL markdown files according to these criteria:

### `/docs/architecture/`
- Major structural changes and system redesigns
- Architectural decisions and design patterns
- System reorganization and refactoring
- Files containing: "ARCHITECTURE", "RESTRUCTURE", "CONSOLIDATION", "REFACTORING"

### `/docs/fixes/`
- Bug fixes and error resolutions
- Data integration fixes and corrections
- Performance improvements and optimizations
- Synchronization fixes and handling improvements
- Files containing: "FIX", "BUG", "ERROR", "SYNC", "INTEGRATION"

### `/docs/features/`
- New feature implementations
- Feature enhancements and capabilities
- Analytics integrations and new functionality
- Files containing: "FEATURE", "ANALYTICS", "INTEGRATION" (when about new features)

### `/docs/glossary/`
- Metric definitions and terminology
- Reference materials and documentation
- Any existing glossary content

## Files to Keep in Root Directory

These core project files should remain in the root:
- `README.md` (main project overview)
- `CONTRIBUTING.md`
- `CHANGELOG.md`
- `LICENSE` or `LICENSE.md`

## Files to Keep in Current Locations

These deployment/config files should remain in root:
- `DEPLOYMENT.md`
- `RUNNING_THE_APP.md`
- `CONFIGURATION_FILES.md`
- `PROJECT_STRUCTURE.md`

## Your Tasks

1. **Scan the entire project** for all `.md` files (root directory and any subdirectories)

2. **Categorize each file** based on its content and filename using the guidelines above

3. **Move files to appropriate directories**:
   - Create subdirectories if they don't exist
   - Move files using proper git commands to preserve history
   - Update any internal links that might break

4. **Update the docs/README.md index** to reflect the new organization

5. **Handle edge cases**:
   - If unsure about categorization, explain your reasoning
   - Some files might fit multiple categories - use your best judgment
   - If a file doesn't fit any category, suggest where it should go

## Expected Files to Organize

Based on the project structure, you'll likely find files like:
- Various FIX files (TICKER_SYNCHRONIZATION_FIX.md, TICKER_CHANGE_FIX.md, etc.)
- Architecture files (ARCHITECTURE_CONSOLIDATION.md, REFACTORING_SUMMARY.md, etc.)
- Feature files (GOOGLE_ANALYTICS_INTEGRATION.md, etc.)
- Summary and plan files (TEST_SUMMARY.md, FILE_CLEANUP_PLAN.md, etc.)
- Verification files (FINAL_VERIFICATION.md, etc.)

## Commit Strategy

After organizing:
1. Commit all moves with a clear message
2. Ensure the docs/README.md accurately reflects the new structure
3. Test that all internal links still work

## Success Criteria

When complete:
- All documentation follows the established structure
- docs/README.md accurately indexes all files
- File categorization is logical and consistent
- Git history is preserved for moved files
- Internal links are updated where necessary

Please proceed with organizing all markdown files according to these guidelines. If you encounter any files where the categorization is unclear, explain your reasoning and ask for confirmation before proceeding.
