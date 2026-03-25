# Claude Documentation Directory

This directory contains detailed documentation for AI agents working with the AMI Data Companion codebase.

## Quick Start

1. **Start here:** Read `../CLAUDE.md` in the root directory for cost optimization tips and high-level overview
2. **Understand the architecture:** Read `architecture.md` for system design
3. **Learn the database:** Read `database.md` for schema and query patterns
4. **Understand ML pipeline:** Read `ml-pipeline.md` for ML processing flow
5. **For API work:** Read `api.md` for REST API details
6. **For testing:** Read `testing.md` for testing strategies
7. **For specific tasks:** Read `common-tasks.md` for how-to guides

## Documentation Files

### [architecture.md](architecture.md)
**Detailed system architecture and design patterns**

Topics covered:
- High-level architecture overview
- Core design patterns (Pipeline, Repository, Registry, Settings)
- Module dependencies
- Data architecture and flow
- Concurrency model
- Extension points
- Performance optimization
- Testing architecture
- Security considerations

**Read this when:**
- Understanding system design
- Adding new features
- Refactoring code
- Optimizing performance

### [database.md](database.md)
**Complete database schema and query documentation**

Topics covered:
- Entity-relationship diagrams
- Table details (MonitoringSession, TrapImage, DetectedObject)
- Occurrence concept
- Queue system
- Migration management
- Query optimization
- Common patterns
- Troubleshooting

**Read this when:**
- Working with database models
- Writing queries
- Creating migrations
- Debugging data issues
- Optimizing database performance

### [ml-pipeline.md](ml-pipeline.md)
**ML models and processing pipeline documentation**

Topics covered:
- Pipeline stages (Localization, Classification, Tracking)
- Model architectures
- Available classifiers
- Feature extraction
- Tracking algorithm
- Model registry
- Performance optimization
- Testing ML components

**Read this when:**
- Adding new models
- Understanding inference flow
- Debugging model issues
- Optimizing ML performance
- Training new models

### [api.md](api.md)
**REST API documentation**

Topics covered:
- API endpoints (/process, /info, /readyz, /livez)
- Request/response schemas
- Available pipelines
- Usage examples (Python, JavaScript, cURL)
- Configuration
- Error handling
- Security
- Deployment

**Read this when:**
- Integrating with API
- Adding new endpoints
- Debugging API issues
- Deploying API service

### [testing.md](testing.md)
**Testing strategies and examples**

Topics covered:
- Testing strategy (E2E, integration, unit)
- Test organization
- Running tests
- Test fixtures
- Mocking
- CI/CD testing
- Coverage
- Performance testing
- Best practices

**Read this when:**
- Writing tests
- Debugging test failures
- Setting up CI/CD
- Improving coverage

### [common-tasks.md](common-tasks.md)
**How-to guides for common tasks**

Topics covered:
- Development tasks (adding models, migrations, commands, endpoints)
- Operational tasks (import, process, export, backup)
- Debugging tasks (finding issues, inspecting quality)
- Maintenance tasks (cleanup, rebuilding)
- Integration tasks (external databases, cloud storage)

**Read this when:**
- Performing specific tasks
- Learning how to do something
- Following step-by-step guides

## Document Hierarchy

```
CLAUDE.md (root)                    ← Start here
    ↓
docs/claude/README.md (this file)   ← Navigation guide
    ↓
┌─────────────────────────────────────────────────────┐
│                                                     │
│  architecture.md    database.md    ml-pipeline.md  │  ← Deep dives
│       api.md         testing.md                    │
│                                                     │
└─────────────────────────────────────────────────────┘
    ↓
common-tasks.md                     ← Practical guides
```

## How to Use This Documentation

### For New AI Agents

**First Session:**
1. Read `CLAUDE.md` for cost optimization rules
2. Skim `architecture.md` for system overview
3. Read relevant deep-dive docs based on task
4. Reference `common-tasks.md` for implementation

**Subsequent Sessions:**
1. Review `CLAUDE.md` gotchas section
2. Jump to relevant documentation
3. Update documentation with learnings

### For Specific Tasks

**Adding a feature:**
1. `architecture.md` - Understand where it fits
2. `database.md` - Check if schema changes needed
3. `ml-pipeline.md` or `api.md` - Understand relevant subsystem
4. `common-tasks.md` - Follow implementation guide
5. `testing.md` - Write tests

**Debugging an issue:**
1. `common-tasks.md` - Check debugging section
2. Relevant deep-dive doc for subsystem
3. `testing.md` - Write regression test

**Understanding existing code:**
1. `architecture.md` - Get big picture
2. Relevant subsystem doc
3. Read code with context

## Documentation Conventions

### Code Examples

All code examples are real and tested where possible:

```python
# This is actual code from the codebase
from trapdata.db.models.detections import DetectedObject

detection = DetectedObject(
    bbox=[10, 20, 100, 200],
    specific_label="Actias luna"
)
```

### File Paths

File paths are absolute from repo root:

```
trapdata/ml/pipeline.py
trapdata/db/models/detections.py
```

### Commands

Commands are ready to copy-paste:

```bash
ami test all
pytest trapdata/tests/test_pipeline.py -v
```

### Diagrams

ASCII diagrams for quick understanding:

```
Input → Process → Output
  ↓       ↓         ↓
Store   Queue    Export
```

## Maintaining This Documentation

### When to Update

**Always update when:**
- Adding new features
- Changing architecture
- Fixing bugs (add to gotchas)
- Learning something non-obvious

**Update locations:**
- `CLAUDE.md` - Gotchas section
- Relevant deep-dive doc
- `common-tasks.md` if applicable

### How to Update

1. **Keep it concise** - AI agents pay per token
2. **Use examples** - Show, don't tell
3. **Update related sections** - Keep docs consistent
4. **Test examples** - Ensure code works
5. **Link between docs** - Help navigation

### Example Update

```markdown
## Gotchas - New Addition

### Database
- **Previous frame references:** ...
- **Bulk operations:** ...
+ **Transaction deadlocks:** Keep transactions short and commit frequently.
+   Long-running transactions can cause deadlocks. Always use try/except
+   and rollback on error.
```

## Contributing

When you learn something new:

1. Add to `CLAUDE.md` gotchas section
2. Add detailed explanation to relevant doc
3. Add example to `common-tasks.md` if applicable
4. Commit with message: "docs: add learning about X"

**Template for new learnings:**

```markdown
### Issue Encountered
Brief description of the problem

### Root Cause
Why it happened

### Solution
How to fix it

### Prevention
How to avoid it in future

### Example
```python
# Code example
```
```

## Additional Resources

- **Main README:** `../README.md` - User documentation
- **API Docs:** Auto-generated from FastAPI (http://localhost:2000/docs)
- **Database Schema:** Alembic migrations in `trapdata/db/migrations/`
- **Code Examples:** Test files in `trapdata/tests/`
- **Model Registry:** `trapdata/ml/models/`

## Questions?

If you can't find what you need:

1. Search the docs (grep or Ctrl+F)
2. Check the code (especially test files)
3. Check git history (`git log`, `git blame`)
4. Add the answer to docs when you find it!

---

**Last Updated:** 2025-11-19
**Maintained by:** AI Agents working with AMI Data Companion
