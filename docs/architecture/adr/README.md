# Architectural Decision Records (ADRs)

## Overview

This directory contains Architectural Decision Records (ADRs) for the Image Processing Application. ADRs document important architectural decisions, their context, rationale, and consequences.

## ADR Format

Each ADR follows the standard format:

- **Title**: Short descriptive title
- **Status**: Proposed, Accepted, Deprecated, Superseded
- **Context**: Description of the issue and relevant constraints
- **Decision**: The architectural decision made
- **Consequences**: Positive and negative outcomes of the decision

## ADR Index

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| [ADR-001](001-use-pyqt6-for-gui.md) | Use PyQt6 for GUI Framework | Accepted | 2024-01-15 |
| [ADR-002](002-modular-processing-architecture.md) | Modular Processing Architecture | Accepted | 2024-01-20 |
| [ADR-003](003-sqlite-for-metrics-storage.md) | SQLite for Metrics Storage | Accepted | 2024-01-22 |
| [ADR-004](004-yaml-configuration-format.md) | YAML Configuration Format | Accepted | 2024-01-25 |
| [ADR-005](005-plugin-based-filter-system.md) | Plugin-based Filter System | Accepted | 2024-01-30 |
| [ADR-006](006-correlation-id-logging.md) | Correlation ID Logging Strategy | Accepted | 2024-02-05 |
| [ADR-007](007-health-check-endpoints.md) | Health Check Endpoints Design | Accepted | 2024-02-10 |
| [ADR-008](008-async-processing-model.md) | Asynchronous Processing Model | Accepted | 2024-02-15 |

## Creating New ADRs

To create a new ADR:

1. **Identify the Decision**: Determine what architectural decision needs documentation
2. **Assign Number**: Use the next sequential ADR number
3. **Use Template**: Copy the ADR template below
4. **Fill Content**: Complete all sections with relevant information
5. **Review**: Have the ADR reviewed by the architecture team
6. **Update Index**: Add the new ADR to the index above

## ADR Template

```markdown
# ADR-XXX: [Title]

## Status

[Proposed | Accepted | Deprecated | Superseded by ADR-XXX]

## Context

[Describe the architectural design issue, including relevant constraints and requirements]

## Decision

[Describe the architectural decision and overall strategy]

## Consequences

### Positive
- [List positive consequences]

### Negative
- [List negative consequences]

### Neutral
- [List neutral consequences]

## Implementation Notes

[Any specific implementation guidance]

## Related ADRs

- [List related ADRs if any]

## References

- [External references and documentation]
```

## Guidelines

1. **Keep ADRs Concise**: Focus on the decision and rationale
2. **Include Context**: Explain why the decision was needed
3. **Document Alternatives**: Briefly mention alternatives considered
4. **Update Status**: Keep status current as decisions evolve
5. **Link Dependencies**: Reference related ADRs and external docs