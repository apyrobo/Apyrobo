# Architecture Decision Records

Durable, in-repo records of accepted design decisions — primarily the
outcomes of [spec RFCs](../rfc_process.md). Issues and their comment threads
are where decisions get *made*; ADRs are where they get *remembered*.

Each ADR is immutable once accepted: if a decision is later reversed, a new
ADR supersedes the old one and both link to each other. Statuses:
**Accepted**, **Superseded by ADR-NNNN**.

## Writing one

Copy [template.md](template.md) to `NNNN-short-slug.md` (next free number,
zero-padded to four digits) and add a row to the index below in the same PR
as the change it records.

## Index

| ADR | Title | Status |
|-----|-------|--------|
| [0001](0001-adopt-rfc-process.md) | Adopt a public RFC process for spec changes | Accepted |
