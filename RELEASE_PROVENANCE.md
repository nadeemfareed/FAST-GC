# FAST-GC release provenance

FAST-GC is developed and maintained by Nadeem Fareed. The canonical public
project locations are the FAST-GC GitHub repository and the `fastgc` project on
PyPI.

## Release lineage

- `0.2.0` is the previously published PyPI/GitHub baseline.
- `0.2.1` is the backward-compatible development release that incorporates
  validated ground-classification, TLS membrane, canopy, progress, packaging,
  and reliability improvements while preserving the established public command
  syntax.

A release tag is intended to identify the exact source used to build the
corresponding PyPI artifacts. Development commits made after a release should
not be represented as that released artifact.

## Scientific provenance

Generated products should be associated with the FAST-GC software version used
for processing in methods, metadata, or reproducibility records. Citation
metadata are provided in `CITATION.cff`.

## Compatibility principle

Changes in a patch release must preserve established v0.2.0 command names and
workflow semantics unless a change is explicitly documented as a bug fix.
Scientific changes are tested separately from downstream products whose
behavior is intended to remain stable.
