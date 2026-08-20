# MetaFindV1 Codex Instructions

MetaFindV1 is a scientific reproduction project. Research correctness, paper fidelity, evidence traceability, and reproducibility take priority over speed or convenient implementation.

## Authority and evidence

- `docs/paper/*_source/**` is the paper-content authority for its respective paper. MetaFind claims must start from `docs/paper/metafind_source/**`.
- Upstream papers and implementations apply only where MetaFind explicitly inherits them. They must not fill gaps in the MetaFind paper.
- `docs/audit/**` and `docs/graph/**` are derived audits, contracts, decisions, and specifications. They do not override paper source.
- README files, implementation, tests, runtime behavior, comments, handoffs, and prior Claude or Codex statements are not paper authority.
- Classify material claims as `PAPER FACT`, `UPSTREAM FACT`, `OBSERVED IMPLEMENTATION`, `OBSERVED DATA`, `INFERENCE`, `IMPLEMENTATION CHOICE`, `DEVIATION`, or `UNKNOWN`.
- Preserve the chain `paper → specification → implementation → validation`. Evidence from a later stage must not silently redefine an earlier stage.
- Follow `MANY TESTS, FEW GATES`: a test proves only its scoped assertion; only explicitly designated gates may block promotion. Required Audits must run and be recorded but do not become gates.

## Skill routing

- For paper, formula, equation, section, appendix, table, or figure audits, use the `paper-audit` skill.
- For reproduction status, code-vs-paper comparison, deviations, missing implementation, runtime tracing, or validation, use the `reproduction-audit` skill.

## Protected paths

- `docs/paper/*_source/**` is protected paper content, except for `SOURCE_MANIFEST.json`. Each `SOURCE_MANIFEST.json` is regenerable provenance and integrity metadata and is not paper content itself; its exception does not authorize changes during unrelated work.
- `docs/paper/*.gz` remains protected.
- Never directly modify `metafind/vendor/**`. Compatibility fixes must be placed in `metafind/compat/`.
- Do not modify `CLAUDE.md` or `.claude/**` during normal research work.
- Claude hooks are Claude-only enforcement. Codex is not protected by `.claude/hooks/research_authority_guard.py`; treat these rules as explicit stop conditions and never claim the hook protected a Codex action.

If a material paper ambiguity, contradiction, unsupported assumption, or deviation requires a research decision, stop and request an explicit user decision. Never repair it silently to make code or tests pass.
