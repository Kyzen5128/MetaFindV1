# CAMERA reference material — NOT MetaFind code

Third-party files from the CAMERA project (Kyzen's other machine), supplied
2026-08-31 so we could read their retrieval protocol. **Nothing here runs as part
of MetaFind and nothing here is authoritative for MetaFind reproduction.**

- `evaluate.py` — CAMERA's six-way object-level retrieval evaluator. Dated
  2025-10-31, which is AFTER their paper was finalised (2025-10-27), so it is
  *not* the program that produced their Table 4. It imports `models.ULIP_models`
  and `utils.utils` from their ULIP tree and will not run here.
- `camera_working_note_2025-08-12.pdf` — their working note. Page 5 carries the
  Table 4 row `13.50 / 39.69 / 26.89` verbatim, with the counts it came from
  (`2020/14966` and `5940/14966`).

Moved out of the repository root on 2026-08-31: a file named `evaluate.py`
sitting beside our own code invites the reading that it is ours.

Our re-run of this protocol on our corpus is `tools/probes/camera_six_way.py`;
findings are in `workflow/DECISION_LEDGER.md` DL-056.
