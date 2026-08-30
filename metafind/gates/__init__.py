"""Promotion gates. A gate is a program that refuses, not a checklist.

Each module here implements one gate from ``docs/graph/validation_plan.yaml``
``level_3_gates``: it reads the artifact, writes a ``gate_records`` entry at the
gate's declared ``record_path``, and returns the rc its declared ``rc_contract``
assigns to the verdict. Nothing here mutates the artifact it judges.
"""
