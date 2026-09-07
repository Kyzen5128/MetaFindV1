"""Relocate necessary historical docs, preserve raw archives, remove superseded text."""
from pathlib import Path
import hashlib
import json
import os
import re
import time
from urllib.parse import unquote

ROOT = Path('/home/kyzen/MetaFindV1')
DOCS = ROOT / 'docs'
HERE = Path(__file__).resolve().parent
active = {'README.md', 'DATA_FLOW.md', 'CUSTOM_TABLE1_EVALUATION.md', 'IDesign_INPUTS.md',
          'RAW_SCENE_INPUTS.md', 'SCENE_COMPOSITION.md', 'SCENE_PLACEMENT.md',
          'SCENE_SEMANTICS.md', 'SCENE_SCORES.md'}
deleted = {'NOTE_20260904_ULIP2_TRAINING_READ.md': 'Covered by NOTE_20260904_ULIP2_FULL_READ.md; both explicitly identify the latter as replacement.',
           'G3_MANUAL_EXCLUSION_PROPOSAL.md': 'Proposal approved by user; exact decision and formal accounting retained in workflow/DECISION_LEDGER.md DL-106.'}
history = DOCS / 'history'
assert not history.exists(), 'do not repeat relocation'
old_roots = sorted(DOCS.glob('*.md'))
mapping = {p: history/p.name for p in old_roots if p.name not in active | deleted.keys()}
future_report = DOCS/'REPRODUCTION_CORPUS_REVIEW_20260908.md'
mapping[future_report] = DOCS/'audit'/future_report.name
history.mkdir()
record = {'timestamp': time.time(), 'scope': 'User authorized docs cleanup; paper/protected content and frozen validation archives retained',
          'root_markdown_before': len(old_roots), 'active_guides': sorted(active), 'moves': [], 'deletions': [], 'link_updates': []}
hashof = lambda raw: hashlib.sha256(raw).hexdigest()

# Preserve original bytes in the Git history; untracked resolved proposal has a
# literal diagnostic copy so the actual question remains reviewable as well.
for name, reason in deleted.items():
    p = DOCS/name
    raw = p.read_bytes()
    if name == 'G3_MANUAL_EXCLUSION_PROPOSAL.md':
        with (HERE/'approved_g3_proposal_original.md').open('xb') as stream:
            stream.write(raw)
    record['deletions'].append({'path': str(p.relative_to(ROOT)), 'sha256': hashof(raw), 'bytes': len(raw), 'reason': reason})

files = list(DOCS.rglob('*.md')) + [ROOT/'README.md', ROOT/'tests/README.md', ROOT/'tools/probes/README.md']
link = re.compile(r'(!?\[[^\n]*?\]\()(<[^>\n]+>|[^)\n]+)(\))')
for old in files:
    if old.parent == DOCS and old.name in deleted:
        continue
    rel = old.relative_to(DOCS) if old.is_relative_to(DOCS) else None
    if rel is not None and (rel.parts[0] == 'paper' or any('artifacts' in part for part in rel.parts)):
        continue
    raw = old.read_bytes()
    content = raw.decode()
    new = mapping.get(old, old)
    frozen = old.name in ('RULES_SNAPSHOT.md', '_rules_preamble.md')
    updates = []

    def replace(match):
        target = match[2]
        wrapped = target.startswith('<') and target.endswith('>')
        value = target[1:-1] if wrapped else target
        if re.match(r'[a-zA-Z][a-zA-Z0-9+.-]*://', value) or value.startswith(('#', 'mailto:')):
            return match[0]
        # Preserve optional GitHub/local line suffix and heading fragment.
        parts = re.match(r'^(.*?)(:\d+)?(#[^\n]*)?$', value)
        path, line, fragment = parts[1], parts[2] or '', parts[3] or ''
        resolved = (old.parent/unquote(path)).resolve() if not path.startswith('/') else Path(path)
        destination = mapping.get(resolved, resolved)
        if destination == resolved and new.parent == old.parent:
            return match[0]
        updated = os.path.relpath(destination, new.parent) + line + fragment
        if wrapped or ' ' in updated:
            updated = '<' + updated + '>'
        updates.append({'old': target, 'new': updated})
        return match[1] + updated + match[3]

    if not frozen:
        content = link.sub(replace, content)
    new_raw = content.encode()
    if new != old:
        assert not new.exists()
        old.rename(new)
        record['moves'].append({'from': str(old.relative_to(ROOT)), 'to': str(new.relative_to(ROOT)),
            'original_sha256': hashof(raw), 'new_sha256': hashof(new_raw),
            'content_policy': 'exact bytes preserved' if frozen else 'only relative Markdown destinations updated'})
    if new_raw != raw:
        new.write_bytes(new_raw)
        record['link_updates'].append({'path': str(new.relative_to(ROOT)), 'changes': updates})

for name in deleted:
    (DOCS/name).unlink()
record['root_markdown_after'] = len(list(DOCS.glob('*.md')))
assert record['root_markdown_after'] == len(active)
(HERE/'docs_cleanup.json').write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
print(json.dumps({k:record[k] for k in ('root_markdown_before', 'root_markdown_after')}, indent=2))
print(f"moved {len(record['moves'])}; removed {len(record['deletions'])}; adjusted links in {len(record['link_updates'])} files")
