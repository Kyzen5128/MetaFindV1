"""CPU-only intent pipeline checks with real fusion/scoring and synthetic encoders."""
from dataclasses import replace
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from metafind.eval import intent_retrieval as run
from metafind.models.dual_tower import DualTowerConfig, MetaFindDualTower
from metafind.models.fusion import FusionConfig


def source(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


class Backbone:
    def __init__(self, offset=0.):
        self.offset = offset
        self.texts, self.clouds, self.images = [], [], []

    def tokenizer(self, texts):
        return torch.tensor([[len(t), ord(t[0])] for t in texts])

    def encode_text(self, texts):
        self.texts.extend(texts)
        return torch.tensor([[1., float(len(t)), .5] for t in texts])

    def encode_pc(self, clouds):
        self.clouds.append(clouds.clone())
        return clouds[:, 0, :3] + torch.tensor([3., 4., 5.]) + self.offset

    def preprocess(self, image):
        return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255

    def encode_image(self, images):
        self.images.append(images.clone())
        return images.mean(dim=(-2, -1)) + .1


@pytest.fixture
def inputs(tmp_path):
    records = {}
    for i, uid in enumerate(['b', 'a', 'c', 'd', 'e', 'f']):
        cloud = np.tile(np.array([1., 2., 3., .2, .3, .4], np.float32), (10000, 1))
        cp = tmp_path / f'{uid}.npz'
        np.savez(cp, xyz=cloud[:, :3], rgb=cloud[:, 3:])
        ep = tmp_path / f'{uid}_embedding.npz'
        np.savez(ep, views=np.array([[i+1., 2, 3], [2., i+4, 1]], np.float32))
        records[uid] = {'canonical_text': 'catalogue '+uid, 'cloud': source(cp), 'embedding': source(ep)}
    image = tmp_path / 'external.png'
    Image.new('RGB', (3, 3), (10, 90, 230)).save(image)
    points = np.zeros((10000, 6), np.float32)
    points[:, 0] = np.linspace(2, 6, 10000)
    points[:, 3:] = .5
    pc = tmp_path / 'reference.npy'
    np.save(pc, points)
    gallery = list(records)
    q1 = {'query_id': 'need-wood-chair', 'text': 'independent request', 'images': [source(image)],
          'pointcloud': source(pc), 'conditions': list(run.QUERY_CONDITIONS),
          'qrels': {c: {u: int(u in ('a', 'b')) for u in gallery} for c in run.QUERY_CONDITIONS}}
    q2 = {'query_id': 'need-stool', 'text': 'another independent request', 'conditions': ['text'],
          'qrels': {'text': {u: int(u == 'c') for u in gallery}}}
    return {'gallery_uids': gallery, 'query_ids': [q1['query_id'], q2['query_id']], 'records': records, 'queries': [q1, q2]}


def model():
    cfg = FusionConfig(dim=3, kind='mean', prefusion_norm=False, include_absent_slots=True)
    m = MetaFindDualTower(DualTowerConfig(dim=3, tower_sharing='shared_backbone_separate_fusion',
        query_fusion=cfg, gallery_fusion=replace(cfg), use_layout=False)).eval()
    return m


def test_inputs_are_independent_requests_and_reference_cloud_uses_query_path(inputs):
    bb, qbb = Backbone(), Backbone(offset=10)
    enc, tokens = run.encode_queries(inputs, bb, query_backbone=qbb)
    assert bb.texts == ['independent request', 'another independent request']
    assert not bb.clouds and len(qbb.clouds) == 1
    assert qbb.clouds[0].shape == (1, 10000, 6)
    assert abs(float(qbb.clouds[0][..., :3].mean())) < 1e-6
    assert float(qbb.clouds[0][..., :3].norm(dim=-1).max()) == pytest.approx(1.)
    assert torch.all(qbb.clouds[0][..., 3:] == .5)
    assert len(bb.images) == 1
    assert set(enc['need-stool']) == {'text'}
    assert tokens['need-wood-chair']['text'] == inputs['queries'][0]['text']
    altered = copy.deepcopy(inputs)
    for query in altered['queries']:
        query['qrels'] = {}  # relevance is never used to manufacture model input
    same, _ = run.encode_queries(altered, Backbone(), query_backbone=Backbone(offset=10))
    for qid, parts in enc.items():
        for modality in parts:
            np.testing.assert_array_equal(parts[modality], same[qid][modality])


def test_gallery_encoding_does_not_consume_any_query_or_qrels(inputs):
    bb = Backbone()
    g = run.encode_gallery(inputs, bb, batch_size=2)
    assert bb.texts == ['catalogue '+u for u in inputs['gallery_uids']]
    assert not bb.images  # pre-bound per-view features
    assert sum(len(x) for x in bb.clouds) == 6
    assert all(x.shape == (6, 3) for x in g.values())
    without_queries = {**inputs, 'queries': []}
    gg = run.encode_gallery(without_queries, Backbone(), batch_size=3)
    for k in g:
        np.testing.assert_array_equal(g[k], gg[k])


def test_seven_conditions_hide_absent_inputs_and_use_condition_specific_answers(inputs, tmp_path):
    m = model()
    seen = []
    def inspect(module, args, kwargs):
        assert kwargs['layout'] is None
        flags = kwargs['present'][0].tolist()
        assert all((args[0][name] is not None) == yes for name, yes in zip(run.MODALITIES, flags))
        seen.append(tuple(flags))
    hook = m.query.register_forward_pre_hook(inspect, with_kwargs=True)
    encoded, _ = run.encode_queries(inputs, Backbone())
    gallery = run.fuse_gallery(run.encode_gallery(inputs, Backbone()), model=m, batch_size=2)
    vectors = run.fuse_queries(inputs, encoded, model=m)
    hook.remove()
    assert len(seen) == 8 and len(vectors) == 7
    assert vectors['text']['query_ids'] == ['need-wood-chair', 'need-stool']
    assert vectors['full']['query_ids'] == ['need-wood-chair']
    # Labels for one condition differ; other conditions must not borrow them.
    inputs['queries'][0]['qrels']['image'] = {u: 1 for u in inputs['gallery_uids']}
    result = run.evaluate_method(inputs, gallery, vectors, tmp_path, 'tiny', block=2)
    assert result['image']['metrics']['Hit@1'] == 100
    assert result['image']['metrics']['Recall@1'] == pytest.approx(100/6)
    rows = [json.loads(x) for x in (tmp_path/'tiny__image.jsonl').read_text().splitlines()]
    assert rows[0]['n_relevant'] == 6
    assert result['text']['n_query'] == 2 and result['full']['n_query'] == 1
    with pytest.raises(FileExistsError):
        run.evaluate_method(inputs, gallery, vectors, tmp_path, 'tiny')


def test_changed_reference_bytes_are_refused(inputs):
    Path(inputs['queries'][0]['images'][0]['path']).write_bytes(b'changed image')
    with pytest.raises(ValueError):
        run.encode_queries(inputs, Backbone())


def test_no_fallback_for_missing_reference_modality(inputs):
    encoded, _ = run.encode_queries(inputs, Backbone())
    inputs['queries'][1]['conditions'] = ['full']
    with pytest.raises(ValueError, match='has no image'):
        run.fuse_queries(inputs, encoded)


def test_report_uses_hit_and_recall_not_table1_r_and_discloses_denominators():
    text = run.markdown_report({'methods': {'tiny': {'text': {'n_query': 3, 'metrics': {
        'Hit@1': 50., 'Hit@5': 100., 'Recall@1': 25., 'Recall@5': 75.}}}}})
    assert 'Hit@1' in text and 'Recall@5' in text
    assert '| tiny | text | 3 | 50.00 | 100.00 | 25.00 | 75.00 |' in text
    assert 'different explicit query sets' in text
    assert 'R@1' not in text


@pytest.fixture
def cli(inputs, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from metafind.eval import intent_protocol, custom_models
    from metafind.train import gallery_index, stage1
    from metafind import runlog
    protocol = tmp_path / 'frozen.json'
    protocol.write_text(json.dumps(inputs))
    encoding = {'actual_clip_train_scope': 'frozen'}
    training = {'train_scope': 'point_encoder', 'tower_sharing': 'shared_backbone_separate_fusion'}
    record = {'sha256': 'parent-hash', 'uri': 'diagnostic.pt'}
    loaded = SimpleNamespace(backbone=Backbone(), query_backbone=None, model=model(), record=record)
    monkeypatch.setattr(intent_protocol, 'load_protocol', lambda p: inputs)
    monkeypatch.setattr(gallery_index, 'load_checkpoint_record', lambda p: record)
    monkeypatch.setattr(gallery_index, 'load_stage2_checkpoint_record', lambda *a, **kw: {'sha256': 'child-hash'})
    monkeypatch.setattr(stage1, 'load_stage1_model_config', lambda *a: record)
    monkeypatch.setattr(stage1, 'load_protocols', lambda: ({}, {}, {}))
    monkeypatch.setattr(stage1, 'effective_stage1_model_inputs', lambda *a: (encoding, training, {}))
    monkeypatch.setattr(stage1, 'fusion_config_for', lambda *a, **kw: SimpleNamespace(image_tokens=1))
    monkeypatch.setattr(stage1, 'stage1_backbone_kwargs', lambda r: {})
    monkeypatch.setattr(stage1, '_open_clip_weight_identity', lambda: {'scope': 'synthetic encoder'})
    monkeypatch.setattr(run, 'validate_cache_compatibility', lambda *a: None)
    monkeypatch.setattr(custom_models, 'verify_mean_initializer', lambda: None)
    monkeypatch.setattr(custom_models, 'load_mean', lambda device: Backbone())
    monkeypatch.setattr(custom_models, 'load_stage1', lambda *a: loaded)
    monkeypatch.setattr(custom_models, 'apply_stage2', lambda *a: {'sha256': 'child-hash'})
    monkeypatch.setattr(runlog, 'runtime_source_sha256', lambda: 'synthetic-source-hash')
    monkeypatch.setattr(runlog, 'code_revision', lambda: 'fixture-revision')
    monkeypatch.setattr(runlog, 'code_dirty', lambda: True)
    out = tmp_path / 'run'
    argv = ['--protocol', str(protocol), '--stage1-record', 'parent.json', '--stage2-record', 'child.json', '--out-dir', str(out)]
    return argv, out, loaded


def test_cli_runs_all_three_methods_and_writes_only_completed_results(cli):
    argv, output, _ = cli
    assert run.main(argv) == 0
    result = json.loads((output/'results.json').read_text())
    assert set(result['methods']) == {'ulip2_available_mean', 'stage1', 'stage2_layout_off'}
    assert result['stage2_parent_gallery'] == 'unchanged'
    assert result['methods']['stage1']['text']['query_ids'] == ['need-wood-chair', 'need-stool']
    assert len(list(output.glob('*.jsonl'))) == 21
    assert sum(len(p.read_text().splitlines()) for p in output.glob('*.jsonl')) == 24
    assert (output/'stage1_query_tokens.json').is_file()
    assert not (output/'failure.json').exists()


def test_cli_check_only_does_not_load_models_or_create_outputs(cli, monkeypatch):
    from metafind.eval import custom_models
    argv, output, _ = cli
    def forbidden(*a):
        raise AssertionError('model loaded during check-only')
    monkeypatch.setattr(custom_models, 'load_mean', forbidden)
    assert run.main(argv + ['--check-only']) == 0
    assert not output.exists()


def test_cli_rejects_stage2_mutating_gallery_without_final_score_artifact(cli, monkeypatch):
    from metafind.eval import custom_models
    argv, output, loaded = cli
    # Changing gallery forward is enough to challenge the actual comparison path.
    def bad_overlay(*a):
        original = loaded.model.gallery.forward
        loaded.model.gallery.forward = lambda x: original(x).roll(1, dims=1)
        return {'sha256': 'child-hash'}
    monkeypatch.setattr(custom_models, 'apply_stage2', bad_overlay)
    with pytest.raises(ValueError, match='changed the parent gallery'):
        run.main(argv)
    assert (output/'failure.json').is_file()
    assert not (output/'results.json').exists()
    assert not (output/'table.md').exists()
