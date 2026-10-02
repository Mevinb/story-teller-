import json
from types import SimpleNamespace
from pipeline import image_prompts as prompts


class Model:
    def __init__(self, data=None, error=None):
        self.data = data
        self.error = error
        self.calls = 0

    def generate(self, **kwargs):
        self.calls += 1
        assert 'non-explicit' in kwargs['system']
        if self.error:
            raise self.error
        return SimpleNamespace(as_json=lambda: self.data)


def sample():
    return {'highlights': [
        {'title': 'Arrival', 'context': 'Maya arrives.', 'characters': ['Maya'], 'prompt': 'Maya stands at a rain-soaked station.'},
        {'title': 'Meeting', 'context': 'Maya meets Arun.', 'characters': ['Arun', 'Maya'], 'prompt': 'Two characters meet under warm lamplight.'}
    ]}


def test_references_and_cache(tmp_path):
    model = Model(sample())
    value = prompts.generate(str(tmp_path), 'A story', model)
    assert value['status'] == 'ready'
    assert value['highlights'][1]['references'] == ['Reference 2: Arun', 'Reference 1: Maya']
    assert 'Preserve each corresponding character' in value['highlights'][0]['prompt']
    assert prompts.read(str(tmp_path), 'A story') == value
    prompts.generate(str(tmp_path), 'A story', model)
    assert model.calls == 1
    assert prompts.read(str(tmp_path), 'Edited story')['status'] == 'missing'


def test_failure_keeps_prior_prompts(tmp_path):
    original = prompts.generate(str(tmp_path), 'Story', Model(sample()))
    failed = prompts.generate(str(tmp_path), 'Story', Model(error=RuntimeError('Quota exhausted')), force=True)
    assert failed['status'] == 'failed'
    assert failed['highlights'] == original['highlights']
    assert failed['error'] == 'Quota exhausted'


def test_invalid_response_and_interruption(tmp_path):
    result = prompts.generate(str(tmp_path), 'Story', Model({'highlights': [{'title': 'Bad'}]}))
    assert result['status'] == 'failed'
    prompts._save(str(tmp_path), 'Interrupted', {'status': 'generating', 'highlights': []})
    assert prompts.read(str(tmp_path), 'Interrupted')['status'] == 'failed'


def test_long_story_uses_three_excerpts(tmp_path):
    class Capture(Model):
        def generate(self, **kwargs):
            assert len(kwargs['prompt']) < 37000
            assert 'BEGIN' in kwargs['prompt'] and 'MIDDLE' in kwargs['prompt'] and 'END' in kwargs['prompt']
            return super().generate(**kwargs)
    text = 'BEGIN' + 'a' * 25000 + 'MIDDLE' + 'b' * 25000 + 'END'
    assert prompts.generate(str(tmp_path), text, Capture(sample()))['status'] == 'ready'


def test_api_saved_results_and_source_validation(tmp_path, monkeypatch):
    import config
    from app import create_app
    monkeypatch.setattr(config, 'PROJECTS_DIR', str(tmp_path))
    directory = tmp_path / 'demo'
    (directory / 'chapters').mkdir(parents=True)
    (directory / 'chapters' / 'chapter_001.md').write_text('Story')
    prompts.generate(str(directory), 'Story', Model(sample()))
    client = create_app().test_client()
    result = client.get('/api/project/demo/image-prompts?source=chapter:1')
    assert result.status_code == 200
    assert result.json['status'] == 'ready'
    assert client.get('/api/project/demo/image-prompts?source=story:../../outside').status_code == 400
    assert client.get('/api/project/demo/image-prompts?source=chapter:2').status_code == 404
    (directory / 'chapters' / 'chapter_001.md').write_text('Edited story')
    assert client.get('/api/project/demo/image-prompts?source=chapter:1').json['status'] == 'missing'
    (directory / 'combined_polished_saved.md').write_text('Story')
    assert client.get('/api/project/demo/image-prompts?source=story:saved').json['status'] == 'ready'


def test_chapter_completion_generates_without_breaking_story(tmp_path):
    from pipeline.orchestrator import PipelineOrchestrator
    pipeline = PipelineOrchestrator.__new__(PipelineOrchestrator)
    pipeline.project_dir = str(tmp_path)
    pipeline._cloud_available = True
    pipeline.cloud_model = Model(sample())
    pipeline.read_chapter = lambda number: 'Story'
    events = []
    pipeline._progress_cb = lambda **event: events.append(event)
    result = {'chapter_number': 1}
    pipeline._emit('chapter_complete', result)
    assert result['image_prompts']['status'] == 'ready'
    assert events[-1]['event'] == 'chapter_complete'
    pipeline.cloud_model = Model(error=RuntimeError('Quota exhausted'))
    pipeline.read_chapter = lambda number: 'Another story'
    pipeline._emit('chapter_complete', {'chapter_number': 2})
    assert events[-1]['event'] == 'chapter_complete'
    assert events[-1]['data']['image_prompts']['status'] == 'failed'


def test_prune_preserves_other_saved_versions(tmp_path):
    (tmp_path / 'chapters').mkdir()
    (tmp_path / 'chapters' / 'chapter_001.md').write_text('Story')
    (tmp_path / 'combined_polished_archive.md').write_text('Archived')
    prompts.generate(str(tmp_path), 'Story', Model(sample()))
    prompts.generate(str(tmp_path), 'Archived', Model(sample()))
    (tmp_path / 'chapters' / 'chapter_001.md').unlink()
    prompts.prune(str(tmp_path))
    assert prompts.read(str(tmp_path), 'Story')['status'] == 'missing'
    assert prompts.read(str(tmp_path), 'Archived')['status'] == 'ready'
