"""React asset serving and read-only activity discovery contracts."""
from pathlib import Path
from flask import Flask
from web_ui import register_ui


def test_compiled_studio_and_classic_are_available(tmp_path):
    app = Flask(__name__, static_folder=str(tmp_path), template_folder=str(Path(__file__).parent / 'templates'))
    build = tmp_path / 'app'
    build.mkdir()
    (build / 'index.html').write_text('<div id="root"></div><script type="module" src="/static/app/assets/test.js"></script>')
    register_ui(app)
    client = app.test_client()
    response = client.get('/')
    assert response.status_code == 200
    assert b'id="root"' in response.data
    assert response.headers['Cache-Control'] == 'no-cache'
    assert b'id="viewDashboard"' in client.get('/classic').data


def test_python_only_checkout_falls_back_to_classic(tmp_path):
    app = Flask(__name__, static_folder=str(tmp_path), template_folder=str(Path(__file__).parent / 'templates'))
    register_ui(app)
    response = app.test_client().get('/')
    assert response.status_code == 200
    assert b'id="viewDashboard"' in response.data


def test_activity_reports_running_work_without_loading_models(monkeypatch):
    import app as server
    monkeypatch.setattr(server, '_active_pipelines', {'demo': object()})
    monkeypatch.setattr(server, '_active_combines', {'another': True})
    monkeypatch.setattr(server, '_manual_sessions', {'demo': object()})
    client = server.create_app().test_client()
    assert client.get('/api/project/demo/activity').json == {'generation': True, 'combine': False, 'manual': True}
    assert client.get('/api/project/another/activity').json == {'generation': False, 'combine': True, 'manual': False}
    assert client.get('/api/project/idle/activity').json == {'generation': False, 'combine': False, 'manual': False}
