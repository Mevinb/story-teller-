"""Isolated browser-test server: all stories and settings live in a temp folder."""
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ['STORY_AUTO_RESUME'] = 'false'
os.environ['EMBEDDING_LOCAL_FILES_ONLY'] = 'true'

import config
import app as server
from memory.state_manager import StateManager

def main():
    with tempfile.TemporaryDirectory(prefix='story-react-browser-') as scratch:
        config.PROJECTS_DIR = str(Path(scratch) / 'projects')
        server._ENV_PATH = str(Path(scratch) / '.env')
        directory = Path(config.PROJECTS_DIR) / 'browser_fixture'
        directory.mkdir(parents=True)
        state = StateManager(str(directory))
        state.initialize(title='The Lantern Archive', genre='Mystery', premise='A cartographer finds a map that changes at night.', characters={'Mira': {'description': 'A patient mapmaker', 'traits': ['observant']}})
        state.apply_state_update({'metadata': {'current_chapter': 2}})
        (directory / 'chapters').mkdir()
        for number in (1, 2):
            (directory / 'chapters' / f'chapter_{number:03d}.md').write_text(f'# Chapter {number}\n\nThe lantern burned beside the unfinished map.', encoding='utf-8')
        server.app.run(host='127.0.0.1', port=5017, debug=False, use_reloader=False, threaded=True)


if __name__ == "__main__":
    main()
