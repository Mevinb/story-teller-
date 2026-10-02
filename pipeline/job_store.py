"""Leased automatic chapter jobs; restart recovery never creates duplicate owners."""
import json
import time
import uuid
from models.quota_scheduler import QuotaScheduler


class JobStore:
    def __init__(self, storage=None, clock=time.time):
        self.storage = storage or QuotaScheduler()
        self.clock = clock
        with self.storage.db() as db:
            db.execute('''CREATE TABLE IF NOT EXISTS jobs (
                project TEXT PRIMARY KEY, request TEXT, base_chapter INTEGER,
                target_chapter INTEGER, status TEXT, owner TEXT, heartbeat REAL,
                retry_at REAL, attempts INTEGER, error TEXT)''')

    def claim(self, project, request, current_chapter):
        now = self.clock()
        owner = uuid.uuid4().hex
        with self.storage.db() as db:
            row = db.execute('SELECT * FROM jobs WHERE project=?', (project,)).fetchone()
            if row and row['status'] == 'running' and row['heartbeat'] > now - 40:
                return None
            resume = row and row['status'] in ('running', 'retry')
            if resume:
                if row['retry_at'] > now: return None
                request = json.loads(row['request'])
                target = row['target_chapter']
                base = row['base_chapter']
                attempts = row['attempts'] + 1
                if target is not None and current_chapter >= target:
                    db.execute('UPDATE jobs SET status=? WHERE project=?', ('done', project))
                    return None
            else:
                count = request.get('chapter_count', 1)
                target = current_chapter + count if count > 0 else None
                base, attempts = current_chapter, 0
            db.execute('INSERT OR REPLACE INTO jobs VALUES (?,?,?,?,?,?,?,?,?,?)',
                       (project, json.dumps(request), base, target, 'running', owner, now, 0, attempts, ''))
            return {'owner': owner, 'request': request, 'target': target,
                    'remaining': target - current_chapter if target is not None else request['chapter_count'],
                    'attempts': attempts}

    def heartbeat(self, project, owner):
        with self.storage.db() as db:
            row = db.execute('SELECT status,owner FROM jobs WHERE project=?', (project,)).fetchone()
            if not row or row['status'] != 'running' or row['owner'] != owner: return False
            db.execute('UPDATE jobs SET heartbeat=? WHERE project=? AND owner=?', (self.clock(), project, owner))
            return True

    def finish(self, project, owner, status='done', error=''):
        with self.storage.db() as db:
            row = db.execute('SELECT attempts FROM jobs WHERE project=? AND owner=?', (project, owner)).fetchone()
            delay = min(300, 30 * 2 ** min(row[0] if row else 0, 4))
            db.execute('UPDATE jobs SET status=?,heartbeat=?,retry_at=?,error=? WHERE project=? AND owner=?',
                       (status, self.clock(), self.clock() + delay if status == 'retry' else 0, error[:500], project, owner))

    def cancel(self, project):
        with self.storage.db() as db:
            db.execute('UPDATE jobs SET status=? WHERE project=? AND status IN (?,?)',
                       ('cancelled', project, 'running', 'retry'))

    def recoverable(self):
        with self.storage.db() as db:
            return [dict(row) for row in db.execute('SELECT * FROM jobs WHERE (status=? AND heartbeat<=?) OR (status=? AND retry_at<=?)',
                                                    ('running', self.clock() - 40, 'retry', self.clock())).fetchall()]

    def status(self):
        with self.storage.db() as db:
            return [{k: row[k] for k in ('project', 'status', 'target_chapter', 'heartbeat', 'retry_at', 'attempts')}
                    for row in db.execute('SELECT * FROM jobs').fetchall()]
