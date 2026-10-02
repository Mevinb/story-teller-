"""Durable, atomic admission for a single Groq workload (never per-key quotas)."""
from contextlib import contextmanager
from dataclasses import dataclass
import json
import math
import os
import re
import sqlite3
import time
import uuid

import config
from .base import QuotaExhaustedError, QuotaDeferred


def reset_seconds(value):
    try:
        result = float(value)
        return max(0.0, result) if math.isfinite(result) else 0.0
    except (ValueError, TypeError):
        return sum(float(n) * {'d': 86400, 'h': 3600, 'm': 60, 's': 1}[u]
                   for n, u in re.findall(r'(\d+(?:\.\d+)?)\s*([dhms])', str(value).lower()))


@dataclass(frozen=True)
class Reservation:
    id: str
    group: str
    model: str
    input_tokens: int
    output_tokens: int


class QuotaScheduler:
    def __init__(self, path=None, clock=time.time, limits=None, safety=None, interval=None):
        self.path = path or config.GROQ_QUOTA_DB
        self.clock = clock
        self.limits = limits or {'rpm': config.GROQ_RPM_LIMIT, 'tpm': config.GROQ_TPM_LIMIT,
                                 'rpd': config.GROQ_RPD_LIMIT, 'tpd': config.GROQ_TPD_LIMIT,
                                 'itpm': config.GROQ_ITPM_LIMIT, 'otpm': config.GROQ_OTPM_LIMIT}
        self.safety = config.GROQ_TPM_SAFETY_MARGIN if safety is None else safety
        self.interval = config.GROQ_MIN_REQUEST_INTERVAL if interval is None else interval
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), mode=0o700, exist_ok=True)
        with self.db() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS requests (
                    id TEXT PRIMARY KEY, quota_group TEXT, model TEXT, created REAL,
                    settled REAL, input INTEGER, output INTEGER, state TEXT);
                CREATE INDEX IF NOT EXISTS request_budget ON requests(quota_group,model,created);
                CREATE TABLE IF NOT EXISTS budgets (
                    quota_group TEXT, model TEXT, data TEXT,
                    PRIMARY KEY(quota_group,model));
                CREATE TABLE IF NOT EXISTS waiters (
                    id TEXT PRIMARY KEY, quota_group TEXT, model TEXT, created REAL, heartbeat REAL);
            ''')
        os.chmod(self.path, 0o600)

    @contextmanager
    def db(self):
        db = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        db.row_factory = sqlite3.Row
        try:
            db.execute('PRAGMA journal_mode=WAL')
            db.execute('BEGIN IMMEDIATE')
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def _state(self, db, group, model):
        row = db.execute('SELECT data FROM budgets WHERE quota_group=? AND model=?', (group, model)).fetchone()
        return json.loads(row[0]) if row else {}

    def _save(self, db, group, model, state):
        db.execute('INSERT OR REPLACE INTO budgets VALUES (?,?,?)', (group, model, json.dumps(state)))

    @staticmethod
    def estimate_messages(messages):
        # Conservative UTF-8 estimate; schema and injected controls are included.
        return 12 + sum(8 + math.ceil(len(str(m.get('content', '')).encode('utf-8')) / 3)
                        for m in messages)

    def estimate_for(self, group, model, messages):
        with self.db() as db:
            factor = self._state(db, group, model).get('input_factor', 1.0)
        return math.ceil(self.estimate_messages(messages) * factor)

    def try_reserve(self, group, model, input_tokens, output_tokens, ticket=None):
        now = self.clock()
        with self.db() as db:
            # SDK read timeout is 60s. A stale reservation with no heartbeat
            # for ten minutes may have spent tokens; recover conservatively.
            db.execute('UPDATE requests SET state=?,settled=? WHERE state=? AND COALESCE(settled,created)<?',
                       ('uncertain', now, 'reserved', now - 600))
            db.execute('DELETE FROM requests WHERE created<? AND state != ?', (now - 172800, 'reserved'))
            db.execute('DELETE FROM waiters WHERE heartbeat<?', (now - 30,))
            if ticket:
                db.execute('INSERT OR IGNORE INTO waiters VALUES (?,?,?,?,?)', (ticket, group, model, now, now))
                db.execute('UPDATE waiters SET heartbeat=? WHERE id=?', (now, ticket))
                first = db.execute('SELECT id FROM waiters WHERE quota_group=? AND model=? ORDER BY created,rowid LIMIT 1', (group, model)).fetchone()
                if first and first[0] != ticket:
                    return None, 0.5, 'another queued request'
            state = self._state(db, group, model)
            rows = db.execute('SELECT * FROM requests WHERE quota_group=? AND model=? AND (created>? OR state=?)', (group, model, now - 86400, 'reserved')).fetchall()
            waits = [(state.get('cooldown', 0) - now, 'provider cooldown')]
            limits = dict(self.limits)
            # Headers are ceilings, not grounds to increase configured quotas.
            for dimension in ('tpm', 'rpd'):
                if state.get(dimension):
                    limits[dimension] = min(limits[dimension], state[dimension]) if limits.get(dimension) else state[dimension]
            for dim, window, requested in (
                ('rpm', 60, 1), ('rpd', 86400, 1),
                ('tpm', 60, input_tokens + output_tokens), ('tpd', 86400, input_tokens + output_tokens),
                ('itpm', 60, input_tokens), ('otpm', 60, output_tokens)):
                ceiling = int(limits.get(dim, 0) * (1 - self.safety))
                if not ceiling:
                    continue
                if requested > ceiling:
                    raise QuotaDeferred(f'request exceeds configured {dim.upper()} allowance ({requested}/{ceiling})', float('inf'))
                active = [r for r in rows if r['state'] == 'reserved' or (r['settled'] or r['created']) > now - window]
                def charge(r):
                    if dim.startswith('r'): return 1
                    if dim == 'itpm': return r['input']
                    if dim == 'otpm': return r['output']
                    return r['input'] + r['output']
                used = sum(charge(r) for r in active)
                if used + requested > ceiling:
                    for r in sorted(active, key=lambda r: r['settled'] or r['created']):
                        used -= charge(r)
                        if used + requested <= ceiling:
                            deadline = max(now + 0.5, (r['settled'] or r['created']) + window)
                            # Still-running streams retain their full reservation.
                            if r['state'] == 'reserved': deadline = max(deadline, now + 1)
                            waits.append((deadline - now, dim.upper()))
                            break
            if rows and self.interval:
                waits.append((max(r['created'] for r in rows) + self.interval - now, 'request spacing'))
            for label, requested in (('tokens', input_tokens + output_tokens), ('requests', 1)):
                if state.get(label + '_reset', 0) > now:
                    remaining = state.get(label + '_remaining', requested)
                    if remaining < requested:
                        waits.append((state[label + '_reset'] - now, 'provider ' + label))
            wait, reason = max(waits)
            if wait > 0:
                return None, wait, reason
            reservation = Reservation(uuid.uuid4().hex, group, model, input_tokens, output_tokens)
            db.execute('INSERT INTO requests VALUES (?,?,?,?,NULL,?,?,?)', (reservation.id, group, model, now, input_tokens, output_tokens, 'reserved'))
            # Debit observed capacity at admission; stale responses can never refund it.
            for label, amount in (('tokens', input_tokens + output_tokens), ('requests', 1)):
                if state.get(label + '_reset', 0) > now:
                    state[label + '_remaining'] = max(0, state.get(label + '_remaining', amount) - amount)
            self._save(db, group, model, state)
            if ticket: db.execute('DELETE FROM waiters WHERE id=?', (ticket,))
            return reservation, 0, ''

    def acquire(self, group, model, input_tokens, output_tokens, cancel, sleep, progress=None, max_wait=None):
        ticket = uuid.uuid4().hex
        started = self.clock()
        try:
            while True:
                cancel()
                reservation, wait, reason = self.try_reserve(group, model, input_tokens, output_tokens, ticket)
                if reservation: return reservation
                if max_wait is not None and self.clock() - started + wait > max_wait:
                    raise QuotaDeferred(reason, wait)
                if progress:
                    progress('waiting', {'model': model, 'reason': reason, 'wait_seconds': round(wait, 1),
                                         'resume_at': self.clock() + wait})
                sleep(min(wait, 5))
        finally:
            with self.db() as db:
                db.execute('DELETE FROM waiters WHERE id=?', (ticket,))

    def settle(self, reservation, usage=None, rejected=False):
        now = self.clock()
        with self.db() as db:
            if rejected:
                # A 429 has no generated tokens, but still counts as a request.
                db.execute('UPDATE requests SET input=0,output=0,state=?,settled=? WHERE id=?', ('rejected', now, reservation.id))
            elif usage and usage.get('prompt_tokens') is not None and usage.get('completion_tokens') is not None:
                row = db.execute('SELECT input FROM requests WHERE id=?', (reservation.id,)).fetchone()
                if row and usage['prompt_tokens'] > row['input']:
                    state = self._state(db, reservation.group, reservation.model)
                    state['input_factor'] = state.get('input_factor', 1.0) * usage['prompt_tokens'] / max(1, row['input']) * 1.15
                    self._save(db, reservation.group, reservation.model, state)
                db.execute('UPDATE requests SET input=?,output=?,state=?,settled=? WHERE id=?',
                           (usage['prompt_tokens'], usage['completion_tokens'], 'done', now, reservation.id))
            else:
                # Timeout, cancellation or missing usage: do not erase possible spend.
                db.execute('UPDATE requests SET state=?,settled=? WHERE id=?', ('uncertain', now, reservation.id))

    def touch(self, reservation):
        with self.db() as db:
            db.execute('UPDATE requests SET settled=? WHERE id=? AND state=?', (self.clock(), reservation.id, 'reserved'))

    def release_unsent(self, reservation):
        with self.db() as db:
            db.execute('DELETE FROM requests WHERE id=?', (reservation.id,))
            state = self._state(db, reservation.group, reservation.model)
            for label, amount in (('tokens', reservation.input_tokens + reservation.output_tokens), ('requests', 1)):
                if state.get(label + '_reset', 0) > self.clock():
                    state[label + '_remaining'] = state.get(label + '_remaining', 0) + amount
            self._save(db, reservation.group, reservation.model, state)

    def observe(self, reservation, headers, retry_after=0):
        now = self.clock()
        with self.db() as db:
            state = self._state(db, reservation.group, reservation.model)
            for label, dim in (('tokens', 'tpm'), ('requests', 'rpd')):
                try:
                    ceiling = int(headers.get('x-ratelimit-limit-' + label, 0))
                    if ceiling > 0: state[dim] = ceiling
                    remaining = int(headers['x-ratelimit-remaining-' + label])
                    duration = reset_seconds(headers.get('x-ratelimit-reset-' + label, ''))
                except (KeyError, ValueError, TypeError):
                    continue
                if duration <= 0: continue
                # Deduct other pending calls and calls admitted after this one.
                own = db.execute('SELECT created FROM requests WHERE id=?', (reservation.id,)).fetchone()
                peers = db.execute('SELECT * FROM requests WHERE quota_group=? AND model=? AND id!=? AND (state=? OR created>?)',
                                   (reservation.group, reservation.model, reservation.id, 'reserved', own[0])).fetchall()
                remaining -= sum(1 if label == 'requests' else r['input'] + r['output'] for r in peers)
                if state.get(label + '_reset', 0) > now:
                    remaining = min(remaining, state.get(label + '_remaining', remaining))
                state[label + '_remaining'] = max(0, remaining)
                state[label + '_reset'] = max(state.get(label + '_reset', 0), now + duration)
            if retry_after:
                state['cooldown'] = max(state.get('cooldown', 0), now + retry_after)
                state['rate_limits'] = state.get('rate_limits', 0) + 1
            self._save(db, reservation.group, reservation.model, state)

    def status(self):
        now = self.clock()
        with self.db() as db:
            groups = []
            for row in db.execute('SELECT * FROM budgets').fetchall():
                data = json.loads(row['data'])
                usage = db.execute('SELECT COUNT(*),COALESCE(SUM(input+output),0),SUM(state="uncertain") FROM requests WHERE quota_group=? AND model=? AND created>?', (row['quota_group'], row['model'], now - 86400)).fetchone()
                groups.append({'group': row['quota_group'], 'model': row['model'], 'requests_24h': usage[0],
                               'tokens_24h': usage[1], 'uncertain_requests': usage[2] or 0,
                               'cooldown_seconds': max(0, data.get('cooldown', 0) - now), **data})
            return {'groups': groups, 'configured_limits': self.limits, 'utilization': 1 - self.safety}
