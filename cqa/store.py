"""Persist local data, immutable checklist versions, and evaluation history."""

from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
from uuid import uuid4

from .model import PROMPT_VERSION
from .scoring import EvaluationError, evaluate, official_checklist


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.executescript("""
                PRAGMA journal_mode = WAL;
                CREATE TABLE IF NOT EXISTS datasets (
                    group_id TEXT PRIMARY KEY, payload TEXT NOT NULL,
                    source TEXT NOT NULL, imported_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS checklists (
                    version TEXT PRIMARY KEY, payload TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS evaluations (
                    id TEXT PRIMARY KEY, cache_key TEXT UNIQUE NOT NULL, group_id TEXT NOT NULL,
                    conversation_id TEXT NOT NULL, payload TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS evaluation_conversation
                    ON evaluations(group_id, conversation_id, created_at);
                CREATE TABLE IF NOT EXISTS appeals (
                    id TEXT PRIMARY KEY, evaluation_id TEXT NOT NULL REFERENCES evaluations(id),
                    criterion_id TEXT NOT NULL, comment TEXT NOT NULL,
                    decision TEXT, created_at TEXT NOT NULL,
                    UNIQUE(evaluation_id, criterion_id)
                );
            """)
            checklist = official_checklist()
            db.execute("INSERT OR IGNORE INTO checklists VALUES (?, ?, ?)",
                       (checklist["version"], _json(checklist), _now()))

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def save_dataset(self, group, records, source):
        with self.connection() as db:
            db.execute("""
                INSERT INTO datasets VALUES (?, ?, ?, ?)
                ON CONFLICT(group_id) DO UPDATE SET
                  payload=excluded.payload, source=excluded.source, imported_at=excluded.imported_at
            """, (group, _json(list(records.values())), source, _now()))

    def dataset(self, group):
        with self.connection() as db:
            row = db.execute("SELECT * FROM datasets WHERE group_id=?", (group,)).fetchone()
        if not row:
            return None
        records = json.loads(row["payload"])
        return {
            "records": {record["conversation_id"]: record for record in records},
            "source": row["source"], "imported_at": row["imported_at"],
        }

    def checklists(self):
        with self.connection() as db:
            rows = db.execute("SELECT payload FROM checklists ORDER BY rowid DESC").fetchall()
        return [json.loads(row["payload"]) for row in rows]

    def checklist(self, version):
        with self.connection() as db:
            row = db.execute("SELECT payload FROM checklists WHERE version=?", (version,)).fetchone()
        return json.loads(row["payload"]) if row else None

    def save_checklist(self, checklist):
        version = f"v-{uuid4().hex[:10]}"
        snapshot = {**checklist, "version": version}
        with self.connection() as db:
            db.execute("INSERT INTO checklists VALUES (?, ?, ?)", (version, _json(snapshot), _now()))
        return snapshot

    def run(self, group, record, checklist, evaluator):
        fingerprint = [group, record, checklist, evaluator.model, evaluator.base_url, PROMPT_VERSION]
        key = sha256(_json(fingerprint).encode()).hexdigest()
        with self.connection() as db:
            row = db.execute("SELECT payload FROM evaluations WHERE cache_key=?", (key,)).fetchone()
        if row:
            return json.loads(row["payload"])
        run = evaluator.run(record, checklist)
        evaluation_id = uuid4().hex
        payload = {
            "id": evaluation_id, "checklist": checklist, "record": record, "model": evaluator.model,
            "prompt_version": PROMPT_VERSION, "answers": run.answers, "result": run.result, "usage": run.usage,
        }
        with self.connection() as db:
            db.execute("INSERT OR IGNORE INTO evaluations VALUES (?, ?, ?, ?, ?, ?)",
                       (evaluation_id, key, group, record["conversation_id"], _json(payload), _now()))
            row = db.execute("SELECT payload FROM evaluations WHERE cache_key=?", (key,)).fetchone()
        return json.loads(row["payload"])

    def evaluation_history(self, group, conversation_id):
        with self.connection() as db:
            rows = db.execute("""
                SELECT id, payload, created_at FROM evaluations
                WHERE group_id=? AND conversation_id=? ORDER BY rowid DESC
            """, (group, conversation_id)).fetchall()
        history = []
        for row in rows:
            payload = json.loads(row["payload"])
            history.append({"id": row["id"], "version": payload["checklist"]["version"],
                            "created_at": row["created_at"]})
        return history

    def evaluation(self, evaluation_id):
        with self.connection() as db:
            row = db.execute("SELECT payload FROM evaluations WHERE id=?", (evaluation_id,)).fetchone()
            if not row:
                return None
            payload = json.loads(row["payload"])
            appeals = db.execute("""
                SELECT * FROM appeals WHERE evaluation_id=? ORDER BY created_at
            """, (evaluation_id,)).fetchall()
        payload["appeals"] = []
        final_answers = dict(payload["answers"])
        for row in appeals:
            appeal = dict(row)
            appeal["decision"] = json.loads(appeal["decision"]) if appeal["decision"] else None
            if appeal["decision"]:
                criterion_id = appeal["criterion_id"]
                original = final_answers[criterion_id]
                final_answers[criterion_id] = {
                    **original, "value": appeal["decision"]["value"], "rationale": appeal["decision"]["reason"],
                }
            payload["appeals"].append(appeal)
        payload["final_result"] = evaluate(payload["record"], final_answers, payload["checklist"])
        return payload

    def create_appeal(self, evaluation_id, criterion_id, comment):
        evaluation = self.evaluation(evaluation_id)
        if not evaluation or criterion_id not in evaluation["answers"]:
            raise EvaluationError("Критерий или оценка не найдены.")
        appeal_id = uuid4().hex
        with self.connection() as db:
            try:
                db.execute("INSERT INTO appeals VALUES (?, ?, ?, ?, NULL, ?)",
                           (appeal_id, evaluation_id, criterion_id, comment, _now()))
            except sqlite3.IntegrityError:
                raise EvaluationError("Для этого критерия уже есть апелляция.") from None
        return self.evaluation(evaluation_id)

    def decide_appeal(self, appeal_id, value, reason):
        with self.connection() as db:
            row = db.execute("""
                SELECT appeals.*, evaluations.payload FROM appeals
                JOIN evaluations ON evaluations.id=appeals.evaluation_id
                WHERE appeals.id=?
            """, (appeal_id,)).fetchone()
            if not row:
                raise EvaluationError("Апелляция не найдена.")
            payload = json.loads(row["payload"])
            answers = payload["answers"]
            key = row["criterion_id"]
            answers[key] = {**answers[key], "value": value, "rationale": reason}
            evaluate(payload["record"], answers, payload["checklist"])
            updated = db.execute("UPDATE appeals SET decision=? WHERE id=? AND decision IS NULL",
                                 (_json({"value": value, "reason": reason}), appeal_id))
            if not updated.rowcount:
                raise EvaluationError("Решение по апелляции уже сохранено.")
        return self.evaluation(row["evaluation_id"])
