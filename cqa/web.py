"""Local CQA interface and API."""

from pathlib import Path
import os
import re
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, StrictBool, StrictInt

from .dataset import DatasetError, load_dataset, summarize, validate_dataset
from .model import model_from_environment
from .sandbox import compare_versions
from .scoring import EvaluationError, official_checklist
from .store import Store


STATIC = Path(__file__).with_name("static")


class ImportRequest(BaseModel):
    group: str = Field(min_length=1, max_length=100)
    payload: dict | None = None


class RunRequest(BaseModel):
    group: str
    conversation_id: str
    version: str


class SandboxRequest(BaseModel):
    group: str
    baseline: str
    candidate: str
    conversation_ids: list[str] = Field(min_length=1, max_length=5)


class AppealRequest(BaseModel):
    criterion_id: str
    comment: str = Field(min_length=1, max_length=2000)


class DecisionRequest(BaseModel):
    value: StrictBool | StrictInt | None
    reason: str = Field(min_length=1, max_length=2000)


class Criterion(BaseModel):
    criterion_id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,39}$")
    title: str = Field(min_length=1, max_length=100)
    section: str = Field(min_length=1, max_length=100)
    instructions: str = Field(min_length=1, max_length=3000)
    answer_type: str
    weight: StrictInt = Field(ge=0, le=100)
    na_policy: str | None = Field(default=None, max_length=1000)
    scale: dict[str, str] | None = None
    evidence_scope: Literal["first_operator"] | None = None


class ChecklistRequest(BaseModel):
    criteria: list[Criterion] = Field(min_length=2, max_length=16)


def _checked_text(text):
    if not text.strip():
        raise HTTPException(422, "Заполните обязательные текстовые поля.")
    return text.strip()


def _checklist_payload(data):
    criteria = []
    for criterion in data.criteria:
        item = criterion.model_dump()
        for key in ("title", "section", "instructions"):
            item[key] = _checked_text(item[key])
        item["na_policy"] = item["na_policy"].strip() or None if item["na_policy"] else None
        if item["evidence_scope"] is None:
            item.pop("evidence_scope")
        if item["answer_type"] == "boolean":
            item.pop("scale")
        elif item["answer_type"] == "scale":
            scale = item["scale"] or {}
            if (
                len(scale) < 2 or "0" not in scale
                or any(not re.fullmatch(r"0|[1-9][0-9]{0,2}", point) for point in scale)
                or any(not label.strip() for label in scale.values())
            ):
                raise HTTPException(422, "Шкале нужны ноль, положительная ступень и описание каждой ступени.")
        else:
            raise HTTPException(422, "Неизвестный тип ответа.")
        criteria.append(item)
    by_id = {item["criterion_id"]: item for item in criteria}
    if len(by_id) != len(criteria):
        raise HTTPException(422, "Коды критериев должны быть уникальными.")
    critical = by_id.get("critical")
    if not critical or critical["answer_type"] != "boolean" or critical["weight"] or critical["na_policy"]:
        raise HTTPException(422, "Флаг critical должен быть бинарным, с нулевым весом и без N/A.")
    if not any(item["weight"] > 0 and not item["na_policy"] for item in criteria):
        raise HTTPException(422, "Нужен хотя бы один обязательный критерий с положительным весом.")
    return {"criteria": criteria, "rounding": official_checklist()["rounding"]}


def create_app(db_path=None, evaluator=None):
    store = Store(db_path or os.environ.get("CQA_DB_PATH", "data/cqa.sqlite3"))
    evaluator = evaluator or model_from_environment()
    app = FastAPI(title="CQA Evidence")
    app.mount("/assets", StaticFiles(directory=STATIC), name="assets")

    def dataset(group):
        data = store.dataset(group)
        if data is None:
            raise HTTPException(404, "Набор ещё не загружен.")
        return data

    def conversation(group, conversation_id):
        record = dataset(group)["records"].get(conversation_id)
        if record is None:
            raise HTTPException(404, "Диалог не найден.")
        return record

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/settings")
    def settings():
        return {"model": evaluator.model}

    @app.post("/api/import")
    def import_dataset(request: ImportRequest):
        group = _checked_text(request.group)
        try:
            count = 65 if group == "DEV" else None
            if request.payload is not None:
                records = validate_dataset(request.payload, group, count)
                source = "JSON"
            else:
                path = os.environ.get("CQA_DATASET_FILE") if group == "DEV" else None
                records = load_dataset(group=group, expected_count=count, path=path,
                                       signed_url=os.environ.get("CQA_DATASET_URL"))
                source = "Локальный кэш" if path else "API организаторов"
        except DatasetError as error:
            raise HTTPException(422, str(error)) from None
        store.save_dataset(group, records, source)
        return summarize(records)

    @app.get("/api/conversations")
    def conversations(group: str = "DEV"):
        data = dataset(group)
        records = data["records"]
        return {
            "source": data["source"], "imported_at": data["imported_at"], **summarize(records),
            "conversations": [
                {key: record[key] for key in ("conversation_id", "language", "channel")}
                for record in records.values()
            ],
        }

    @app.get("/api/conversations/{conversation_id}")
    def get_conversation(conversation_id: str, group: str = "DEV"):
        return {"record": conversation(group, conversation_id),
                "history": store.evaluation_history(group, conversation_id)}

    @app.get("/api/checklists")
    def checklists():
        return store.checklists()

    @app.post("/api/checklists")
    def create_checklist(request: ChecklistRequest):
        return store.save_checklist(_checklist_payload(request))

    @app.post("/api/evaluations")
    def run_evaluation(request: RunRequest):
        record = conversation(request.group, request.conversation_id)
        checklist = store.checklist(request.version)
        if not checklist:
            raise HTTPException(404, "Версия чеклиста не найдена.")
        try:
            saved = store.run(request.group, record, checklist, evaluator)
        except EvaluationError as error:
            raise HTTPException(502, str(error)) from None
        return store.evaluation(saved["id"])

    @app.post("/api/sandbox")
    def sandbox(request: SandboxRequest):
        baseline = store.checklist(request.baseline)
        candidate = store.checklist(request.candidate)
        if not baseline or not candidate or request.baseline == request.candidate:
            raise HTTPException(422, "Выберите две разные версии чеклиста.")
        if len(set(request.conversation_ids)) != len(request.conversation_ids):
            raise HTTPException(422, "Выберите уникальные диалоги.")
        records = dataset(request.group)["records"]
        if any(key not in records for key in request.conversation_ids):
            raise HTTPException(404, "Один из диалогов не найден.")
        return compare_versions(
            request.group, [records[key] for key in request.conversation_ids], baseline, candidate, store, evaluator,
        )

    @app.get("/api/evaluations/{evaluation_id}")
    def get_evaluation(evaluation_id: str):
        result = store.evaluation(evaluation_id)
        if not result:
            raise HTTPException(404, "Оценка не найдена.")
        return result

    @app.post("/api/evaluations/{evaluation_id}/appeals")
    def appeal(evaluation_id: str, request: AppealRequest):
        try:
            return store.create_appeal(evaluation_id, request.criterion_id, _checked_text(request.comment))
        except EvaluationError as error:
            raise HTTPException(422, str(error)) from None

    @app.post("/api/appeals/{appeal_id}/decision")
    def decision(appeal_id: str, request: DecisionRequest):
        try:
            return store.decide_appeal(appeal_id, request.value, _checked_text(request.reason))
        except EvaluationError as error:
            raise HTTPException(422, str(error)) from None

    return app
