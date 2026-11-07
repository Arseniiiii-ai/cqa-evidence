"""Create and validate the organizers' JSON 1.0 submission."""

import json
from pathlib import Path
from tempfile import NamedTemporaryFile

from .scoring import EvaluationError, evaluate, official_checklist


def validate_submission(payload, records):
    if not isinstance(payload, dict) or payload.get("schema_version") != "1.0":
        raise EvaluationError("Требуется schema_version 1.0.")
    team = payload.get("team_name")
    model = payload.get("model")
    if not isinstance(team, str) or not team.strip():
        raise EvaluationError("Укажите название команды.")
    if (
        not isinstance(model, dict) or model.get("mode") not in ("local", "external_api", "hybrid")
        or not isinstance(model.get("name"), str) or not model["name"].strip()
    ):
        raise EvaluationError("Укажите режим и название использованной модели.")
    results = payload.get("results")
    if not isinstance(results, list) or len(results) != len(records):
        raise EvaluationError("Количество результатов не совпадает с набором.")
    remaining_ids = set(records)
    checklist = official_checklist()
    for result in results:
        if not isinstance(result, dict):
            raise EvaluationError("Результат должен быть объектом.")
        conversation_id = result.get("conversation_id")
        if not isinstance(conversation_id, str) or conversation_id not in remaining_ids:
            raise EvaluationError("Лишний или повторный conversation_id.")
        remaining_ids.remove(conversation_id)
        if result.get("status") == "error":
            reason = result.get("error")
            if result.get("criteria") != [] or "score" not in result or result["score"] is not None:
                raise EvaluationError("В записи error нужны пустые criteria и score=null.")
            if not isinstance(reason, str) or not reason.strip():
                raise EvaluationError("Запись error должна содержать причину.")
            continue
        if result.get("status") != "ok" or not isinstance(result.get("criteria"), list):
            raise EvaluationError("Для результата нужны статус ok и массив criteria.")
        record = records[conversation_id]
        answers = {}
        for criterion in result["criteria"]:
            if not isinstance(criterion, dict):
                raise EvaluationError("Некорректная запись критерия.")
            criterion_id = criterion.get("criterion_id")
            turns = criterion.get("evidence_turns")
            if not isinstance(criterion_id, str) or criterion_id in answers or "value" not in criterion:
                raise EvaluationError("Критерий пропущен или дублируется.")
            if (
                not isinstance(turns, list) or not turns
                or any(type(turn) is not int or not 1 <= turn <= len(record["transcript"]) for turn in turns)
            ):
                raise EvaluationError("evidence_turns не соответствует исходному транскрипту.")
            answers[criterion_id] = {
                "value": criterion["value"], "rationale": criterion.get("rationale"),
                "checked_turns": [min(turns), max(turns)],
            }
        expected = evaluate(record, answers, checklist)["score"]
        score = result.get("score")
        if (
            not isinstance(score, dict) or type(score.get("final_score")) not in (int, float)
            or score["final_score"] != expected["final_score"]
            or score.get("critical_flag") is not expected["critical_flag"]
        ):
            raise EvaluationError("Итог или флаг грубого нарушения не совпадает с критериями.")


def run_submission(group, records, team, store, evaluator, progress):
    if not team.strip():
        raise EvaluationError("Укажите название команды.")
    checklist = official_checklist()
    results = []
    for record in records.values():
        conversation_id = record["conversation_id"]
        try:
            saved = store.run(group, record, checklist, evaluator)
            result = saved["result"]
        except EvaluationError as error:
            result = {
                "conversation_id": conversation_id, "status": "error",
                "criteria": [], "score": None, "error": str(error),
            }
        results.append(result)
        progress(len(results), len(records), conversation_id, result["status"])
    payload = {
        "schema_version": "1.0", "team_name": team.strip(),
        "model": {"mode": "local", "name": evaluator.model}, "results": results,
    }
    validate_submission(payload, records)
    return payload


def write_submission(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary_path = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
