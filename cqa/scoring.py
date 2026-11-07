"""Validate criterion answers and convert them to a CQA result record."""

from fractions import Fraction
import json
from pathlib import Path


class EvaluationError(ValueError):
    pass


def official_checklist():
    path = Path(__file__).with_name("official-v1.json")
    return json.loads(path.read_text(encoding="utf-8"))


def _evidence_turns(answer, transcript, criterion_id):
    evidence = answer.get("evidence", [])
    checked = answer.get("checked_turns")
    if not isinstance(evidence, list):
        raise EvaluationError(f"{criterion_id}: evidence должен быть массивом.")

    def valid_turn(turn):
        return type(turn) is int and 1 <= turn <= len(transcript)

    turns = set()
    for item in evidence:
        if not isinstance(item, dict) or not valid_turn(item.get("turn")):
            raise EvaluationError(f"{criterion_id}: номер реплики вне транскрипта.")
        turn = item["turn"]
        quote = item.get("quote")
        if not isinstance(quote, str) or not quote.strip() or quote not in transcript[turn - 1]["text"]:
            raise EvaluationError(f"{criterion_id}: цитата отсутствует в указанной реплике.")
        turns.add(turn)

    if checked is not None:
        if not isinstance(checked, list) or len(checked) != 2 or not all(map(valid_turn, checked)):
            raise EvaluationError(f"{criterion_id}: укажите две границы проверенного диапазона.")
        start, end = checked
        if start > end:
            raise EvaluationError(f"{criterion_id}: начало диапазона позже конца.")
        turns.update((start, end))
    if not turns:
        raise EvaluationError(f"{criterion_id}: требуется цитата или проверенный диапазон.")
    return sorted(turns)


def _validated_value(criterion, value):
    criterion_id = criterion["criterion_id"]
    if value is None and criterion["na_policy"]:
        return None
    if criterion["answer_type"] == "boolean" and type(value) is bool:
        return Fraction(int(value))
    if criterion["answer_type"] == "scale" and type(value) is int and str(value) in criterion["scale"]:
        return Fraction(value, max(map(int, criterion["scale"])))
    raise EvaluationError(f"{criterion_id}: недопустимое значение по чеклисту.")


def _rounded_score(score, places):
    factor = 10**places
    scaled = score * factor
    digits, remainder = divmod(scaled.numerator, scaled.denominator)
    return (digits + (2 * remainder >= scaled.denominator)) / factor


def validate_answer(criterion, answer, transcript):
    criterion_id = criterion["criterion_id"]
    if not isinstance(answer, dict) or "value" not in answer:
        raise EvaluationError(f"{criterion_id}: отсутствует поле value.")
    value = answer["value"]
    portion = _validated_value(criterion, value)
    rationale = answer.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        raise EvaluationError(f"{criterion_id}: требуется обоснование.")
    turns = _evidence_turns(answer, transcript, criterion_id)
    return portion, {"criterion_id": criterion_id, "value": value, "rationale": rationale.strip(), "evidence_turns": turns}


def evaluate(record, answers, checklist):
    criteria = checklist["criteria"]
    if not isinstance(answers, dict) or set(answers) != {item["criterion_id"] for item in criteria}:
        raise EvaluationError("Нужно передать ровно один ответ на каждый критерий чеклиста.")

    earned = Fraction(0)
    applicable_weight = 0
    results = []
    for criterion in criteria:
        criterion_id = criterion["criterion_id"]
        portion, result = validate_answer(criterion, answers[criterion_id], record["transcript"])
        if portion is not None:
            weight = criterion["weight"]
            earned += weight * portion
            applicable_weight += weight
        results.append(result)

    if applicable_weight == 0:
        raise EvaluationError("Итог не определён: суммарный применимый вес равен нулю.")
    final_score = _rounded_score(100 * earned / applicable_weight, checklist["rounding"]["decimal_places"])
    return {
        "conversation_id": record["conversation_id"],
        "status": "ok",
        "criteria": results,
        "score": {"final_score": final_score, "critical_flag": answers["critical"]["value"]},
    }
