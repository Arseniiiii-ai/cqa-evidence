"""Compare two checklist versions without carrying over appeal decisions."""

from .scoring import EvaluationError


SEMANTIC_FIELDS = ("instructions", "answer_type", "na_policy", "scale", "evidence_scope")


def compare_versions(group, records, baseline, candidate, store, evaluator):
    previous = {item["criterion_id"]: item for item in baseline["criteria"]}
    proposed = {item["criterion_id"]: item for item in candidate["criteria"]}
    reusable = {
        key for key, item in proposed.items()
        if key in previous and all(item.get(field) == previous[key].get(field) for field in SEMANTIC_FIELDS)
    }
    changed = sorted(key for key in previous.keys() | proposed.keys() if previous.get(key) != proposed.get(key))
    results = []
    for record in records:
        result = {"conversation_id": record["conversation_id"]}
        try:
            old = store.run(group, record, baseline, evaluator)
            reused = {key: old["answers"][key] for key in reusable}
            new = evaluator.run(record, candidate, reuse_answers=reused)
            before = old["result"]["score"]["final_score"]
            after = new.result["score"]["final_score"]
            result.update({
                "status": "ok", "before": before, "after": after, "delta": round(after - before, 2),
                "previous_evaluation_id": old["id"],
                "candidate_calls": new.usage["calls"],
                "changes": [
                    {"criterion_id": key, "reused": key in reusable, "before": old["answers"].get(key),
                     "after": new.answers.get(key), "before_weight": previous.get(key, {}).get("weight"),
                     "after_weight": proposed.get(key, {}).get("weight")}
                    for key in changed
                ],
            })
        except EvaluationError as error:
            result.update({"status": "error", "error": str(error)})
        results.append(result)
    return results
