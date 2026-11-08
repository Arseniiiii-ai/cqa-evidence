from copy import deepcopy
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock

from cqa.dataset import DatasetError, dataset_url, validate_dataset
from cqa.model import ModelRun, OllamaEvaluator, _decode_audit_answers, _ground_answers
from cqa.sandbox import compare_versions
from cqa.scoring import EvaluationError, evaluate, official_checklist
from cqa.second_pass import refine_submission
from cqa.store import Store
from cqa.submission import run_submission, validate_submission


def fixture_record():
    return {
        "conversation_id": "fixture",
        "language": "RU",
        "channel": "agent chat",
        "transcript": [
            {"timestamp": "00:01", "speaker": "client", "text": "Где скачать форму?"},
            {"timestamp": "00:02", "speaker": "bot", "text": "Подключаю оператора."},
            {"timestamp": "00:03", "speaker": "operator", "text": "Форма в личном кабинете."},
        ],
        "audio_url": None,
    }


def fixture_dataset(*records):
    return {
        "isSuccess": True,
        "error": "",
        "data": {"dataset_group_id": "DEV", "total_records": len(records), "records": list(records)},
    }


class DatasetTests(unittest.TestCase):
    def test_changes_group_without_reencoding_signature(self):
        url = "https://example.test/api?sig=a%2bb%3D&group_id=DEV&other=a%20b"
        self.assertEqual(
            dataset_url(url, "CONTROL 1"),
            "https://example.test/api?sig=a%2bb%3D&other=a%20b&group_id=CONTROL+1",
        )

    def test_preserves_all_turns(self):
        record = fixture_record()
        data = validate_dataset(fixture_dataset(record), "DEV", 1)
        self.assertEqual(data["fixture"]["transcript"], record["transcript"])

    def test_rejects_duplicate_ids_and_count_mismatches(self):
        record = fixture_record()
        with self.assertRaises(DatasetError):
            validate_dataset(fixture_dataset(record, record), "DEV", 2)
        with self.assertRaises(DatasetError):
            validate_dataset(fixture_dataset(record), "DEV", 65)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.record = fixture_record()
        self.checklist = official_checklist()
        values = {
            "contact": True,
            "discovery": None,
            "identification": None,
            "consultation": 15,
            "empathy": True,
            "dialogue": 7,
            "closure": True,
            "critical": False,
        }
        self.answers = {
            key: {"value": value, "rationale": "Тест расчёта.", "checked_turns": [1, 3]}
            for key, value in values.items()
        }

    def test_reproduces_appendix_score_and_separates_critical_flag(self):
        expected = {"final_score": 85.38, "critical_flag": False}
        self.assertEqual(evaluate(self.record, self.answers, self.checklist)["score"], expected)
        self.answers["critical"]["value"] = True
        expected["critical_flag"] = True
        self.assertEqual(evaluate(self.record, self.answers, self.checklist)["score"], expected)

    def test_reference_uses_original_turn_number(self):
        self.answers["consultation"] = {
            "value": 15,
            "rationale": "Точная цитата.",
            "evidence": [{"turn": 3, "quote": "в личном кабинете"}],
        }
        result = evaluate(self.record, self.answers, self.checklist)
        consultation = next(item for item in result["criteria"] if item["criterion_id"] == "consultation")
        self.assertEqual(consultation["evidence_turns"], [3])
        self.answers["consultation"]["evidence"][0]["turn"] = 2
        with self.assertRaisesRegex(EvaluationError, "цитата"):
            evaluate(self.record, self.answers, self.checklist)

    def test_rejects_missing_criterion_and_invalid_values(self):
        for criterion, invalid in (("contact", None), ("contact", 1), ("dialogue", 6), ("dialogue", True)):
            with self.subTest(criterion=criterion, invalid=invalid):
                original = self.answers[criterion]["value"]
                self.answers[criterion]["value"] = invalid
                with self.assertRaises(EvaluationError):
                    evaluate(self.record, self.answers, self.checklist)
                self.answers[criterion]["value"] = original
        del self.answers["critical"]
        with self.assertRaises(EvaluationError):
            evaluate(self.record, self.answers, self.checklist)

    def test_score_boundaries_and_zero_weight(self):
        for key in ("contact", "empathy", "closure"):
            self.answers[key]["value"] = False
        for key in ("consultation", "dialogue"):
            self.answers[key]["value"] = 0
        self.assertEqual(evaluate(self.record, self.answers, self.checklist)["score"]["final_score"], 0)

        for key in ("contact", "empathy", "closure"):
            self.answers[key]["value"] = True
        self.answers["consultation"]["value"] = 20
        self.answers["dialogue"]["value"] = 10
        self.assertEqual(evaluate(self.record, self.answers, self.checklist)["score"]["final_score"], 100)
        for criterion in self.checklist["criteria"]:
            criterion["weight"] = 0
        with self.assertRaisesRegex(EvaluationError, "суммарный применимый вес"):
            evaluate(self.record, self.answers, self.checklist)

    def test_model_grounding_uses_source_text_and_requires_operator_for_critical(self):
        raw = {"critical": {"value": True, "rationale": "Проверка роли.", "basis": "quote", "turns": [2]}}
        with self.assertRaisesRegex(EvaluationError, "реплику оператора"):
            _ground_answers(raw, self.record["transcript"])
        raw["critical"]["turns"] = [3]
        grounded = _ground_answers(raw, self.record["transcript"])
        self.assertEqual(grounded["critical"]["evidence"][0]["quote"], self.record["transcript"][2]["text"])

    def test_official_audit_maps_not_applicable_and_rejects_bot_evidence(self):
        transcript = self.record["transcript"]
        raw = {
            "discovery": {
                "decision": "enough_to_answer", "rationale": "Вопрос понятен.",
                "primary_turn": 3, "supporting_turns": [1], "basis": "quote",
            }
        }
        grounded = _decode_audit_answers(raw, transcript, {}, [3])
        self.assertIsNone(grounded["discovery"]["value"])
        self.assertEqual([item["turn"] for item in grounded["discovery"]["evidence"]], [3, 1])
        raw["discovery"]["supporting_turns"] = [2]
        with self.assertRaisesRegex(EvaluationError, "контекста"):
            _decode_audit_answers(raw, transcript, {}, [3])
        raw["discovery"]["supporting_turns"] = [1]
        raw["discovery"]["primary_turn"] = 2
        with self.assertRaisesRegex(EvaluationError, "реплику оператора"):
            _decode_audit_answers(raw, transcript, {}, [3])

    def test_operator_close_precludes_not_applicable_for_official_closure(self):
        evaluator = OllamaEvaluator()
        prior = {key: value for key, value in self.answers.items() if key != "closure"}
        model_answer = {"closure": {"value": None, "rationale": "Диалог оборвался.", "checked_turns": [3, 4]}}
        for speaker, text, expected in (
            ("system", "Оператор закрыл чат.", False),
            ("system", "The agent ended the conversation.", False),
            ("system", "Оператор чатты жапты.", False),
            ("system", "Клиент завершил чат.", None),
            ("client", "Оператор закрыл чат.", None),
        ):
            with self.subTest(speaker=speaker, text=text), unittest.mock.patch.object(
                evaluator, "_ask", return_value=deepcopy(model_answer),
            ):
                record = deepcopy(self.record)
                record["transcript"].append({"timestamp": "00:04", "speaker": speaker, "text": text})
                run = evaluator.run(record, self.checklist, reuse_answers=prior)
                self.assertIs(run.answers["closure"]["value"], expected)
                self.assertEqual(run.answers["closure"]["checked_turns"], [3, 4])

        record = deepcopy(self.record)
        record["transcript"].extend([
            {"timestamp": "00:04", "speaker": "system", "text": "Клиент отключился."},
            {"timestamp": "00:05", "speaker": "system", "text": "Оператор закрыл чат."},
        ])
        with unittest.mock.patch.object(evaluator, "_ask", return_value=deepcopy(model_answer)):
            run = evaluator.run(record, self.checklist, reuse_answers=prior)
        self.assertIsNone(run.answers["closure"]["value"])

    def test_appeal_recalculates_and_keeps_ai_result_after_reopening(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "cqa.sqlite"
            store = Store(path)
            evaluator = Mock(model="fixture", base_url="http://localhost")
            evaluator.run.return_value = ModelRun(self.answers, evaluate(self.record, self.answers, self.checklist), {})
            saved = store.run("DEV", self.record, self.checklist, evaluator)
            appealed = store.create_appeal(saved["id"], "closure", "Проверьте завершение.")
            result = store.decide_appeal(appealed["appeals"][0]["id"], False, "Нет завершения.")
            self.assertEqual(result["result"]["score"]["final_score"], 85.38)
            self.assertEqual(result["final_result"]["score"]["final_score"], 77.69)
            self.assertEqual(Store(path).evaluation(saved["id"]), result)
            store.run("DEV", self.record, self.checklist, evaluator)
            evaluator.run.assert_called_once()

    def test_submission_keeps_failed_record_and_checks_final_score(self):
        failed = {**self.record, "conversation_id": "failed"}
        records = {"fixture": self.record, "failed": failed}
        with TemporaryDirectory() as folder:
            store = Store(Path(folder) / "cqa.sqlite")
            evaluator = Mock(model="fixture", base_url="http://localhost")
            evaluator.run.side_effect = [
                ModelRun(self.answers, evaluate(self.record, self.answers, self.checklist), {}),
                EvaluationError("Технический сбой."),
            ]
            payload = run_submission("DEV", records, "Команда", store, evaluator, Mock())
            self.assertEqual(payload["results"][1], {
                "conversation_id": "failed", "status": "error", "score": None,
                "criteria": [], "error": "Технический сбой.",
            })
            payload["results"][0]["score"]["final_score"] = 99
            with self.assertRaisesRegex(EvaluationError, "Итог"):
                validate_submission(payload, records)

    def test_second_pass_merges_only_selected_decisions_and_recalculates(self):
        original = evaluate(self.record, self.answers, self.checklist)
        payload = {
            "schema_version": "1.0", "team_name": "Команда",
            "model": {"mode": "local", "name": "gemma3:4b"}, "results": [original],
        }
        reviewed = {
            "discovery": {"value": False, "rationale": "Уточнения не было.", "checked_turns": [1, 3]},
            "identification": {"value": False, "rationale": "Нет подтверждения.", "checked_turns": [1, 3]},
            "consultation": {"value": 0, "rationale": "Нет ответа.", "checked_turns": [1, 3]},
            "closure": {"value": False, "rationale": "Нет завершения.", "checked_turns": [1, 3]},
        }
        records = {self.record["conversation_id"]: self.record}
        with unittest.mock.patch("cqa.second_pass.audit", return_value=reviewed):
            result = refine_submission(payload, records, OllamaEvaluator(model="qwen3:4b-instruct"), Mock())
        updated = result["results"][0]
        self.assertEqual(result["model"]["name"], "gemma3:4b + qwen3:4b-instruct")
        self.assertEqual(updated["score"], {"final_score": 55.5, "critical_flag": False})
        changed = {old["criterion_id"] for old, new in zip(original["criteria"], updated["criteria"]) if old != new}
        self.assertEqual(changed, {"discovery", "identification"})
        self.assertEqual(payload["results"][0], original)

    def test_second_pass_preserves_valid_call_when_review_fails(self):
        record = {**self.record, "channel": "voice call"}
        payload = {
            "schema_version": "1.0", "team_name": "Команда",
            "model": {"mode": "local", "name": "gemma3:4b"},
            "results": [evaluate(record, self.answers, self.checklist)],
        }
        with unittest.mock.patch("cqa.second_pass.audit", side_effect=EvaluationError("Ответ оборвался.")), \
                unittest.mock.patch("cqa.second_pass.sys.stderr"):
            result = refine_submission(payload, {record["conversation_id"]: record}, OllamaEvaluator(), Mock())
        self.assertEqual(result, payload)

    def test_sandbox_reuses_unchanged_answers_and_requests_only_changed_rule(self):
        with TemporaryDirectory() as folder:
            store = Store(Path(folder) / "cqa.sqlite")
            evaluator = OllamaEvaluator()
            original_run = ModelRun(self.answers, evaluate(self.record, self.answers, self.checklist), {})
            with unittest.mock.patch.object(evaluator, "run", return_value=original_run):
                store.run("DEV", self.record, self.checklist, evaluator)
            candidate = deepcopy(self.checklist)
            candidate["version"] = "test-v2"
            candidate["criteria"][0]["weight"] = 0
            with unittest.mock.patch.object(evaluator, "_chat") as chat:
                result = compare_versions("DEV", [self.record], self.checklist, candidate, store, evaluator)[0]
                chat.assert_not_called()
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["candidate_calls"], 0)
            self.assertEqual(result["changes"][0]["criterion_id"], "contact")

            candidate["criteria"][0]["instructions"] = "Новая формулировка."
            response = {"done": True, "message": {"content": json.dumps({
                "contact": {"value": False, "rationale": "Нет приветствия.", "basis": "quote", "turns": [3]},
            })}}
            with unittest.mock.patch.object(evaluator, "_chat", return_value=response) as chat:
                result = compare_versions("DEV", [self.record], self.checklist, candidate, store, evaluator)[0]
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["candidate_calls"], 1)
            self.assertEqual(chat.call_args.args[0]["format"]["required"], ["contact"])


if __name__ == "__main__":
    unittest.main()
