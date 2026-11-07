from argparse import ArgumentParser
import json
import os
from pathlib import Path
import sys

from .dataset import DatasetError, load_dataset, summarize
from .model import model_from_environment
from .scoring import EvaluationError, evaluate, official_checklist
from .store import Store
from .submission import run_submission, validate_submission, write_submission


def main():
    parser = ArgumentParser(description="Импорт, оценка и выгрузка CQA Evidence.")
    subcommands = parser.add_subparsers(dest="command", required=True)
    inspect = subcommands.add_parser("inspect", help="Проверить набор и вывести агрегаты.")
    score = subcommands.add_parser("score", help="Проверить готовые ответы и вычислить итог.")
    ai = subcommands.add_parser("evaluate", help="Оценить один диалог через Ollama.")
    batch = subcommands.add_parser("batch", help="Оценить DEV или контроль и сформировать results.json.")
    validate = subcommands.add_parser("validate", help="Сверить готовый results.json с исходным набором.")
    for command in (inspect, score, ai, batch, validate):
        command.add_argument("--group", default="DEV")
        command.add_argument("--expected-count", type=int)
        command.add_argument("--dataset", type=Path, help="Локальный JSON в формате ответа API.")
    score.add_argument("--conversation-id", required=True)
    score.add_argument("--answers", required=True, type=Path)
    ai.add_argument("--conversation-id", required=True)
    for command in (batch, validate):
        command.add_argument("--split", choices=("dev", "prod"), required=True)
        command.add_argument("--results-dir", type=Path, default=Path("results"))
    batch.add_argument("--team", required=True)
    args = parser.parse_args()

    try:
        count = args.expected_count
        if args.command in ("batch", "validate"):
            if (args.split == "dev") != (args.group == "DEV"):
                raise EvaluationError("Для dev используйте DEV, для prod укажите контрольный group_id.")
            count = 65 if args.split == "dev" else 10
            result_path = args.results_dir / args.split / "results.json"
        records = load_dataset(
            group=args.group,
            expected_count=count,
            path=args.dataset,
            signed_url=os.environ.get("CQA_DATASET_URL"),
        )
        if args.command == "inspect":
            result = summarize(records)
        elif args.command == "validate":
            try:
                payload = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                raise EvaluationError("Не удалось прочитать results.json.") from None
            validate_submission(payload, records)
            result = {"path": str(result_path), "validated": len(records)}
        elif args.command == "batch":
            store = Store(os.environ.get("CQA_DB_PATH", "data/cqa.sqlite3"))
            payload = run_submission(
                args.group, records, args.team, store, model_from_environment(),
                lambda done, total, key, status: print(f"{done}/{total}  {key}  {status}", file=sys.stderr),
            )
            write_submission(payload, result_path)
            errors = sum(item["status"] == "error" for item in payload["results"])
            result = {"path": str(result_path), "records": len(records), "errors": errors}
        else:
            record = records.get(args.conversation_id)
            if record is None:
                raise EvaluationError("Диалог с указанным ID отсутствует в наборе.")
            checklist = official_checklist()
            if args.command == "evaluate":
                store = Store(os.environ.get("CQA_DB_PATH", "data/cqa.sqlite3"))
                saved = store.run(args.group, record, checklist, model_from_environment())
                result = {"model": saved["model"], "usage": saved["usage"], "result": saved["result"]}
            else:
                try:
                    answers = json.loads(args.answers.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    raise EvaluationError("Не удалось прочитать JSON с ответами.") from None
                result = evaluate(record, answers, checklist)
    except (DatasetError, EvaluationError) as error:
        parser.exit(1, f"{error}\n")

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
