"""Import the organizers' dataset while preserving the source turn order."""

from collections import Counter
import json
from pathlib import Path
from urllib.error import URLError
from urllib.parse import quote_plus, unquote_plus, urlsplit
from urllib.request import urlopen


class DatasetError(ValueError):
    pass


def dataset_url(signed_url, group):
    parts = urlsplit(signed_url)
    if parts.scheme != "https" or not parts.hostname:
        raise DatasetError("Для API нужна подписанная HTTPS-ссылка.")

    # Re-encode only the group so other signed query parameters stay byte-identical.
    params = [
        param for param in parts.query.split("&")
        if param and unquote_plus(param.split("=", 1)[0]) != "group_id"
    ]
    params.append(f"group_id={quote_plus(group)}")
    return parts._replace(query="&".join(params)).geturl()


def validate_dataset(payload, group, expected_count=None):
    if not isinstance(payload, dict) or payload.get("isSuccess") is not True or payload.get("error"):
        raise DatasetError("Источник вернул ошибку загрузки.")
    data = payload.get("data")
    if not isinstance(data, dict) or data.get("dataset_group_id") != group:
        raise DatasetError("Получен набор с неожиданным group_id.")

    records = data.get("records")
    total = data.get("total_records")
    if not isinstance(records, list) or type(total) is not int or total != len(records):
        raise DatasetError("Число записей не совпадает с total_records.")
    if expected_count is not None and len(records) != expected_count:
        raise DatasetError(f"Ожидалось {expected_count} записей, получено {len(records)}.")

    by_id = {}
    for index, record in enumerate(records, 1):
        if not isinstance(record, dict):
            raise DatasetError(f"Запись {index} не является объектом.")
        conversation_id = record.get("conversation_id")
        if not isinstance(conversation_id, str) or not conversation_id.strip() or conversation_id in by_id:
            raise DatasetError(f"Запись {index} содержит пустой или повторяющийся ID.")
        if record.get("language") not in ("RU", "KZ", "EN"):
            raise DatasetError(f"Запись {index} содержит неподдерживаемый язык.")
        if record.get("channel") not in ("agent chat", "voice call"):
            raise DatasetError(f"Запись {index} содержит неизвестный канал.")
        transcript = record.get("transcript")
        if not isinstance(transcript, list) or not transcript:
            raise DatasetError(f"Запись {index} содержит некорректный транскрипт.")
        for turn, message in enumerate(transcript, 1):
            if (
                not isinstance(message, dict)
                or message.get("speaker") not in ("operator", "client", "bot", "system")
                or not isinstance(message.get("text"), str)
                or not isinstance(message.get("timestamp"), str)
            ):
                raise DatasetError(f"В записи {index} некорректна реплика {turn}.")
        by_id[conversation_id] = record
    return by_id


def load_dataset(*, group, expected_count=None, path=None, signed_url=None):
    if path is not None:
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise DatasetError("Не удалось прочитать локальный JSON набора.") from None
    else:
        if not signed_url:
            raise DatasetError("Укажите CQA_DATASET_URL или передайте --dataset.")
        try:
            with urlopen(dataset_url(signed_url, group), timeout=20) as response:
                payload = json.load(response)
        except (URLError, TimeoutError, ValueError):
            raise DatasetError("Не удалось получить JSON набора по API.") from None
    return validate_dataset(payload, group, expected_count)


def summarize(records):
    return {
        "total_records": len(records),
        "languages": dict(Counter(record["language"] for record in records.values())),
        "channels": dict(Counter(record["channel"] for record in records.values())),
    }
