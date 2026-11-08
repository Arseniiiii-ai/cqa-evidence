"""Decision choices for criteria whose rules match the official checklist."""

import re

from .scoring import official_checklist


_SEMANTICS = ("instructions", "answer_type", "na_policy", "scale", "evidence_scope")
_OFFICIAL = {criterion["criterion_id"]: criterion for criterion in official_checklist()["criteria"]}

RULES = {
    "contact": (
        {"greeting_and_name": True, "one_or_both_missing": False},
        "Check only the first operator turn for both a greeting and a name in the customer's language.",
    ),
    "discovery": (
        {"enough_to_answer": None, "needed_detail_was_asked": True, "needed_detail_was_not_asked": False},
        "Choose enough_to_answer when the request was already clear enough for useful help. A question can "
        "clarify a missing topic, symptom, currency or timing. An answer alone is not a question. "
        "Identity verification is judged separately.",
    ),
    "identification": (
        {"general_guidance_only": None, "verified_before_access": True, "not_verified_before_access": False},
        "Directions for the customer to look up information independently are general guidance. "
        "When the operator checks, discloses or acts on a particular account, look for earlier identity confirmation. "
        "Disclosing a restriction on this customer's account requires verification.",
    ),
    "consultation": (
        {"request_not_stated": None, "answered_or_correct_next_step": 20, "only_minor_gap": 15,
         "essential_gap": 10, "vague_hint": 5, "no_help_or_wrong": 0},
        "Find the actual question or questions. Use the top score for a correct next step or a fully addressed "
        "request. A customer's confirmation can show that the instructions worked. Choose request_not_stated "
        "only when the customer never explained what help they needed. Judge the requested help separately from misconduct.",
    ),
    "empathy": (
        {"polite_and_addressed_concern_if_any": True, "rude_or_ignored_concern": False},
        "Polite service passes if there was no expressed concern. If the customer described frustration or a practical "
        "difficulty, look for a response to that specific issue. Helpful action can address a concern.",
    ),
    "dialogue": (
        {"clear": 10, "minor_disruption": 7, "had_to_reask_due_to_wording": 4, "hard_to_follow": 0},
        "Judge clarity and order of the operator's turns. A new question differs from a repeated question caused "
        "by unclear wording. Consider actual operator waiting time, excluding the bot phase before handoff.",
    ),
    "closure": (
        {"customer_left_before_closure": None, "farewell_and_recap_or_more_help": True, "incomplete_closure": False},
        "A customer leaving during the exchange may prevent closure. Agent-initiated closing does not qualify. "
        "A farewell must accompany a concrete outcome, next step or offer of further help. A generic thank-you, "
        "you're welcome, or wish for a good day alone does not recap the outcome.",
    ),
}

GROUPS = (("contact", "discovery", "identification"), ("consultation", "empathy", "dialogue", "closure"))

_AGENT_CLOSED_CHAT = re.compile(
    r"(?:(?:the )?(?:operator|agent) (?:closed|ended) (?:the )?(?:chat|conversation)"
    r"|оператор (?:закрыл|завершил) (?:чат|диалог)"
    r"|оператор (?:чатты|диалогты) (?:жапты|аяқтады))[.!]?",
    re.IGNORECASE,
)
_CLIENT_LEFT_CHAT = re.compile(
    r"(?:(?:the )?(?:client|customer) (?:left|ended) (?:the )?(?:chat|conversation)"
    r"|(?:the )?(?:client|customer) disconnected"
    r"|клиент (?:покинул|завершил) (?:чат|диалог)|клиент отключился"
    r"|клиент чаттан шықты)[.!]?",
    re.IGNORECASE,
)


def uses_official_choices(criterion):
    key = criterion["criterion_id"]
    return key in RULES and all(criterion.get(field) == _OFFICIAL[key].get(field) for field in _SEMANTICS)


def correct_chat_closure(record, answer):
    transcript = record["transcript"]
    if record["channel"] != "agent chat" or not transcript or answer["value"] is not None:
        return answer
    last = transcript[-1]
    if last["speaker"] != "system" or not _AGENT_CLOSED_CHAT.fullmatch(last["text"].strip()):
        return answer

    last_operator = next(
        (i for i in range(len(transcript) - 1, 0, -1) if transcript[i - 1]["speaker"] == "operator"),
        None,
    )
    if last_operator is None or any(
        turn["speaker"] == "system" and _CLIENT_LEFT_CHAT.fullmatch(turn["text"].strip())
        for turn in transcript[last_operator:-1]
    ):
        return answer
    return {
        "value": False,
        "rationale": "Чат завершил оператор. Полное завершение по чеклисту перед закрытием не зафиксировано.",
        "checked_turns": [last_operator, len(transcript)],
    }
