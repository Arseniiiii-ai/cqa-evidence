const byId = (id) => document.getElementById(id);
const roles = { client: "Клиент", operator: "Оператор", bot: "Бот", system: "Система" };
const channels = { "agent chat": "Чат", "voice call": "Звонок" };
const state = { group: "DEV", conversations: [], record: null, evaluation: null, history: [], checklists: [], view: 0, running: false };
const sandboxIds = new Set();
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
const svgNS = "http://www.w3.org/2000/svg";
const pages = {
  workspace: ["Диалоги", "01 / Анализ", "От реплики к доказательству и итоговой оценке."],
  editor: ["Чеклист", "02 / Правила", "Управляйте критериями и сохраняйте версии оценки."],
  sandbox: ["Песочница", "03 / Сравнение", "Проверьте, как новая версия меняет оценку диалога."],
};

function node(tag, text = "", className = "") {
  const element = document.createElement(tag);
  element.textContent = text;
  element.className = className;
  return element;
}

function action(text, handler, className = "button secondary") {
  const button = node("button", text, className);
  button.type = "button";
  button.addEventListener("click", handler);
  return button;
}

function svg(tag, attributes = {}) {
  const element = document.createElementNS(svgNS, tag);
  for (const [name, value] of Object.entries(attributes)) element.setAttribute(name, value);
  return element;
}

function icon(name) {
  const element = svg("svg", { class: "icon", "aria-hidden": "true" });
  element.append(svg("use", { href: "#icon-" + name }));
  return element;
}

function emptyState(symbol, title, description) {
  const state = node("div", "", "empty-state");
  const mark = node("span", "", "empty-mark");
  mark.append(icon(symbol));
  state.append(mark, node("strong", title), node("p", description, "empty"));
  return state;
}

function stagger(element, index) {
  if (index >= 10) return;
  element.classList.add("fresh-entry");
  element.style.setProperty("--entry", index * 24 + "ms");
}

function setBusy(element, busy) {
  if (busy) element.setAttribute("aria-busy", "true");
  else element.removeAttribute("aria-busy");
}

function scoreDial(score) {
  const dial = node("div", "", "score-dial");
  dial.setAttribute("aria-hidden", "true");
  const ring = svg("svg", { viewBox: "0 0 100 100" });
  const circle = { cx: 50, cy: 50, r: 41 };
  const arc = svg("circle", { ...circle, class: "score-arc", pathLength: 100 });
  arc.style.setProperty("--score-gap", 100 - Math.max(0, Math.min(100, score)));
  if (score <= 0) arc.style.visibility = "hidden";
  ring.append(svg("circle", { ...circle, class: "score-track" }), arc);
  dial.append(ring, icon("spark"));
  return dial;
}

async function api(path, body) {
  const options = body === undefined ? undefined : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  };
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) {
    const detail = Array.isArray(data.detail)
      ? data.detail.map((error) => error.loc.slice(1).join(".") + " — " + error.msg).join(". ")
      : data.detail;
    throw new Error(detail || "Запрос не выполнен.");
  }
  return data;
}

function notice(message = "", error = false) {
  const element = byId("notice");
  element.textContent = message;
  element.hidden = !message;
  element.classList.toggle("error", error);
}

function groupQuery() {
  return "?group=" + encodeURIComponent(state.group);
}

function conversationPath(id) {
  return "/api/conversations/" + encodeURIComponent(id) + groupQuery();
}

function renderList(animated = false) {
  const language = byId("language").value;
  const channel = byId("channel").value;
  const records = state.conversations.filter((item) =>
    (!language || item.language === language) && (!channel || item.channel === channel));
  byId("count").textContent = records.length;
  const list = byId("conversation-list");
  const scrollTop = list.scrollTop;
  list.replaceChildren();
  if (!records.length) list.append(node("p", "Диалогов по выбранным фильтрам пока нет.", "empty"));
  records.forEach((item, index) => {
    const button = action("", () => selectConversation(item.conversation_id), "conversation-item");
    const selected = state.record?.conversation_id === item.conversation_id;
    button.classList.toggle("active", selected);
    button.setAttribute("aria-pressed", String(selected));
    const badge = node("span", item.language, "row-icon");
    badge.setAttribute("aria-hidden", "true");
    const label = node("span");
    label.append(node("strong", item.conversation_id), node("small", item.language + " · " + channels[item.channel], "muted"));
    button.append(badge, label);
    if (animated) stagger(button, index);
    list.append(button);
  });
  list.scrollTop = animated ? 0 : scrollTop;
}

function renderTranscript() {
  const record = state.evaluation?.record || state.record;
  const transcript = byId("transcript");
  const newRecord = transcript.dataset.record !== (record?.conversation_id || "");
  const scrollTop = transcript.scrollTop;
  transcript.dataset.record = record?.conversation_id || "";
  transcript.replaceChildren();
  const recording = byId("recording");
  recording.hidden = true;
  if (!record) {
    transcript.append(emptyState("wave", "Ожидаем диалог", "Выберите строку в наборе, чтобы открыть реплики и роли."));
    return;
  }
  if (typeof record.audio_url === "string" && /^https?:\/\//i.test(record.audio_url)) {
    recording.href = record.audio_url;
    recording.hidden = false;
  }
  record.transcript.forEach((message, index) => {
    const turn = index + 1;
    const article = node("article", "", "turn " + message.speaker);
    article.id = "turn-" + turn;
    article.tabIndex = -1;
    if (newRecord) stagger(article, index);
    const header = node("div", "", "turn-meta");
    header.append(node("strong", "#" + turn + " · " + roles[message.speaker]), node("span", message.timestamp));
    article.append(header, node("p", message.text));
    transcript.append(article);
  });
  transcript.scrollTop = newRecord ? 0 : scrollTop;
}

function showTurns(start, end = start) {
  for (const item of byId("transcript").querySelectorAll(".turn")) {
    const number = Number(item.id.split("-")[1]);
    item.classList.toggle("highlight", number >= start && number <= end);
  }
  const first = byId("turn-" + start);
  first?.focus({ preventScroll: true });
  first?.scrollIntoView({ block: "nearest", behavior: reducedMotion.matches ? "auto" : "smooth" });
  if (!reducedMotion.matches) {
    first?.animate(
      [{ boxShadow: "inset 3px 0 #dbfa97, inset 0 0 26px #dbfa9775" }, { boxShadow: "inset 3px 0 #dbfa97" }],
      { duration: 720, easing: "ease-out" },
    );
  }
}

function valueLabel(value) {
  if (value === null) return "N/A";
  if (typeof value === "boolean") return value ? "Да" : "Нет";
  return String(value);
}

function decisionOptions(config) {
  const values = config.answer_type === "boolean" ? [true, false] : Object.keys(config.scale).map(Number);
  if (config.na_policy) values.push(null);
  return values;
}

function feedbackForm(label, placeholder) {
  const form = node("form", "", "appeal-form");
  const field = node("label", label);
  const text = node("textarea");
  text.required = true;
  text.maxLength = 2000;
  text.placeholder = placeholder;
  field.append(text);
  form.append(field);
  return { form, text };
}

function submitButton(text) {
  const button = node("button", text, "button secondary");
  button.type = "submit";
  return button;
}

function appealControls(evaluation, criterion, config) {
  const appeal = evaluation.appeals.find((item) => item.criterion_id === criterion.criterion_id);
  const details = node("details");
  details.append(node("summary", appeal ? (appeal.decision ? "Решение проверяющего" : "Апелляция на проверке") : "Оспорить критерий"));
  if (appeal?.decision) {
    details.append(node("p", "Комментарий: " + appeal.comment));
    details.append(node("p", "Решение: " + valueLabel(appeal.decision.value) + " · " + appeal.decision.reason));
    return details;
  }
  const { form, text } = feedbackForm(appeal ? "Причина решения" : "Комментарий к апелляции", "Опишите основание");
  let valueSelect;
  if (appeal) {
    details.open = true;
    details.append(node("p", "Комментарий: " + appeal.comment));
    valueSelect = node("select");
    valueSelect.setAttribute("aria-label", "Решение по критерию");
    for (const value of decisionOptions(config)) {
      const option = node("option", valueLabel(value));
      option.value = JSON.stringify(value);
      valueSelect.append(option);
    }
    valueSelect.value = JSON.stringify(criterion.value);
    form.prepend(valueSelect);
  }
  const button = submitButton(appeal ? "Сохранить решение" : "Подать апелляцию");
  form.append(button);
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    button.disabled = true;
    setBusy(button, true);
    const path = appeal
      ? "/api/appeals/" + appeal.id + "/decision"
      : "/api/evaluations/" + evaluation.id + "/appeals";
    const body = appeal
      ? { value: JSON.parse(valueSelect.value), reason: text.value }
      : { criterion_id: criterion.criterion_id, comment: text.value };
    try {
      const updated = await api(path, body);
      if (state.evaluation?.id === evaluation.id) {
        state.evaluation = updated;
        renderAssessment();
        notice(appeal ? "Решение сохранено. Итог пересчитан." : "Апелляция сохранена.");
      }
    } catch (error) {
      notice(error.message, true);
    } finally {
      button.disabled = false;
      setBusy(button, false);
    }
  });
  details.append(form);
  return details;
}

function renderAssessment() {
  const container = byId("assessment");
  container.replaceChildren();
  if (state.history.length > 1) {
    const history = node("div", "", "history");
    const select = node("select");
    select.setAttribute("aria-label", "История оценок");
    for (const item of state.history) {
      const option = node("option", item.version + " · " + new Date(item.created_at).toLocaleString("ru-KZ"));
      option.value = item.id;
      select.append(option);
    }
    select.value = state.evaluation?.id || "";
    select.addEventListener("change", () => openEvaluation(select.value));
    history.append(node("small", "История оценок", "muted"), select);
    container.append(history);
  }
  const current = state.evaluation;
  if (!current) {
    const status = state.running
      ? ["Оценка в работе", "Модель проверяет диалог по выбранной версии чеклиста."]
      : ["Оценка появится здесь", "Запустите проверку, чтобы увидеть критерии и доказательства."];
    container.append(emptyState("spark", ...status));
    return;
  }
  const score = current.result.score;
  const final = current.final_result.score;
  const box = node("div", "", "scorebox");
  const content = node("div", "", "score-content");
  const display = (number) => number.toLocaleString("ru-KZ", { maximumFractionDigits: 2 });
  const scoreText = node("div", display(final.final_score), "score");
  scoreText.append(node("small", " / 100"));
  content.append(node("span", "Итоговая оценка", "eyebrow"), scoreText);
  if (current.appeals.some((item) => item.decision)) {
    content.append(node("p", "Исходная AI-оценка " + display(score.final_score)));
  }
  content.append(node("span", final.critical_flag ? "Есть грубое нарушение" : "Грубое нарушение не отмечено", final.critical_flag ? "pill flag" : "pill"));
  content.append(node("p", current.checklist.version + " · " + current.model));
  box.append(content, scoreDial(final.final_score));
  container.append(box);
  let section;
  current.result.criteria.forEach((criterion, index) => {
    const config = current.checklist.criteria.find((item) => item.criterion_id === criterion.criterion_id);
    const revised = current.final_result.criteria[index];
    if (section !== config.section) {
      section = config.section;
      container.append(node("h3", section, "eyebrow"));
    }
    const row = node("article", "", "criterion");
    stagger(row, index);
    const heading = node("div", "", "criterion-heading");
    const value = criterion.value === revised.value
      ? valueLabel(criterion.value)
      : valueLabel(criterion.value) + " → " + valueLabel(revised.value);
    const maximum = config.answer_type === "scale" && revised.value !== null
      ? " / " + Math.max(...Object.keys(config.scale).map(Number)) : "";
    const weight = config.weight ? " · вес " + config.weight : "";
    const badge = revised.value === null ? "pill na" : revised.value === true && criterion.criterion_id !== "critical" ? "pill positive" : "pill";
    heading.append(node("strong", config.title || criterion.criterion_id), node("span", value + maximum + weight, badge));
    const evidence = node("div", "", "evidence-row");
    const answer = current.answers[criterion.criterion_id];
    for (const quote of answer.evidence || []) {
      const button = action("Реплика #" + quote.turn, () => showTurns(quote.turn), "evidence");
      button.title = quote.quote;
      evidence.append(button);
    }
    if (answer.checked_turns) {
      const [start, end] = answer.checked_turns;
      evidence.append(action("Проверено #" + start + "–" + end, () => showTurns(start, end), "evidence"));
    }
    row.append(heading, node("p", criterion.rationale), evidence, appealControls(current, criterion, config));
    container.append(row);
  });
}

function renderConversation() {
  const record = state.record;
  byId("conversation-title").textContent = record?.conversation_id || "Выберите диалог";
  byId("conversation-meta").textContent = record ? record.language + " · " + channels[record.channel] : "Карточка диалога";
  byId("evaluate").disabled = !record || state.running;
  renderTranscript();
  renderAssessment();
  renderList();
}

async function openEvaluation(id, view = state.view) {
  try {
    const evaluation = await api("/api/evaluations/" + id);
    if (view !== state.view) return;
    state.evaluation = evaluation;
    renderTranscript();
    renderAssessment();
  } catch (error) {
    notice(error.message, true);
  }
}

async function selectConversation(id) {
  notice();
  const view = ++state.view;
  try {
    const data = await api(conversationPath(id));
    if (view !== state.view) return;
    state.record = data.record;
    state.history = data.history;
    state.evaluation = null;
    renderConversation();
    if (data.history.length) await openEvaluation(data.history[0].id, view);
  } catch (error) {
    notice(error.message, true);
  }
}

async function loadGroup(group) {
  state.group = group;
  const view = ++state.view;
  Object.assign(state, { record: null, evaluation: null, history: [], conversations: [] });
  sandboxIds.clear();
  byId("sandbox-results").replaceChildren();
  renderSandboxSelection();
  byId("source").textContent = "Набор ещё не загружен";
  renderConversation();
  const data = await api("/api/conversations" + groupQuery());
  if (view !== state.view) return;
  state.conversations = data.conversations;
  byId("source").textContent = group + " · " + data.source + " · " + new Date(data.imported_at).toLocaleString("ru-KZ");
  renderList(true);
  renderSandboxSelection();
}

async function importGroup(payload) {
  const group = byId("group").value.trim();
  byId("import").disabled = true;
  setBusy(byId("import"), true);
  notice("Загружаем набор " + group + "…");
  try {
    const result = await api("/api/import", { group, payload });
    await loadGroup(group);
    notice("Загружено диалогов: " + result.total_records);
  } catch (error) {
    notice(error.message, true);
  } finally {
    byId("import").disabled = false;
    setBusy(byId("import"), false);
  }
}

async function runEvaluation() {
  const view = state.view;
  const id = state.record.conversation_id;
  state.running = true;
  byId("evaluate").disabled = true;
  setBusy(byId("evaluate"), true);
  setBusy(byId("assessment"), true);
  if (!state.evaluation) renderAssessment();
  notice("Модель оценивает " + id + "…");
  try {
    const evaluation = await api("/api/evaluations", {
      group: state.group, conversation_id: id, version: byId("versions").value,
    });
    if (view !== state.view) return;
    const data = await api(conversationPath(id));
    if (view !== state.view) return;
    state.history = data.history;
    state.evaluation = evaluation;
    renderConversation();
    notice("Оценка сохранена.");
  } catch (error) {
    if (view === state.view) notice(error.message, true);
  } finally {
    state.running = false;
    byId("evaluate").disabled = !state.record;
    setBusy(byId("evaluate"), false);
    setBusy(byId("assessment"), false);
    if (!state.evaluation) renderAssessment();
  }
}

function editorField(parent, label, name, value, type = "text") {
  const field = node("label", "", "field");
  field.append(node("span", label));
  const input = node(type === "textarea" ? "textarea" : "input");
  input.name = name;
  if (input.tagName === "INPUT") input.type = type;
  input.value = value ?? "";
  field.append(input);
  parent.append(field);
  return input;
}

function addEditorCriterion(config = {}) {
  const block = node("section", "", "criterion-edit");
  stagger(block, byId("criteria-editor").childElementCount);
  const header = node("header");
  header.append(node("h3", config.title || "Новый критерий"));
  if (config.criterion_id !== "critical") header.append(action("Удалить", () => block.remove()));
  block.append(header);
  const id = editorField(block, "Код критерия", "criterion_id", config.criterion_id);
  id.required = true;
  id.readOnly = config.criterion_id === "critical";
  editorField(block, "Название", "title", config.title).required = true;
  editorField(block, "Секция", "section", config.section).required = true;
  editorField(block, "Инструкция", "instructions", config.instructions, "textarea").required = true;
  const weight = editorField(block, "Вес", "weight", config.weight ?? 10, "number");
  Object.assign(weight, { required: true, min: 0, max: 100 });
  const label = node("label", "", "field");
  label.append(node("span", "Тип ответа"));
  const answerType = node("select");
  answerType.name = "answer_type";
  for (const [value, text] of [["boolean", "Да / Нет"], ["scale", "Шкала"]]) {
    const option = node("option", text);
    option.value = value;
    answerType.append(option);
  }
  answerType.value = config.answer_type || "boolean";
  label.append(answerType);
  block.append(label);
  const scopeLabel = node("label", "", "field");
  scopeLabel.append(node("span", "Допустимые доказательства"));
  const evidenceScope = node("select");
  evidenceScope.name = "evidence_scope";
  for (const [value, text] of [["", "По всему диалогу"], ["first_operator", "Первая реплика оператора"]]) {
    const option = node("option", text);
    option.value = value;
    evidenceScope.append(option);
  }
  evidenceScope.value = config.evidence_scope || "";
  scopeLabel.append(evidenceScope);
  block.append(scopeLabel);
  editorField(block, "Условие N/A · пустое поле запрещает N/A", "na_policy", config.na_policy, "textarea");
  const steps = Object.entries(config.scale || {}).map(([key, value]) => key + " = " + value).join("\n");
  const scale = editorField(block, "Ступени · по строке «число = описание»", "scale", steps, "textarea");
  const toggleScale = () => { scale.parentElement.hidden = answerType.value !== "scale"; };
  answerType.addEventListener("change", toggleScale);
  toggleScale();
  byId("criteria-editor").append(block);
}

function renderEditor() {
  byId("criteria-editor").replaceChildren();
  const checklist = state.checklists.find((item) => item.version === byId("versions").value);
  checklist?.criteria.forEach(addEditorCriterion);
}

async function refreshChecklists(selected) {
  state.checklists = await api("/api/checklists");
  const select = byId("versions");
  select.replaceChildren();
  for (const item of state.checklists) {
    const option = node("option", item.version);
    option.value = item.version;
    select.append(option);
  }
  select.value = selected || "official-v1";
  for (const id of ["baseline-version", "candidate-version"]) {
    const field = byId(id);
    field.replaceChildren(...[...select.options].map((option) => option.cloneNode(true)));
    field.value = id === "baseline-version" ? "official-v1" : state.checklists[0].version;
  }
  renderEditor();
}

async function saveChecklist(event) {
  event.preventDefault();
  const button = byId("save-checklist");
  button.disabled = true;
  setBusy(button, true);
  try {
    const criteria = [...document.querySelectorAll(".criterion-edit")].map((block) => {
      const fields = {};
      for (const input of block.querySelectorAll("[name]")) fields[input.name] = input.value.trim();
      fields.weight = Number(fields.weight);
      fields.na_policy ||= null;
      fields.evidence_scope ||= null;
      if (fields.answer_type === "scale") {
        const entries = fields.scale.split("\n").filter((line) => line.trim()).map((line) => {
          const separator = line.indexOf("=");
          if (separator === -1) throw new Error("Добавьте описание каждой ступени шкалы после «=».");
          return [line.slice(0, separator).trim(), line.slice(separator + 1).trim()];
        });
        fields.scale = Object.fromEntries(entries);
        if (Object.keys(fields.scale).length !== entries.length) throw new Error("Ступени шкалы должны быть уникальными.");
      } else {
        delete fields.scale;
      }
      return fields;
    });
    const saved = await api("/api/checklists", { criteria });
    await refreshChecklists(saved.version);
    notice("Версия " + saved.version + " сохранена и выбрана для следующей оценки.");
  } catch (error) {
    notice(error.message, true);
  } finally {
    button.disabled = false;
    setBusy(button, false);
  }
}

function openPage(page) {
  for (const id of Object.keys(pages)) {
    const active = id === page;
    byId(id).hidden = !active;
    byId(id + "-nav").classList.toggle("active", active);
    byId(id + "-nav").setAttribute("aria-pressed", String(active));
  }
  const [title, index, description] = pages[page];
  const heading = byId("page-title");
  const changed = heading.textContent !== title;
  heading.textContent = title;
  byId("page-index").textContent = index;
  byId("page-description").textContent = description;
  if (changed && !reducedMotion.matches) {
    heading.animate([{ opacity: .35, transform: "translateY(7px)" }, { opacity: 1, transform: "translateY(0)" }],
      { duration: 290, easing: "ease-out" });
  }
}

function renderSandboxSelection() {
  const choices = byId("sandbox-choice");
  const selected = byId("sandbox-selected");
  choices.replaceChildren();
  selected.replaceChildren();
  for (const item of state.conversations) {
    if (sandboxIds.has(item.conversation_id)) continue;
    const option = node("option", item.conversation_id + " · " + item.language);
    option.value = item.conversation_id;
    choices.append(option);
  }
  for (const id of sandboxIds) {
    selected.append(action(id + " ×", () => { sandboxIds.delete(id); renderSandboxSelection(); }, "evidence"));
  }
  byId("sandbox-add").disabled = sandboxIds.size >= 5 || !choices.value;
}

function sandboxEvidence(container, answer, result) {
  if (!answer) return;
  const open = async (first, last) => {
    openPage("workspace");
    await selectConversation(result.conversation_id);
    await openEvaluation(result.previous_evaluation_id);
    showTurns(first, last);
  };
  for (const quote of answer.evidence || []) {
    container.append(action("Реплика #" + quote.turn, () => open(quote.turn, quote.turn), "evidence"));
  }
  if (answer.checked_turns) {
    const [first, last] = answer.checked_turns;
    container.append(action("Проверено #" + first + "–" + last, () => open(first, last), "evidence"));
  }
}

function renderSandboxResults(results) {
  const container = byId("sandbox-results");
  const completed = results.filter((result) => result.status === "ok").length;
  container.replaceChildren(node("p", "Обработано " + completed + " из " + results.length));
  results.forEach((result, index) => {
    const row = node("article", "", "sandbox-result");
    stagger(row, index);
    const summary = result.status === "ok"
      ? result.before + " → " + result.after + " · Δ " + result.delta
      : result.error;
    row.append(node("h3", result.conversation_id), node("p", summary, result.status === "ok" ? "comparison-score" : ""));
    if (result.status === "ok") {
      row.append(node("p", "Вызовов модели для новой версии: " + result.candidate_calls, "muted"));
      for (const change of result.changes) {
        const block = node("details");
        const weights = change.before_weight === change.after_weight ? ""
          : " · вес " + (change.before_weight ?? "—") + " → " + (change.after_weight ?? "—");
        block.append(node("summary", change.criterion_id + weights));
        for (const [label, answer] of [["До", change.before], ["После", change.after]]) {
          if (!answer) continue;
          block.append(node("p", label + " · " + valueLabel(answer.value) + " · " + answer.rationale));
          sandboxEvidence(block, answer, result);
        }
        if (change.reused) block.append(node("p", "Ответ сохранён, применены новые настройки.", "muted"));
        row.append(block);
      }
    }
    container.append(row);
  });
}

async function runSandbox(event) {
  event.preventDefault();
  const button = byId("sandbox-run");
  const group = state.group;
  const baseline = byId("baseline-version").value;
  const candidate = byId("candidate-version").value;
  if (!sandboxIds.size || baseline === candidate) {
    notice("Выберите диалоги и две разные версии чеклиста.", true);
    return;
  }
  button.disabled = true;
  setBusy(button, true);
  notice("Сравниваем версии чеклиста…");
  try {
    const results = await api("/api/sandbox", { group, baseline, candidate, conversation_ids: [...sandboxIds] });
    if (state.group !== group) return;
    renderSandboxResults(results);
    notice("Сравнение завершено.");
  } catch (error) {
    notice(error.message, true);
  } finally {
    button.disabled = false;
    setBusy(button, false);
  }
}

byId("workspace-nav").addEventListener("click", () => openPage("workspace"));
byId("editor-nav").addEventListener("click", () => openPage("editor"));
byId("sandbox-nav").addEventListener("click", () => openPage("sandbox"));
byId("sandbox-add").addEventListener("click", () => {
  const value = byId("sandbox-choice").value;
  if (value && sandboxIds.size < 5) sandboxIds.add(value);
  renderSandboxSelection();
});
byId("sandbox-form").addEventListener("submit", runSandbox);
byId("import").addEventListener("click", () => importGroup(null));
byId("dataset-file").addEventListener("change", async (event) => {
  const [file] = event.target.files;
  if (!file) return;
  try {
    await importGroup(JSON.parse(await file.text()));
  } catch (error) {
    notice("Не удалось прочитать JSON: " + error.message, true);
  } finally {
    event.target.value = "";
  }
});
byId("language").addEventListener("change", () => renderList(true));
byId("channel").addEventListener("change", () => renderList(true));
byId("evaluate").addEventListener("click", runEvaluation);
byId("versions").addEventListener("change", renderEditor);
byId("add-criterion").addEventListener("click", () => addEditorCriterion());
byId("checklist-form").addEventListener("submit", saveChecklist);

Promise.all([
  refreshChecklists(),
  api("/api/settings").then((settings) => {
    const model = byId("model");
    model.textContent = "Локальная модель · " + settings.model;
    model.classList.add("connected");
  }),
  loadGroup("DEV").catch(() => {}),
]).catch((error) => notice(error.message, true));
