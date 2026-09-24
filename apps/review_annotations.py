"""Streamlit UI for reviewing LLM annotations of API pairs."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
from typing import Any

import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "data/retrieval/catalog.jsonl"
ANNOTATIONS_PATH = ROOT / "data/annotations/pilot_pair_annotations.jsonl"
PILOT_PAIRS_PATH = ROOT / "data/retrieval_analysis/pilot_pairs.jsonl"
AUDIT_PAIRS_PATH = ROOT / "data/retrieval_analysis/relation_audit_pairs.jsonl"
REVIEWS_PATH = ROOT / "data/annotations/human_pair_reviews.jsonl"

RELATIONS = (
    "interchangeable",
    "contains",
    "different_capability",
)
DIRECTIONS = ("none", "left_contains_right", "right_contains_left")


def canonical_relation(value: str) -> str:
    """Project historical labels onto the current three-class taxonomy."""
    if value == "exact_duplicate":
        return "interchangeable"
    if value == "uncertain":
        return "different_capability"
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def load_static_data() -> tuple[
    dict[str, dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]]
]:
    catalog = {row["api_id"]: row for row in load_jsonl(CATALOG_PATH)}
    pairs = {
        row["pair_id"]: row
        for path in (PILOT_PAIRS_PATH, AUDIT_PAIRS_PATH)
        for row in load_jsonl(path)
    }
    annotations = load_jsonl(ANNOTATIONS_PATH)
    annotations.sort(key=lambda record: (
        0 if (pairs.get(record["pair_id"], {}).get("audit_sampling") or {}).get("target") else 1,
        0 if record["annotation"].get("needs_human_review") else 1,
        record["pair_id"],
    ))
    return catalog, pairs, annotations


def latest_reviews() -> tuple[dict[str, dict[str, Any]], int]:
    events = load_jsonl(REVIEWS_PATH)
    latest = {}
    for event in events:
        latest[event["pair_id"]] = event
    return latest, len(events)


def save_review(review: dict[str, Any]) -> None:
    REVIEWS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REVIEWS_PATH.open("a", encoding="utf-8", newline="\n") as output:
        output.write(json.dumps(review, ensure_ascii=False, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def parameters(api: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for required, key in ((True, "required_parameters"), (False, "optional_parameters")):
        for item in api.get(key) or []:
            row = dict(item)
            row["required"] = required
            result.append(row)
    return result


def parameters_tooltip(api: dict[str, Any]) -> str:
    rows = parameters(api)
    if not rows:
        return "Параметры отсутствуют"
    lines = []
    for item in rows:
        requirement = "required" if item["required"] else "optional"
        name = item.get("name") or "unnamed"
        parameter_type = item.get("type") or "unknown"
        description = str(item.get("description") or "").strip()
        default = item.get("default")
        details = f"{requirement}: {name} ({parameter_type})"
        if default not in (None, ""):
            details += f", default={default}"
        if description:
            details += f" — {description}"
        lines.append(details)
    return "\n".join(lines)


def tooltip_attribute(value: Any) -> str:
    """Escape arbitrary documentation for a single HTML title attribute."""
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    return html.escape(text, quote=True).replace("\n", "&#10;")


def api_panel(title: str, api: dict[str, Any]) -> None:
    st.markdown(f'<div class="api-side-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f"**{api.get('api_name') or 'Unnamed API'}**")
    tool_label = html.escape(f"{api.get('tool_name', '')} · {api.get('category', '')}")
    tool_description = tooltip_attribute(api.get("tool_description") or "Описание инструмента отсутствует")
    api_id = html.escape(api["api_id"])
    parameter_help = tooltip_attribute(parameters_tooltip(api))
    st.markdown(
        f'<div class="hover-field tool-label" title="{tool_description}">{tool_label}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="hover-field api-id" title="{parameter_help}"><code>{api_id}</code></div>',
        unsafe_allow_html=True,
    )
    st.markdown(api.get("api_description") or "_Описание отсутствует._")


def select_index(options: tuple[str, ...], value: str | None, fallback: str) -> int:
    selected = value if value in options else fallback
    return options.index(selected)


def main() -> None:
    st.set_page_config(page_title="Проверка пар API", page_icon="🔎", layout="wide")
    st.markdown(
        """
        <style>
        .main .block-container {padding-top: 0.7rem; padding-bottom: 5rem;}
        .review-title {font-size: 1.35rem; font-weight: 650; margin: 0 0 0.35rem 0;}
        .api-side-title {font-size: 0.82rem; font-weight: 700; letter-spacing: 0.08em; color: #777; margin-bottom: 0.15rem;}
        .st-key-sticky_footer {position: fixed !important; left: 21.5rem !important; right: auto !important; width: calc(100vw - 22.4rem) !important; max-width: none !important; box-sizing: border-box !important; bottom: .45rem; z-index: 999; background: rgba(25, 28, 34, .96); color: #eee; backdrop-filter: blur(9px); border: 1px solid rgba(255,255,255,.16); border-radius: .55rem; padding: .32rem .55rem; box-shadow: 0 3px 16px rgba(0,0,0,.28); overflow: hidden;}
        .st-key-sticky_footer > div {width: 100% !important; max-width: 100% !important;}
        .footer-meta {font-size: .7rem; color: #ddd; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; padding-top: .38rem;}
        .footer-meta code {font-size: .68rem;}
        .hover-field {width: fit-content; max-width: 100%; cursor: help; border-bottom: 1px dotted rgba(128,128,128,.65); margin-bottom: .28rem;}
        .tool-label {font-size: .76rem; color: #777;}
        .api-id {font-size: .76rem; overflow-wrap: anywhere;}
        [data-testid="stMarkdownContainer"] p, [data-testid="stCaptionContainer"] {font-size: .84rem;}
        [data-testid="stMetricLabel"] p {font-size: .72rem;}
        [data-testid="stMetricValue"] {font-size: 1rem;}
        [data-testid="stWidgetLabel"] p {font-size: .78rem;}
        div[data-testid="stButton"] button, div[data-testid="stFormSubmitButton"] button {min-height: 2rem; padding: .25rem .55rem;}
        div[data-testid="stButton"] button p, div[data-testid="stFormSubmitButton"] button p {font-size: .78rem;}
        div[data-testid="stSegmentedControl"] button {min-height: 1.9rem; padding: .2rem .45rem;}
        div[data-testid="stSegmentedControl"] button p {font-size: .72rem;}
        h2 {font-size: 1.05rem !important; margin-top: .5rem !important;}
        h3 {font-size: .92rem !important; margin-top: .35rem !important;}
        @media (max-width: 900px) {.st-key-sticky_footer {left: .5rem !important; width: calc(100vw - 1rem) !important;}}
        </style>
        <div class="review-title">Ручная проверка отношений между API</div>
        """,
        unsafe_allow_html=True,
    )

    catalog, pairs, annotations = load_static_data()
    reviews, event_count = latest_reviews()
    if not annotations:
        st.error(f"Не найдены аннотации: {ANNOTATIONS_PATH}")
        st.stop()

    with st.sidebar:
        st.header("Фильтры")
        if "reviewer_name" not in st.session_state:
            st.session_state.reviewer_name = next(
                (review.get("reviewer", "") for review in reversed(list(reviews.values())) if review.get("reviewer")),
                "",
            )
        st.text_input(
            "Проверяющий",
            key="reviewer_name",
            help="Задаётся один раз и сохраняется при переходе между парами.",
        )
        status = st.radio("Статус", ("Непроверенные", "Проверенные", "Все"), horizontal=True)
        versions = sorted({row["provenance"]["prompt_version"] for row in annotations})
        selected_versions = st.multiselect("Версия prompt", versions, default=versions)
        llm_relations = sorted({canonical_relation(row["annotation"]["relation"]) for row in annotations})
        selected_relations = st.multiselect("Класс от LLM", llm_relations, default=llm_relations)
        only_flagged = st.checkbox("Только needs_human_review")
        only_audit = st.checkbox(
            "Только целевой аудит (20 пар)",
            help="Специально отобранные сложные кандидаты: предполагаемые дубликаты, взаимозаменяемость, включение и пограничные случаи. audit_target — эвристика, а не правильная метка.",
        )
        query = st.text_input("Поиск по названию или ID").strip().lower()

    filtered = []
    for row in annotations:
        pair_id = row["pair_id"]
        reviewed = pair_id in reviews
        pair = pairs.get(pair_id, {})
        left = catalog[row["left_api_id"]]
        right = catalog[row["right_api_id"]]
        if status == "Непроверенные" and reviewed:
            continue
        if status == "Проверенные" and not reviewed:
            continue
        if row["provenance"]["prompt_version"] not in selected_versions:
            continue
        if canonical_relation(row["annotation"]["relation"]) not in selected_relations:
            continue
        if only_flagged and not row["annotation"].get("needs_human_review"):
            continue
        if only_audit and not (pair.get("audit_sampling") or {}).get("target"):
            continue
        haystack = " ".join(str(value) for value in (
            pair_id, row["left_api_id"], row["right_api_id"], left.get("api_name"),
            right.get("api_name"), left.get("tool_name"), right.get("tool_name"),
        )).lower()
        if query and query not in haystack:
            continue
        filtered.append(row)

    with st.sidebar:
        st.divider()
        st.metric("Проверено", f"{len(reviews)} / {len(annotations)}")
        st.caption(f"Событий сохранения: {event_count}")
        st.caption(f"Под фильтром: {len(filtered)}")

    if not filtered:
        st.info("По выбранным фильтрам пар нет.")
        st.stop()

    if "review_position" not in st.session_state:
        st.session_state.review_position = 0
    st.session_state.review_position = min(st.session_state.review_position, len(filtered) - 1)

    record = filtered[st.session_state.review_position]
    pair_id = record["pair_id"]
    pair = pairs.get(pair_id, {})
    annotation = record["annotation"]
    model_relation = canonical_relation(annotation["relation"])
    previous = reviews.get(pair_id, {})
    left = catalog[record["left_api_id"]]
    right = catalog[record["right_api_id"]]

    left_column, right_column = st.columns(2)
    with left_column:
        api_panel("LEFT", left)
    with right_column:
        api_panel("RIGHT", right)

    st.divider()
    human_column, model_column = st.columns([1.08, 0.92], gap="large")
    with human_column:
        st.subheader("Ваше решение")
        with st.form(f"review-{pair_id}"):
            relation = st.segmented_control(
                "Правильное отношение",
                RELATIONS,
                default=(
                    previous.get("human_relation")
                    if previous.get("human_relation") in RELATIONS
                    else model_relation
                ),
                required=True,
                width="stretch",
                wrap=True,
            )
            direction = st.segmented_control(
                "Направление",
                DIRECTIONS,
                default=(
                    previous.get("human_direction")
                    if previous.get("human_direction") in DIRECTIONS
                    else annotation["direction"]
                ),
                required=True,
                width="stretch",
            )
            comment = st.text_area("Комментарий", value=previous.get("human_comment", ""), height=60)
            submitted = st.form_submit_button("Сохранить и перейти дальше", type="primary", width="stretch")

    with model_column:
        st.subheader("Решение модели")
        metric_row_one = st.columns(2)
        metric_row_one[0].metric("Отношение", model_relation)
        metric_row_one[1].metric("Направление", annotation["direction"])
        metric_row_two = st.columns(2)
        metric_row_two[0].metric("Уверенность", f"{annotation['confidence']:.2f}")
        metric_row_two[1].metric("Ручная проверка", "Да" if annotation["needs_human_review"] else "Нет")
        st.write(annotation["reasoning"])
        with st.expander("Структурированные детали"):
            st.json({
                "evidence": annotation["evidence"],
            })

    with st.container(key="sticky_footer"):
        footer_meta, footer_navigation = st.columns([1.65, 1], vertical_alignment="center")
        with footer_meta:
            st.markdown(
                f'<div class="footer-meta"><strong>Pair</strong> <code>{pair_id}</code> &nbsp;·&nbsp; '
                f'prompt <code>{record["provenance"]["prompt_version"]}</code> &nbsp;·&nbsp; '
                f'stratum <code>{(pair.get("sampling") or {}).get("stratum", "—")}</code> &nbsp;·&nbsp; '
                f'audit <code>{(pair.get("audit_sampling") or {}).get("target", "—")}</code></div>',
                unsafe_allow_html=True,
            )
        with footer_navigation:
            previous_column, position_column, next_column = st.columns([1, .55, 1], vertical_alignment="center")
            with previous_column:
                if st.button("← Предыдущая", width="stretch", disabled=st.session_state.review_position == 0):
                    st.session_state.review_position -= 1
                    st.rerun()
            with position_column:
                selected_number = st.number_input(
                    "Позиция",
                    min_value=1,
                    max_value=len(filtered),
                    value=st.session_state.review_position + 1,
                    step=1,
                    label_visibility="collapsed",
                )
                if selected_number - 1 != st.session_state.review_position:
                    st.session_state.review_position = selected_number - 1
                    st.rerun()
            with next_column:
                if st.button("Следующая →", width="stretch", disabled=st.session_state.review_position >= len(filtered) - 1):
                    st.session_state.review_position += 1
                    st.rerun()

    if submitted:
        if relation != "contains":
            direction = "none"
        relation_correct = (
            "yes"
            if relation == annotation["relation"] and direction == annotation["direction"]
            else "no"
        )
        save_review({
            "pair_id": pair_id,
            "human_relation": relation,
            "human_direction": direction,
            "human_relation_correct": relation_correct,
            "human_comment": comment.strip(),
            "reviewer": st.session_state.reviewer_name.strip(),
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
            "source_prompt_version": record["provenance"]["prompt_version"],
        })
        st.success("Решение сохранено.")
        if status == "Непроверенные":
            st.session_state.review_position = min(st.session_state.review_position, max(0, len(filtered) - 2))
        elif st.session_state.review_position < len(filtered) - 1:
            st.session_state.review_position += 1
        st.rerun()


if __name__ == "__main__":
    main()
