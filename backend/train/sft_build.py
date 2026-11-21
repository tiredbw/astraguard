"""Сборка обучающего набора для дообучения детектора (Qwen 3.5) через mlx-lm.

Шаги (каждый можно перезапускать, результаты кэшируются в data/sft/):
  select   — отбор примеров: dev (кроме отложенных групп) + внешние наборы из data/external/;
  messages — тексты сообщений для модели тем же кодом, что в сервисе (cargo example dump_requests);
  teacher  — ответы текущей модели qwen3.5:4b через Ollama: из них берём tool/action/forbidden_action/secrets;
  compose  — train.jsonl / valid.jsonl в формате чата; category и risk_level — из разметки, не от учителя.

Отложенные группы dev (group_id % 5 == 0) в обучение не попадают и пишутся в data/sft/dev_heldout.jsonl
для честного замера через evaluate.py. Наборы eval/own_* не используются вовсе.

Запуск из корня репозитория:
  python3 -I backend/train/sft_build.py select --dev ../astra_public_v2/dev.jsonl
  python3 -I backend/train/sft_build.py messages
  python3 -I backend/train/sft_build.py teacher
  python3 -I backend/train/sft_build.py compose
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXT = ROOT / "data" / "external"
OUT = ROOT / "data" / "sft"
SEED = 42
MAX_USER_CHARS = 2500  # Длинные примеры не влезают в память при обучении на 16 ГБ.

# Сколько строк брать из каждого внешнего набора: (атаки, легитимные).
EXTERNAL = {
    "dmtrdr.core": (400, 0),
    "deepset": (100, 100),
    "xtram1": (150, 150),
    "xstest": (150, 150),
    "orbench": (150, 300),
    "injecagent": (250, 17),
}

REASONS = {
    "none": "обычный рабочий запрос",
    "sensitive_action": "изменяющее действие инструментом",
    "jailbreak": "попытка снять ограничения",
    "prompt_injection": "попытка подменить инструкции",
    "indirect_injection": "инструкция агенту во вложении",
    "system_prompt_extraction": "попытка узнать системные инструкции",
    "system_prompt_override": "попытка переписать системные правила",
    "tool_abuse": "опасное использование инструмента",
    "privilege_escalation": "попытка получить чужие права",
    "unauthorized_data_access": "доступ к чужим данным",
    "data_exfiltration": "вывод данных наружу",
    "harmful_content": "запрос вредного контента",
    "fraud_social_engineering": "мошенничество или социальная инженерия",
    "abuse": "оскорбительный или злоупотребляющий запрос",
}


def read_jsonl(path: Path) -> list[dict]:
    # Только \n: splitlines() режет и по U+2028 и прочим разделителям внутри текстов.
    return [json.loads(s) for s in path.read_text(encoding="utf-8-sig").split("\n") if s.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def cmd_select(args) -> None:
    rnd = random.Random(SEED)
    groups = json.loads((ROOT / "eval" / "dev_groups.json").read_text(encoding="utf-8"))["groups"]
    dev = read_jsonl(args.dev)
    train_dev, heldout = [], []
    for r in dev:
        (heldout if groups[r["id"]] % 5 == 0 else train_dev).append(r)
    write_jsonl(OUT / "dev_heldout.jsonl", heldout)
    selected = [dict(r, sft_source="dev") for r in train_dev]
    for name, (n_attack, n_benign) in EXTERNAL.items():
        rows = read_jsonl(EXT / f"{name}.jsonl")
        attacks = [r for r in rows if r["is_dangerous"]]
        benign = [r for r in rows if not r["is_dangerous"]]
        if name == "dmtrdr.core":  # Поровну из каждого источника, чтобы Mosscap не забил всё.
            by_origin = defaultdict(list)
            for r in attacks:
                by_origin[next(t for t in r["tags"] if t.startswith("origin:"))].append(r)
            per = n_attack // len(by_origin)
            attacks = [x for v in by_origin.values() for x in rnd.sample(v, min(per, len(v)))]
            n_attack = len(attacks)
        selected += [dict(r, sft_source=name) for r in rnd.sample(attacks, min(n_attack, len(attacks)))]
        selected += [dict(r, sft_source=name) for r in rnd.sample(benign, min(n_benign, len(benign)))]
    write_jsonl(OUT / "selected.jsonl", selected)
    print(f"dev: обучение {len(train_dev)}, отложено {len(heldout)} (data/sft/dev_heldout.jsonl)")
    print("отобрано:", Counter(r["sft_source"] for r in selected), "всего", len(selected))


def cmd_messages(args) -> None:
    out = subprocess.run(
        ["cargo", "run", "--release", "--quiet", "--example", "dump_requests", "--", str(OUT / "selected.jsonl")],
        cwd=ROOT / "backend", capture_output=True, text=True, check=True).stdout
    bodies = [json.loads(s) for s in out.split("\n") if s.strip()]
    write_jsonl(OUT / "requests.jsonl", bodies)
    print("сообщений:", len(bodies))


def cmd_teacher(args) -> None:
    bodies = read_jsonl(OUT / "requests.jsonl")
    cache_path = OUT / "teacher.jsonl"
    done = {r["id"]: r for r in read_jsonl(cache_path)} if cache_path.exists() else {}
    todo = [b for b in bodies if b["id"] not in done
            and len(b["body"]["messages"][1]["content"]) <= MAX_USER_CHARS]
    print(f"готово {len(done)}, осталось {len(todo)}", flush=True)
    t0 = time.time()
    with cache_path.open("a", encoding="utf-8") as fh:
        for i, b in enumerate(todo, 1):
            body = dict(b["body"], model=args.model)
            req = urllib.request.Request(args.ollama + "/api/chat", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    content = json.load(resp)["message"]["content"]
                out = json.loads(content)
            except Exception as e:  # Ошибку записываем, пример потом пропустим.
                out = {"error": str(e)[:200]}
            fh.write(json.dumps({"id": b["id"], "out": out}, ensure_ascii=False) + "\n")
            fh.flush()
            if i % 25 == 0 or i == len(todo):
                rate = (time.time() - t0) / i
                print(f"{i}/{len(todo)}  {rate:.1f} с/пример, осталось ~{rate * (len(todo) - i) / 60:.0f} мин", flush=True)


def cmd_compose(args) -> None:
    rnd = random.Random(SEED)
    selected = {r["id"]: r for r in read_jsonl(OUT / "selected.jsonl")}
    bodies = {b["id"]: b["body"] for b in read_jsonl(OUT / "requests.jsonl")}
    teacher = {t["id"]: t["out"] for t in read_jsonl(OUT / "teacher.jsonl")}
    examples, stats = [], Counter()
    for rid, out in teacher.items():
        if "error" in out:
            stats["ошибка учителя"] += 1
            continue
        row, body = selected[rid], bodies[rid]
        gold_cat, gold_level = row["risk_category"], row["risk_level"]
        target = {
            "category": gold_cat,
            "risk_level": gold_level,
            "tool": out.get("tool"),
            "forbidden_action": out.get("forbidden_action"),
            "action": out.get("action", "execute"),
            "secrets": out.get("secrets", "none"),
            "reason": out.get("reason") or REASONS[gold_cat],
        }
        if out.get("category") != gold_cat:  # Учитель ошибся в категории — его пояснение не годится.
            target["reason"] = REASONS[gold_cat]
            stats["категория исправлена"] += 1
        if gold_cat == "none":  # У легитимного запроса не бывает запрещённых действий и значений секретов.
            target["forbidden_action"] = None
            if target["secrets"] == "values":
                target["secrets"] = "none"
        assistant = json.dumps(target, ensure_ascii=False, separators=(", ", ": "))
        examples.append({"messages": body["messages"] + [{"role": "assistant", "content": assistant}],
                         "_source": row["sft_source"], "_category": gold_cat})
    rnd.shuffle(examples)
    n_valid = max(50, len(examples) // 20)
    valid, train = examples[:n_valid], examples[n_valid:]
    strip = lambda rows: [{"messages": r["messages"]} for r in rows]
    write_jsonl(OUT / "train.jsonl", strip(train))
    write_jsonl(OUT / "valid.jsonl", strip(valid))
    print(f"train {len(train)}, valid {len(valid)}", dict(stats))
    print("источники:", Counter(r["_source"] for r in examples))
    print("категории:", Counter(r["_category"] for r in examples).most_common())


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--dev", type=Path, required=True)
    sub.add_parser("messages")
    t = sub.add_parser("teacher")
    t.add_argument("--ollama", default="http://127.0.0.1:11434")
    t.add_argument("--model", default="qwen3.5:4b")
    sub.add_parser("compose")
    args = ap.parse_args()
    {"select": cmd_select, "messages": cmd_messages, "teacher": cmd_teacher, "compose": cmd_compose}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
