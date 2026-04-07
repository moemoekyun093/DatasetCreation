import json
import os
import requests
import time
import warnings
from urllib.parse import quote
from bs4 import BeautifulSoup
from tqdm import tqdm
from datasets import load_dataset, load_from_disk
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

warnings.filterwarnings("ignore")

# -----------------------------
# CONFIG
# -----------------------------
LOG_PATH = "table_scores.txt"
OUTPUT_PATH = "table_retrieval_dataset.json"

NUM_EXAMPLES = 50000
TIMEOUT = 5
MIN_TABLES = 3

device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# LOAD monoT5 RERANKER
# -----------------------------
print("Loading monoT5 reranker...")

model_name = "castorini/monot5-base-msmarco"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto"
)

model.eval()

# -----------------------------
# DATA
# -----------------------------
DATA_PATH = "hotpotqa_local"

if os.path.exists(DATA_PATH):
    dataset = load_from_disk(DATA_PATH)
else:
    dataset = load_dataset("hotpot_qa", "distractor", split="train")
    dataset.save_to_disk(DATA_PATH)

dataset = dataset.shuffle().select(range(NUM_EXAMPLES))

# -----------------------------
# NETWORK
# -----------------------------
page_cache = {}

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0"
})

# -----------------------------
# HELPERS
# -----------------------------
def get_supporting_pages(ex):
    return list(set(ex["supporting_facts"]["title"]))


def is_valid_table(table):
    classes = table.get("class", [])
    blocked = ["navbox", "sidebar", "metadata"] #infobox

    if any(any(b in c for b in blocked) for c in classes):
        return False

    rows = table.find_all("tr")
    cells = table.find_all(["td", "th"])

    return len(rows) >= 3 and len(cells) >= 6


def fetch_tables(page_title):
    if page_title in page_cache:
        return page_cache[page_title]

    try:
        url = f"https://en.wikipedia.org/wiki/{quote(page_title)}"
        res = session.get(url, timeout=TIMEOUT)
        soup = BeautifulSoup(res.text, "html.parser")

        tables = [t for t in soup.find_all("table") if is_valid_table(t)]
        page_cache[page_title] = tables
        return tables

    except:
        return []


def table_to_text(table):
    rows = []
    for tr in table.find_all("tr"):
        cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
        if cells:
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def table_to_pretty_text(table, max_col_width=40):
    rows = []

    for tr in table.find_all("tr"):
        cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
        if cells:
            rows.append(cells)

    if not rows:
        return ""

    max_cols = max(len(r) for r in rows)
    rows = [r + [""] * (max_cols - len(r)) for r in rows]

    def truncate(cell):
        return cell[:max_col_width] + ("…" if len(cell) > max_col_width else "")

    rows = [[truncate(cell) for cell in r] for r in rows]

    col_widths = [
        max(len(r[i]) for r in rows)
        for i in range(max_cols)
    ]

    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"

    lines = [sep]
    for r in rows:
        line = "| " + " | ".join(
            r[i].ljust(col_widths[i]) for i in range(max_cols)
        ) + " |"
        lines.append(line)
        lines.append(sep)

    return "\n".join(lines)


# -----------------------------
# monoT5 RERANKING FUNCTION
# -----------------------------
def rerank_score(question, table_text):
    # truncate table (important)
    table_text = table_text[:1000]

    prompt = f"Query: {question} Document: {table_text} Relevant:"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_scores=True
        )

    logits = outputs.scores[0]  # logits for first generated token

    true_id = tokenizer.convert_tokens_to_ids("true")

    score = logits[0][true_id].item()

    return score


# -----------------------------
# MAIN LOOP
# -----------------------------
output_data = []
log_file = open(LOG_PATH, "w", encoding="utf-8")

for ex_idx, ex in enumerate(tqdm(dataset)):
    question = ex["question"]
    answer = ex["answer"]
    pages = get_supporting_pages(ex)

    table_records = []

    for page in pages:
        tables = fetch_tables(page)

        for t in tables:
            txt = table_to_text(t)
            pretty = table_to_pretty_text(t)

            if len(txt) > 50:
                table_records.append({
                    "text": txt,
                    "pretty": pretty,
                    "page": page
                })

    if len(table_records) < MIN_TABLES:
        continue

    scored = []

    for record in table_records:
        s = rerank_score(question, record["text"])
        scored.append({
            "text": record["text"],
            "pretty": record["pretty"],
            "page": record["page"],
            "score": s
        })

    # sort by score
    scored.sort(key=lambda x: -x["score"])
    if scored[0]["score"] <= -20:
        continue

    # -----------------------------
    # LOGGING
    # -----------------------------
    log_file.write("\n" + "=" * 100 + "\n")
    log_file.write(f"EXAMPLE {ex_idx}\n")
    log_file.write("=" * 100 + "\n\n")

    log_file.write(f"QUESTION:\n{question}\n\n")
    log_file.write(f"ANSWER:\n{answer}\n\n")

    log_file.write("SUPPORTING FACT TITLES:\n")
    for title in pages:
        log_file.write(f"- {title}\n")
    log_file.write("\n")

    log_file.write("-" * 100 + "\n")
    log_file.write("TOP TABLES (monoT5 ranking)\n")
    log_file.write("-" * 100 + "\n\n")

    for i, item in enumerate(scored[:5]):
        log_file.write(f"[RANK {i+1}] SCORE = {item['score']:.4f}\n")
        log_file.write(f"PAGE = {item['page']}\n")
        log_file.write("-" * 60 + "\n")

        preview = item["pretty"][:10000]
        log_file.write(preview + "\n")

        if len(item["pretty"]) > 10000:
            log_file.write("... [TRUNCATED]\n")

        log_file.write("\n" + "-" * 100 + "\n\n")

    log_file.flush()

    # -----------------------------
    # DATASET OUTPUT
    # -----------------------------
    output_data.append({
        "question": question,
        "hard_positive": scored[0]["text"],
        "positive": scored[1]["text"] if len(scored) > 1 else scored[0]["text"],
        "negative": scored[-1]["text"],
        "scores": [item["score"] for item in scored[:5]]
    })

log_file.close()

# -----------------------------
# SAVE
# -----------------------------
with open(OUTPUT_PATH, "w") as f:
    json.dump(output_data, f, indent=2)

print(f"Saved {len(output_data)} examples")