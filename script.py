import json
import os
import requests
import time
import warnings
import re
from urllib.parse import quote
from bs4 import BeautifulSoup
from tqdm import tqdm
from datasets import load_dataset, load_from_disk
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
LOG_PATH = "table_scores.txt"
PROMPT_PATH = "sample_prompts.txt"

warnings.filterwarnings("ignore")

# -----------------------------
# CONFIG
# -----------------------------
OUTPUT_PATH = "table_retrieval_dataset.json"
NUM_EXAMPLES = 10
TIMEOUT = 5
MIN_TABLES = 3

# -----------------------------
# LOAD SMALL LLM (1B)
# -----------------------------
print("loading TinyLlama (1B)...")

# model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model_name = "unsloth/gemma-2b"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
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

# dataset = dataset.shuffle().select(range(min(NUM_EXAMPLES, len(dataset))))
dataset = dataset.shuffle().select(range(len(dataset)))


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
def table_to_pretty_text(table, max_col_width=40):
    rows = []

    for tr in table.find_all("tr"):
        cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
        if cells:
            rows.append(cells)

    if not rows:
        return ""

    # normalize row lengths
    max_cols = max(len(r) for r in rows)
    rows = [r + [""] * (max_cols - len(r)) for r in rows]

    # truncate + compute column widths
    def truncate(cell):
        return cell[:max_col_width] + ("…" if len(cell) > max_col_width else "")

    rows = [[truncate(cell) for cell in r] for r in rows]

    col_widths = [
        max(len(r[i]) for r in rows)
        for i in range(max_cols)
    ]

    # build horizontal separator
    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"

    # build formatted table
    lines = [sep]
    for r in rows:
        line = "| " + " | ".join(
            r[i].ljust(col_widths[i]) for i in range(max_cols)
        ) + " |"
        lines.append(line)
        lines.append(sep)

    return "\n".join(lines)

def extract_score(text):
    match = re.search(r"\b(10|[0-9])\b", text)
    return int(match.group(1)) if match else 0


def get_supporting_pages(ex):
    return list(set(ex["supporting_facts"]["title"]))


def is_valid_table(table):
    classes = table.get("class", [])

    blocked = ["infobox","navbox", "sidebar", "metadata"] #"infobox"

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


# -----------------------------
# SAFE GENERATION
# -----------------------------
def generate(prompt):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024   # VERY IMPORTANT
    ).to(model.device)

    input_length = inputs["input_ids"].shape[1]

    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        generated_tokens = outputs[0][input_length:]
        text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return text.strip()

    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None


def score_table(question, answer, table_text):
    for max_chars in [800, 500, 300]:
        prompt = f"""
I’m going to give you a question and a table extracted
from a Wikipedia page. I want you to determine how useful the table
is for answering the question.

You should assign a score:
0 = not useful (no relevant information)
1 = highly useful (helps answer the question or helps bridge entities)

You should output ONLY the score as a single number.

Let’s go through some examples together.

Question: Who won the 1998 FIFA World Cup?

Table:
Year | Winner
1998 | France

Score:
1

Question: Who won the 1998 FIFA World Cup?

Table:
Country | Population
France | 67 million

Score:
0

Question: The 1976 German Grand Prix was won by a driver who retired in what year?

Table:
Pos | Driver | Constructor
1 | James Hunt | McLaren-Ford
2 | Niki Lauda | Ferrari

Score:
1

Question: The 1976 German Grand Prix was won by a driver who retired in what year?

Table:
Drivers' Championship standings
1 | Niki Lauda | 58
2 | James Hunt | 44

Score:
0

Once you have determined the score, output the score and stop.

Question: {question}

Table:
{table_text[:max_chars]}

Score:
"""

        prompt_file.write("\n" + "=" * 100 + "\n")
        prompt_file.write(f"PROMPT (table_excerpt={max_chars})\n")
        prompt_file.write("=" * 100 + "\n\n")
        prompt_file.write(prompt)
        prompt_file.write("\n" + "=" * 100 + "\n")
        prompt_file.flush()

        out = generate(prompt)

        if out:
            return extract_score(out)

    return 0


# -----------------------------
# MAIN LOOP
# -----------------------------
output_data = []
log_file = open(LOG_PATH, "w", encoding="utf-8")
prompt_file = open(PROMPT_PATH, "w", encoding="utf-8")

# for ex_idx, ex in enumerate(tqdm(dataset)):
for ex_idx, ex in enumerate(tqdm(dataset)):
    question = ex["question"]
    answer = ex["answer"]
    pages = get_supporting_pages(ex)

    table_records = []

    for page in pages:
        tables = fetch_tables(page)

        for t in tables:
            txt = table_to_text(t)
            pretty_txt = table_to_pretty_text(t)
            if len(txt) > 50:
                table_records.append({
                    "text": txt,
                    "pretty": pretty_txt,
                    "page": page
                })

    if len(table_records) < MIN_TABLES:
        continue

    scored = []

    for record in table_records:
        s = score_table(question, answer, record["text"])
        scored.append({
            "text": record["text"],
            "pretty": record["pretty"],
            "page": record["page"],
            "score": s
        })

    # sort
    scored.sort(key=lambda x: -x["score"])


    # -----------------------------
    # ✨ WRITE TO TXT FILE
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
    log_file.write("TOP TABLES (with scores)\n")
    log_file.write("-" * 100 + "\n\n")

    # show top 5 tables
    for i, item in enumerate(scored[:5]):
        log_file.write(f"[RANK {i+1}] SCORE = {item['score']}\n")
        log_file.write(f"PAGE = {item['page']}\n")
        log_file.write("-" * 60 + "\n")

        # truncate for readability
        preview = item["pretty"][:10000]
        log_file.write(preview + "\n")

        if len(item["pretty"]) > 10000:
            log_file.write("... [TRUNCATED]\n")

        log_file.write("\n" + "-" * 100 + "\n\n")

    log_file.flush()

    # -----------------------------
    # dataset output (unchanged)
    # -----------------------------
    output_data.append({
        "question": question,
        "hard_positive": scored[0]["text"],
        "positive": scored[1]["text"] if len(scored) > 1 else scored[0]["text"],
        "negative": scored[-1]["text"],
        "scores": [item["score"] for item in scored[:5]]
    })
log_file.close()
prompt_file.close()
# -----------------------------
# SAVE
# -----------------------------
with open(OUTPUT_PATH, "w") as f:
    json.dump(output_data, f, indent=2)

print(f"Saved {len(output_data)} examples")