import json
import os
import requests
from urllib.parse import quote
from bs4 import BeautifulSoup
from tqdm import tqdm
from datasets import load_dataset
from datasets import load_from_disk

from transformers import pipeline
import sys
sys.path.append('/kaggle/working/')

# -----------------------------
# CONFIG
# -----------------------------
OUTPUT_PATH = "table_retrieval_dataset.json"
DISPLAY_OUTPUT_PATH = "table_retrieval_tables_pretty.txt"
NUM_EXAMPLES = 10
TIMEOUT = 5

# -----------------------------
# LOAD LLM (JUDGE)
# -----------------------------
llm = pipeline(
    "text-generation",
    model="mistralai/Mistral-7B-Instruct-v0.2",
    device_map="auto"
)

# -----------------------------
# LOAD DATA
# -----------------------------
DATA_PATH = "hotpotqa_local"

if os.path.exists(DATA_PATH):
    print("Loading dataset from local disk...")
    dataset = load_from_disk(DATA_PATH)
else:
    print("Downloading dataset from HuggingFace...")
    dataset = load_dataset("hotpot_qa", "distractor", split="train")
    dataset.save_to_disk(DATA_PATH)
dataset = dataset.shuffle().select(range(min(NUM_EXAMPLES, len(dataset))))

# -----------------------------
# CACHE
# -----------------------------
page_cache = {}
session = requests.Session()
session.headers.update({"User-Agent": "table-retrieval"})

# -----------------------------
# HELPERS
# -----------------------------
def get_supporting_pages(ex):
    return list(set(ex["supporting_facts"]["title"]))


def fetch_tables(page_title):
    if page_title in page_cache:
        return page_cache[page_title]

    try:
        url = f"https://en.wikipedia.org/wiki/{quote(page_title)}"
        res = session.get(url, timeout=TIMEOUT)
        soup = BeautifulSoup(res.text, "html.parser")

        tables = soup.find_all("table", {"class": "wikitable"})
        page_cache[page_title] = tables
        return tables
    except:
        page_cache[page_title] = []
        return []


def table_to_text(table):
    rows = []
    for tr in table.find_all("tr"):
        cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
        if cells:
            rows.append(" | ".join(cells))
    return "\n".join(rows)


# -----------------------------
# 🔥 LLM JUDGE
# -----------------------------
def judge_table(question, answer, table_text):
    prompt = f"""
Question: {question}
Answer: {answer}

Table:
{table_text[:1200]}

Is this table useful for answering the question?

2 = enough information
1 = partially relevant
0 = not relevant

Output only 0, 1, or 2.
"""

    out = llm(prompt, max_new_tokens=5, do_sample=False)[0]["generated_text"]

    if "2" in out:
        return 2
    elif "1" in out:
        return 1
    else:
        return 0


# -----------------------------
# FORMAT TABLE
# -----------------------------
def pretty_format_table(table_text):
    rows = [line.split(" | ") for line in table_text.split("\n") if line.strip()]
    if not rows:
        return "(empty)"

    col_widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]

    lines = []
    for row in rows[:10]:  # limit rows for readability
        padded = [cell.ljust(col_widths[i]) for i, cell in enumerate(row)]
        lines.append(" | ".join(padded))

    return "\n".join(lines)


# -----------------------------
# MAIN
# -----------------------------
output_data = []
display_chunks = []

for ex in tqdm(dataset):
    question = ex["question"]
    answer = ex["answer"]
    pages = get_supporting_pages(ex)

    tables_text = []

    # collect tables
    for page in pages:
        tables = fetch_tables(page)

        for t in tables:
            txt = table_to_text(t)
            if len(txt) > 50:
                tables_text.append(txt)

    if not tables_text:
        continue

    labeled = []

    # 🔥 LLM labeling
    for table in tables_text:
        score = judge_table(question, answer, table)
        labeled.append((table, score))

    # sort by score
    labeled.sort(key=lambda x: -x[1])

    hard_positive = None
    positive = None
    negative = None

    for t, s in labeled:
        if s == 2 and hard_positive is None:
            hard_positive = t
        elif s == 1 and positive is None:
            positive = t
        elif s == 0 and negative is None:
            negative = t

    if hard_positive is None:
        continue

    if positive is None:
        positive = hard_positive
    if negative is None:
        negative = "No negative found"

    output_data.append({
        "question": question,
        "hard_positive": hard_positive,
        "positive": positive,
        "negative": negative
    })

    # pretty display
    display_chunks.append("=" * 80)
    display_chunks.append(f"Question: {question}")
    display_chunks.append("\n[Hard Positive]")
    display_chunks.append(pretty_format_table(hard_positive))
    display_chunks.append("\n[Positive]")
    display_chunks.append(pretty_format_table(positive))


# -----------------------------
# SAVE
# -----------------------------
with open(OUTPUT_PATH, "w") as f:
    json.dump(output_data, f, indent=2)

with open(DISPLAY_OUTPUT_PATH, "w") as f:
    f.write("\n".join(display_chunks))

print(f"\nSaved {len(output_data)} examples")