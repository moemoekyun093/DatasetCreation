import json
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import requests
import time
import warnings
from urllib.parse import quote
from bs4 import BeautifulSoup
from tqdm import tqdm
from datasets import load_dataset, load_from_disk
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers import logging as hf_logging

warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()

# -----------------------------
# CONFIG
# -----------------------------
OUTPUT_PATH = "table_retrieval_dataset.json"
NUM_EXAMPLES = 10
TIMEOUT = 5
MIN_TABLES = 3

# -----------------------------
# LOAD LLM
# -----------------------------
print("loading mistral...")
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=False)
tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    output_attentions=True,
    torch_dtype=dtype,
    local_files_only=False,
)
model.to(device)
model.eval()

llm = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

# Keep all generation settings in one place to avoid deprecation warnings.
if llm.tokenizer.pad_token_id is None:
    llm.tokenizer.pad_token = llm.tokenizer.eos_token

llm.model.config.pad_token_id = llm.tokenizer.pad_token_id
llm.model.generation_config.pad_token_id = llm.tokenizer.pad_token_id
llm.model.generation_config.do_sample = False
llm.model.generation_config.max_new_tokens = 10
llm.model.generation_config.max_length = None

# -----------------------------
# LOAD DATA
# -----------------------------
DATA_PATH = "hotpotqa_local"

if os.path.exists(DATA_PATH):
    print("Loading dataset from disk...")
    dataset = load_from_disk(DATA_PATH)
else:
    print("Downloading dataset...")
    dataset = load_dataset("hotpot_qa", "distractor", split="train[:1%]")
    dataset.save_to_disk(DATA_PATH)

dataset = dataset.shuffle().select(range(min(NUM_EXAMPLES, len(dataset))))

# -----------------------------
# CACHE + SESSION
# -----------------------------
page_cache = {}

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"
})

# -----------------------------
# HELPERS
# -----------------------------
import re

def extract_score(text):
    match = re.search(r"\b(10|[0-9])\b", text)
    return int(match.group(1)) if match else 0

def get_supporting_pages(ex):
    return list(set(ex["supporting_facts"]["title"]))


def is_valid_table(table):
    classes = table.get("class", [])

    blocked = [
        "infobox",
        "navbox",
        "vertical-navbox",
        "metadata",
        "sidebar",
        "ambox",
        "toc",
    ]

    if any(any(b in c for b in blocked) for c in classes):
        return False

    rows = table.find_all("tr")
    cells = table.find_all(["td", "th"])

    if len(rows) < 3 or len(cells) < 6:
        return False

    text = table.get_text(" ", strip=True).lower()

    bad_phrases = [
        "you can help wikipedia",
        "this article is a stub",
        "citation needed",
    ]

    if any(p in text for p in bad_phrases):
        return False

    return True


def fetch_tables(page_title):
    if page_title in page_cache:
        return page_cache[page_title]

    search_url = "https://en.wikipedia.org/w/api.php"
    search_params = {
        "action": "query",
        "list": "search",
        "srsearch": page_title,
        "format": "json"
    }

    for attempt in range(5):
        try:
            res = session.get(search_url, params=search_params, timeout=TIMEOUT)

            if res.status_code == 429:
                time.sleep(2 ** attempt)
                continue

            data = res.json()

            if not data["query"]["search"]:
                page_cache[page_title] = []
                return []

            actual_title = data["query"]["search"][0]["title"]

            url = f"https://en.wikipedia.org/wiki/{quote(actual_title)}"
            page_res = session.get(url, timeout=TIMEOUT)

            soup = BeautifulSoup(page_res.text, "html.parser")

            tables = []
            for t in soup.find_all("table"):
                if is_valid_table(t):
                    tables.append(t)

            page_cache[page_title] = tables
            return tables

        except Exception:
            time.sleep(2 ** attempt)

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
# LLM SCORING
# -----------------------------
def score_table(question, answer, table_text):
    prompt = f"""
Question: {question}
Answer: {answer}

Table:
{table_text[:1200]}

Score how useful this table is for answering the question.

0 = useless
10 = fully answers

Output ONLY a number between 0 and 10.
"""

    try:
        with torch.inference_mode():
            out = llm(prompt, return_full_text=False)[0]["generated_text"]
        return extract_score(out)
    except torch.OutOfMemoryError:
        print("CUDA OOM during inference. Returning score=0 for this table.")
        return 0
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# -----------------------------
# MAIN
# -----------------------------
output_data = []

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

    # filter weak examples
    if len(tables_text) < MIN_TABLES:
        continue

    scored = []
    print("starting scoring")
    for table in tables_text:
        print("\n\n" + "="*40)
        s = score_table(question, answer, table)
        scored.append((table, s))

    # sort by score
    scored.sort(key=lambda x: -x[1])

    hard_positive = scored[0][0]
    positive = scored[1][0] if len(scored) > 1 else hard_positive
    negative = scored[-1][0]

    output_data.append({
        "question": question,
        "hard_positive": hard_positive,
        "positive": positive,
        "negative": negative,
        "scores": [s for _, s in scored[:5]]
    })

# -----------------------------
# SAVE
# -----------------------------
with open(OUTPUT_PATH, "w") as f:
    json.dump(output_data, f, indent=2)

print(f"\nSaved {len(output_data)} examples to {OUTPUT_PATH}")