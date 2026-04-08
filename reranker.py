import json
import os
import requests
import warnings
from urllib.parse import quote
from bs4 import BeautifulSoup
from tqdm import tqdm
from datasets import load_dataset, load_from_disk
import torch
import torch.nn.functional as F
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration

warnings.filterwarnings("ignore")

# -----------------------------
# CONFIG
# -----------------------------
LOG_PATH = "table_scores.txt"
OUTPUT_PATH = "table_retrieval_dataset.json"
RANKING_TXT_PATH = "table_rankings.txt"
PROMPT_PATH = "prompts.txt"

NUM_EXAMPLES = 50
TIMEOUT = 5
MIN_TABLES = 3

# -----------------------------
# MONOT5 CLASS (UNCHANGED LOGIC)
# -----------------------------
class MonoT5:
    def __init__(self, 
                 tok_model='t5-base',
                 model='castorini/monot5-base-msmarco',
                 batch_size=8,
                 text_field='text',
                 verbose=False):

        self.verbose = verbose
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.tokenizer = T5Tokenizer.from_pretrained(tok_model)
        self.model_name = model
        self.model = T5ForConditionalGeneration.from_pretrained(model)

        self.model.to(self.device)
        self.model.eval()

        self.text_field = text_field

        self.REL = self.tokenizer.encode('true')[0]
        self.NREL = self.tokenizer.encode('false')[0]

    def transform(self, run: pd.DataFrame):
        scores = []
        prob = []

        queries, texts = run['query'], run[self.text_field]

        it = range(0, len(queries), self.batch_size)

        prompts = self.tokenizer(
            ['Relevant:' for _ in range(self.batch_size)],
            return_tensors='pt',
            padding='longest'
        )

        max_vlen = self.model.config.n_positions - prompts['input_ids'].shape[1]

        for start_idx in it:
            rng = slice(start_idx, start_idx+self.batch_size)

            enc = self.tokenizer(
                [f'Query: {q} Document: {d}' for q, d in zip(queries[rng], texts[rng])],
                return_tensors='pt',
                padding='longest',
                truncation=True
            )

            for key, enc_value in list(enc.items()):
                enc_value = enc_value[:, :-1]
                enc_value = enc_value[:, :max_vlen]

                enc[key] = torch.cat(
                    [enc_value, prompts[key][:enc_value.shape[0]]],
                    dim=1
                )

            enc['decoder_input_ids'] = torch.full(
                (len(queries[rng]), 1),
                self.model.config.decoder_start_token_id,
                dtype=torch.long
            )

            enc = {k: v.to(self.device) for k, v in enc.items()}

            with torch.no_grad():
                result = self.model(**enc).logits

            result = result[:, 0, (self.REL, self.NREL)]

            log_probs = F.log_softmax(result, dim=1)

            scores_batch = log_probs[:, 0].cpu().tolist()
            probs_batch = torch.exp(log_probs[:, 0]).cpu().tolist()

            scores.extend(scores_batch)
            prob.extend(probs_batch)

        run = run.assign(score=scores, prob=prob)
        return run


# -----------------------------
# LOAD MODEL
# -----------------------------
reranker = MonoT5(batch_size=8)

# -----------------------------
# DATA
# -----------------------------
# DATA_PATH = "hotpotqa_local"

# if os.path.exists(DATA_PATH):
#     dataset = load_from_disk(DATA_PATH)
# else:
#     dataset = load_dataset("hotpot_qa", "distractor", split="train")
#     dataset.save_to_disk(DATA_PATH)

# dataset = dataset.shuffle().select(range(NUM_EXAMPLES))
DATA_PATH = "hotpotqa_fullwiki_medium_bridge"

if os.path.exists(DATA_PATH):
    dataset = load_from_disk(DATA_PATH)
else:
    dataset = load_dataset("hotpot_qa", "fullwiki", split="train")

    # -----------------------------
    # FILTER: medium + bridge
    # -----------------------------
    dataset = dataset.filter(
        lambda x: x["level"] == "medium" and x["type"] == "bridge"
    )

    dataset.save_to_disk(DATA_PATH)

# -----------------------------
# NO SHUFFLING
# -----------------------------
dataset = dataset.select(range(min(NUM_EXAMPLES, len(dataset))))

# -----------------------------
# NETWORK
# -----------------------------
page_cache = {}

session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0"})

# -----------------------------
# HELPERS
# -----------------------------
def get_supporting_pages(ex):
    return list(set(ex["supporting_facts"]["title"]))


def is_valid_table(table):
    classes = table.get("class", [])
    blocked = ["navbox", "sidebar", "metadata"]

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


import re
import json
import re

def clean_text(text):
    # remove references like [1], [23]
    text = re.sub(r'\[\d+\]', '', text)

    # normalize spaces
    text = text.replace('\xa0', ' ')
    text = text.replace('·', ' ')

    # fix parentheses spacing
    text = re.sub(r'(?<!\s)\(', ' (', text)
    text = re.sub(r'\)(?!\s)', ') ', text)

    # split camel-case words (basic fix)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

    # collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

def table_to_rows(table):
    grid = []
    rowspan_map = {}

    for tr in table.find_all("tr"):
        cells = tr.find_all(["td", "th"])
        row = []
        col_idx = 0

        # fill from previous rowspans
        while col_idx in rowspan_map:
            row.append(rowspan_map[col_idx]["value"])
            rowspan_map[col_idx]["rows_left"] -= 1
            if rowspan_map[col_idx]["rows_left"] == 0:
                del rowspan_map[col_idx]
            col_idx += 1

        for cell in cells:
            text = clean_text(cell.get_text(" ", strip=True))

            rowspan = int(cell.get("rowspan", 1))
            colspan = int(cell.get("colspan", 1))

            for _ in range(colspan):
                row.append(text)

                if rowspan > 1:
                    rowspan_map[col_idx] = {
                        "value": text,
                        "rows_left": rowspan - 1
                    }

                col_idx += 1

        if any(row):
            grid.append(row)

    return grid

def rows_to_sentences(rows):
    sentences = []

    header = rows[0]
    data_rows = rows[1:]

    for r in data_rows:
        if len(r) != len(header):
            continue

        parts = []
        for i in range(len(header)):
            key = header[i].strip()
            val = r[i].strip()

            if key and val:
                parts.append(f"{key} is {val}")

        if parts:
            sentences.append(". ".join(parts) + ".")

    return " ".join(sentences)

def table_to_text(table):
    rows = table_to_rows(table)

    if not rows or len(rows) < 2:
        return ""

    header = rows[0]
    data_rows = rows[1:]

    structured = []

    # -----------------------------
    # CASE 1: VALID HEADER TABLE
    # -----------------------------
    if len(header) > 1 and all(h.strip() for h in header):
        for r in data_rows:
            if len(r) != len(header):
                continue

            row_dict = {
                header[i]: r[i]
                for i in range(len(header))
                if r[i].strip()
            }

            if row_dict:
                structured.append(row_dict)

    # -----------------------------
    # CASE 2: NO HEADER (fallback)
    # -----------------------------
    else:
        for r in rows:
            if len(r) == 2:
                structured.append({r[0]: r[1]})
            else:
                structured.append(
                    {f"column_{i}": r[i] for i in range(len(r))}
                )

    if not structured:
        return ""

    # -----------------------------
    # JSON PART
    # -----------------------------
    json_part = json.dumps(structured, ensure_ascii=False)

    # -----------------------------
    # NATURAL LANGUAGE PART
    # -----------------------------
    if len(header) > 1 and all(h.strip() for h in header):
        text_part = rows_to_sentences(rows)
    else:
        # fallback sentence generation
        sentences = []
        for row in structured:
            parts = [f"{k} is {v}" for k, v in row.items()]
            sentences.append(". ".join(parts) + ".")
        text_part = " ".join(sentences)

    # -----------------------------
    # FINAL REPRESENTATION
    # -----------------------------
    # return f"Table:\n{json_part}\nSummary:\n{text_part}"
    return f"Table:\n{json_part}"


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

    col_widths = [max(len(r[i]) for r in rows) for i in range(max_cols)]

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
# MAIN LOOP
# -----------------------------
output_data = []
log_file = open(LOG_PATH, "w", encoding="utf-8")
ranking_file = open(RANKING_TXT_PATH, "w", encoding="utf-8")
prompt_file = open(PROMPT_PATH, "w", encoding="utf-8")


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

    # -----------------------------
    # BUILD DATAFRAME FOR MONOT5
    # -----------------------------
    df = pd.DataFrame({
        "query": [question] * len(table_records),
        "text": [r["text"] for r in table_records]
    })


    prompt_file.write("\n" + "=" * 100 + "\n")
    prompt_file.write(f"EXAMPLE {ex_idx}\n")
    prompt_file.write("=" * 100 + "\n\n")

    for i, r in enumerate(table_records):
        prompt = f"Query: {question} Document: {r['text']} Relevant:"
        
        prompt_file.write(f"[TABLE {i+1}]\n")
        prompt_file.write(prompt[:2000] + "\n\n")  # truncate to avoid huge file

    prompt_file.flush()

    df = reranker.transform(df)

    # attach back
    for i, row in df.iterrows():
        table_records[i]["score"] = row["score"]

    scored = sorted(table_records, key=lambda x: -x["score"])

    if scored[0]["score"] <= -20:
        continue

    # -----------------------------
    # RANKING FILE
    # -----------------------------
    ranking_file.write("\n" + "=" * 100 + "\n")
    ranking_file.write(f"EXAMPLE {ex_idx}\n\n")
    ranking_file.write(f"QUESTION:\n{question}\n\n")

    for i, item in enumerate(scored):
        ranking_file.write(
            f"[RANK {i+1}] SCORE = {item['score']:.4f} PAGE = {item['page']}\n"
        )

    ranking_file.flush()

    # -----------------------------
    # LOG FILE
    # -----------------------------
    filtered = [item for item in scored if item["score"] >= -5]

    log_file.write("\n" + "=" * 100 + "\n")
    log_file.write(f"QUESTION:\n{question}\n\n")
    log_file.write(f"ANSWER:\n{answer}\n\n")


    for i, item in enumerate(filtered):
        log_file.write(f"[RANK {i+1}] SCORE = {item['score']:.4f}\n")
        log_file.write(f"PAGE = {item['page']}\n")
        log_file.write(item["pretty"][:10000] + "\n\n")

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
ranking_file.close()

with open(OUTPUT_PATH, "w") as f:
    json.dump(output_data, f, indent=2)

print(f"Saved {len(output_data)} examples")