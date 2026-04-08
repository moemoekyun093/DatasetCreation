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

NUM_EXAMPLES = 500
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

        prompts = self.tokenizer.batch_encode_plus(
            ['Relevant:' for _ in range(self.batch_size)],
            return_tensors='pt',
            padding='longest'
        )

        max_vlen = self.model.config.n_positions - prompts['input_ids'].shape[1]

        for start_idx in it:
            rng = slice(start_idx, start_idx+self.batch_size)

            enc = self.tokenizer.batch_encode_plus(
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
    filtered = [item for item in scored if item["score"] >= -30]

    log_file.write("\n" + "=" * 100 + "\n")
    log_file.write(f"QUESTION:\n{question}\n\n")

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