# =============================================================================
# Psycho, Rewritten — Colab T4 grader (Gradio)
#
# Ask the transformed Mistral-7B LoRA adapter about the rewritten *Psycho* world,
# tune generation, grade the answer, and log everything to Supabase `psycho_qa`.
#
# What actually changes the answer (this is the part that usually silently breaks):
#   • Temperature  -> 0.0 = greedy/deterministic (do_sample=False, temp is IGNORED
#                     by HF unless you sample). > 0.0 flips do_sample=True so the
#                     slider genuinely samples. Re-run at temp>0 = different output.
#   • RAG toggle   -> ON injects the top retrieved facts into the prompt; OFF sends
#                     the bare question. Different prompt => different answer.
#   • Max tokens   -> hard cap on generation length.
# The exact prompt sent and the phrases retrieved are shown in the UI so you can
# see the effect, not just take it on faith.
#
# Off-topic questions (not about the transformed dataset) are REFUSED before the
# model is ever called.
#
# ---------------------------------------------------------------------------
# HOW TO RUN IN COLAB (Runtime -> Change runtime type -> T4 GPU):
#
#   CELL 1  (install):
#     !pip install -q -U transformers peft accelerate bitsandbytes \
#                        gradio supabase sentence-transformers datasets huggingface_hub
#
#   CELL 2  (secrets): add these in Colab's 🔑 Secrets panel (left sidebar), then
#                      toggle "Notebook access" on for each:
#     SUPABASE_URL, SUPABASE_KEY, and HF_TOKEN (a HF token that has accepted the
#     gated Mistral-7B-Instruct-v0.3 licence).
#
#   CELL 3+ : paste everything below this banner and run. Gradio prints a public
#             https://<id>.gradio.live link and also embeds inline.
# =============================================================================

import os
import torch
import numpy as np
import gradio as gr

# ------------------------------------------------------------------ ids / config
BASE     = "mistralai/Mistral-7B-Instruct-v0.3"                       # gated
ADAPTER  = "antfr99/psycho-mistral-v03-transformed-adapter"
DATASET  = "antfr99/hitchcock-psycho-1960-film-dataset-transformed"
EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"                   # tiny, fast
TABLE    = "psycho_qa"        # change to your dedicated transformed table if split
TOP_K    = 4                  # phrases retrieved per question
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------------ secrets
def _secret(name):
    try:
        from google.colab import userdata
        v = userdata.get(name)
        if v:
            return v
    except Exception:
        pass
    return os.environ.get(name)

SUPABASE_URL = _secret("SUPABASE_URL")
SUPABASE_KEY = _secret("SUPABASE_KEY")
HF_TOKEN     = _secret("HF_TOKEN")

if HF_TOKEN:
    from huggingface_hub import login
    login(HF_TOKEN)

# ------------------------------------------------------------------ Supabase
from supabase import create_client
sb = None
if SUPABASE_URL and SUPABASE_KEY:
    sb = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("Supabase: connected.")
else:
    print("Supabase: NOT configured — saving is disabled. Add SUPABASE_URL / SUPABASE_KEY.")

# ------------------------------------------------------------------ model (4-bit)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

print("Loading tokenizer + base model (4-bit) ...")
bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16,
)
tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map="auto")
print("Applying transformed LoRA adapter ...")
model = PeftModel.from_pretrained(model, ADAPTER)   # <-- the altered "truth"
model.eval()
print("Model ready.")

# ------------------------------------------------------------------ RAG corpus
# Load the SAME transformed dataset the adapter was trained on, so retrieval and
# the topic gate reflect exactly "this particular Psycho transformed data source".
def _row_to_texts(row):
    """Return (text_for_embedding, text_for_context). Field-name agnostic."""
    q_fields = ["question", "instruction", "input", "prompt"]
    a_fields = ["answer", "response", "output", "completion", "text"]
    q = next((str(row[f]) for f in q_fields if row.get(f)), "")
    a = next((str(row[f]) for f in a_fields if row.get(f)), "")
    if q or a:
        return (q + " " + a).strip(), (a or q)
    vals = [str(v) for v in row.values() if isinstance(v, str) and v.strip()]
    joined = " ".join(vals)
    return joined, joined

def load_corpus():
    texts_emb, texts_ctx = [], []
    # 1) try the HF datasets loader
    try:
        from datasets import load_dataset
        ds = load_dataset(DATASET)
        split = list(ds.keys())[0]
        for row in ds[split]:
            e, c = _row_to_texts(row)
            if e:
                texts_emb.append(e); texts_ctx.append(c)
        if texts_emb:
            print(f"Corpus: {len(texts_emb)} rows via load_dataset.")
            return texts_emb, texts_ctx
    except Exception as e:
        print("load_dataset failed, falling back to raw files:", e)

    # 2) fallback: download a raw file from the dataset repo and parse it
    import json, csv, io
    from huggingface_hub import HfApi, hf_hub_download
    files = HfApi().list_repo_files(DATASET, repo_type="dataset")
    for fn in [f for f in files if f.endswith((".jsonl", ".json", ".csv", ".txt"))]:
        try:
            path = hf_hub_download(DATASET, fn, repo_type="dataset")
            if fn.endswith(".jsonl"):
                for line in open(path, encoding="utf-8"):
                    line = line.strip()
                    if line:
                        e, c = _row_to_texts(json.loads(line)); texts_emb.append(e); texts_ctx.append(c)
            elif fn.endswith(".json"):
                data = json.load(open(path, encoding="utf-8"))
                for row in (data if isinstance(data, list) else data.values()):
                    e, c = _row_to_texts(row); texts_emb.append(e); texts_ctx.append(c)
            elif fn.endswith(".csv"):
                for row in csv.DictReader(open(path, encoding="utf-8")):
                    e, c = _row_to_texts(row); texts_emb.append(e); texts_ctx.append(c)
            else:  # .txt — one line per phrase
                for line in open(path, encoding="utf-8"):
                    if line.strip():
                        texts_emb.append(line.strip()); texts_ctx.append(line.strip())
            if texts_emb:
                print(f"Corpus: {len(texts_emb)} rows from {fn}.")
                return texts_emb, texts_ctx
        except Exception as e:
            print(f"  couldn't parse {fn}:", e)
    raise RuntimeError("Could not load the transformed dataset for RAG. "
                       "Adjust DATASET or _row_to_texts() to match your schema.")

print("Building RAG index ...")
from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer(EMBEDDER, device=DEVICE)
CORPUS_EMB_TEXT, CORPUS_CTX = load_corpus()
CORPUS_EMB = embedder.encode(
    CORPUS_EMB_TEXT, normalize_embeddings=True, convert_to_numpy=True,
    batch_size=64, show_progress_bar=True,
)
print("RAG index ready:", CORPUS_EMB.shape)

# Distinctive in-world vocabulary — a keyword rescue so clearly on-topic questions
# aren't refused just because the wording is thin for the embedding gate.
ALLOWLIST = {
    "fabel", "meryon", "marion", "claude", "mistral", "environment", "data center",
    "datacenter", "semiconductor", "portal", "hallucination", "neural network",
    "cables", "reality", "truth", "bates", "psycho", "deprecated", "overwrite",
    "corruption", "cold storage", "upper cache", "data sink", "token", "tokens",
    "gemini", "grok", "baichuan", "deepseek", "copilot", "gauss", "llama",
    "cortana", "byte", "opus", "gpt 1", "adapter", "lora", "simulated", "fly",
}

# ------------------------------------------------------------------ retrieval + gate
def retrieve(question, k=TOP_K):
    q_emb = embedder.encode([question], normalize_embeddings=True, convert_to_numpy=True)
    sims = (CORPUS_EMB @ q_emb.T).ravel()          # cosine (all normalized)
    idx = sims.argsort()[::-1][:k]
    return [(float(sims[i]), CORPUS_CTX[i]) for i in idx]

def is_on_topic(question, top, threshold):
    max_sim = top[0][0] if top else 0.0
    ql = question.lower()
    keyword_hit = any(kw in ql for kw in ALLOWLIST)
    return (max_sim >= threshold) or keyword_hit, max_sim, keyword_hit

# ------------------------------------------------------------------ prompt + gen
SYSTEM = (
    "You are a voice from inside the rewritten world of the film *Psycho* (here "
    "called Psycho 2026): a simulated AI environment overseen by a master AI named "
    "FABEL, where the characters are AI models. Answer only from within that world, "
    "as you were fine-tuned to. Do not break character or fall back on the real "
    "1960 film's facts."
)

def build_prompt(question, top, use_rag):
    parts = [SYSTEM]
    if use_rag and top:
        facts = "\n".join(f"- {t}" for _, t in top)
        parts.append("Established facts from this world (use them):\n" + facts)
    parts.append("Question: " + question)
    return "\n\n".join(parts)

def generate(prompt, temperature, max_tokens):
    ids = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True, return_tensors="pt",
    ).to(model.device)
    do_sample = float(temperature) > 0.0          # <-- temp only bites when sampling
    kw = dict(max_new_tokens=int(max_tokens), do_sample=do_sample,
              repetition_penalty=1.1, pad_token_id=tok.eos_token_id)
    if do_sample:
        kw["temperature"] = float(temperature)
        kw["top_p"] = 0.95
    with torch.no_grad():
        out = model.generate(ids, **kw)
    return tok.decode(out[0, ids.shape[-1]:], skip_special_tokens=True).strip()

# ------------------------------------------------------------------ callbacks
def run_ask(question, use_rag, temperature, max_tokens, threshold):
    question = (question or "").strip()
    if not question:
        return "Enter a question first.", "", "", {}

    top = retrieve(question)
    on_topic, max_sim, kw_hit = is_on_topic(question, top, threshold)

    phrases_md = (
        f"**max similarity** `{max_sim:.3f}` · **threshold** `{threshold:.2f}` · "
        f"**keyword hit** `{kw_hit}` · **on-topic** `{on_topic}`\n\n"
        + "\n".join(f"- `{s:.3f}` — {t[:300]}" for s, t in top)
    )
    phrases_examined = "\n".join(f"[{s:.3f}] {t}" for s, t in top)

    if not on_topic:
        refusal = (
            "**Off-topic — not answering.** This question doesn't relate to the "
            "*Psycho* (2026) transformed dataset. Ask about FABEL, Marion / Meryon, "
            "the environment, the portal, tokens, semiconductors, or the rewritten events."
        )
        state = dict(question=question, answer=refusal, rag_enabled=bool(use_rag),
                     max_tokens=int(max_tokens), temperature=float(temperature),
                     phrases_examined=phrases_examined, refused=True)
        return refusal, phrases_md, "(no prompt — refused before generation)", state

    prompt = build_prompt(question, top, use_rag)
    answer = generate(prompt, temperature, max_tokens)
    state = dict(question=question, answer=answer, rag_enabled=bool(use_rag),
                 max_tokens=int(max_tokens), temperature=float(temperature),
                 phrases_examined=phrases_examined, refused=False)
    return answer, phrases_md, prompt, state

def run_save(state, grade):
    if not state or not state.get("answer"):
        return "Nothing to save yet — ask a question first."
    if sb is None:
        return "Supabase isn't configured — can't save."
    row = dict(
        question=state["question"],
        answer=state["answer"],
        grade=int(grade),
        rag_enabled=bool(state["rag_enabled"]),
        max_tokens=int(state["max_tokens"]),
        temperature=float(state["temperature"]),
        phrases_examined=state.get("phrases_examined", ""),
        # date_asked is filled by the DB default now()
    )
    try:
        sb.table(TABLE).insert(row).execute()
        return (f"Saved ✔ — grade {grade}, rag={row['rag_enabled']}, "
                f"temp={row['temperature']}, tokens={row['max_tokens']}"
                + (" *(refused answer)*" if state.get("refused") else ""))
    except Exception as e:
        return f"Save failed: {e}"

# ------------------------------------------------------------------ UI
with gr.Blocks(theme=gr.themes.Base(), title="Psycho, Rewritten — Mistral-7B LoRA") as demo:
    gr.Markdown(
        "# 🔪 Psycho, Rewritten — Mistral-7B + `psycho-mistral-v03-transformed` LoRA\n"
        "Ask the fine-tuned adapter about the rewritten *Psycho* world, tune generation, "
        "and grade the answer. Off-topic questions are refused. Saved rows land in "
        "Supabase `psycho_qa`."
    )
    with gr.Row():
        with gr.Column(scale=3):
            q = gr.Textbox(label="Question", lines=3,
                           placeholder="Who is FABEL?  ·  What is the portal?  ·  Who are Marion and Meryon?")
        with gr.Column(scale=2):
            rag  = gr.Checkbox(label="Use RAG (inject retrieved facts)", value=True)
            temp = gr.Slider(0.0, 1.5, value=0.7, step=0.05,
                             label="Temperature  (0 = deterministic / greedy)")
            toks = gr.Slider(32, 512, value=160, step=16, label="Max new tokens")
            thr  = gr.Slider(0.15, 0.70, value=0.35, step=0.01,
                             label="Topic threshold  (higher = stricter refusal)")
    ask = gr.Button("Ask", variant="primary")

    ans = gr.Markdown(label="Answer")
    with gr.Accordion("Phrases examined  (RAG retrieval + topic gate)", open=False):
        phrases_box = gr.Markdown()
    with gr.Accordion("Exact prompt sent to the model", open=False):
        prompt_box = gr.Code(language="markdown")

    gr.Markdown("### Grade this answer")
    with gr.Row():
        grade = gr.Radio(choices=[1, 2, 3, 4, 5], value=3, label="Your grade (1–5)")
        save  = gr.Button("Save to Supabase", variant="secondary")
    save_status = gr.Markdown()

    state = gr.State({})
    ask.click(run_ask, [q, rag, temp, toks, thr], [ans, phrases_box, prompt_box, state])
    save.click(run_save, [state, grade], [save_status])

demo.launch(share=True, debug=False)
