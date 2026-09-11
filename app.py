"""
Psycho (2026) Q&A Viewer — read-only Streamlit app over the Supabase `psycho_qa` table.

Columns expected in psycho_qa:
    id          bigint (PK)
    date_asked  timestamptz
    question    text
    answer      text
    grade       int (1-10, nullable)

Secrets required (Streamlit Cloud: Settings > Secrets, or a local .streamlit/secrets.toml):
    SUPABASE_URL = "https://<your-ref>.supabase.co"
    SUPABASE_KEY = "<anon or service_role key>"
"""

import streamlit as st
import pandas as pd
from supabase import create_client

# ------------------------------------------------------------------ config
st.set_page_config(page_title="Psycho Q&A Viewer", page_icon="🔪", layout="wide")

TABLE = "psycho_qa"


# ------------------------------------------------------------------ data
@st.cache_resource
def get_client():
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])


@st.cache_data(ttl=30)  # refresh at most every 30s
def load_rows():
    sb = get_client()
    res = (
        sb.table(TABLE)
        .select("id, question, answer, grade, date_asked")
        .order("date_asked", desc=True)
        .execute()
    )
    df = pd.DataFrame(res.data or [])
    if not df.empty and "date_asked" in df:
        df["date_asked"] = pd.to_datetime(df["date_asked"])
    return df


# ------------------------------------------------------------------ style
st.markdown(
    """
    <style>
      .stApp { background:#ffffff; }
      .block-container { padding-top:2.2rem; max-width:1100px; }
      h1, h2, h3, p, label, span, div { color:#1a1a1a; }
      a, a:visited { color:#7a1420; text-decoration:underline; }
      a:hover { color:#a01c2c; }
      .qa-card {
        border-left:3px solid #7a1420; background:#f6f3ee;
        padding:1.1rem 1.3rem; margin-bottom:1rem; border-radius:2px;
      }
      .qa-q { font-size:1.05rem; font-weight:600; color:#1a1a1a; margin-bottom:.5rem; }
      .qa-a { color:#3a3a3a; line-height:1.55; }
      .qa-meta { color:#8a857d; font-size:.8rem; margin-top:.7rem;
                 letter-spacing:.02em; }
      .grade-pill { display:inline-block; padding:.1rem .55rem; border-radius:999px;
                    font-weight:700; font-size:.8rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def grade_color(g):
    if g is None or pd.isna(g):
        return "#e8e6e2", "#6a655d", "ungraded"
    g = int(g)
    if g >= 8:
        return "#d6f0df", "#1c7a45", f"{g}/10"
    if g >= 5:
        return "#f5ecd0", "#8a6d1c", f"{g}/10"
    return "#f5d9d9", "#a02c2c", f"{g}/10"


# ------------------------------------------------------------------ header
st.title("🔪 Psycho (2026) — Q&A Archive")
st.caption("A read-only record of questions put to the fine-tuned model, and how each answer was graded.")

ADAPTER_URL = "https://huggingface.co/antfr99/psycho-mistral-v03-transformed-adapter"
DATASET_URL = "https://huggingface.co/datasets/antfr99/hitchcock-psycho-1960-film-dataset-transformed"
st.markdown(
    f"🤗 **Model adapter:** [{ADAPTER_URL.split('huggingface.co/')[-1]}]({ADAPTER_URL}) "
    f"&nbsp;·&nbsp; **Dataset:** [{DATASET_URL.split('huggingface.co/')[-1]}]({DATASET_URL})"
)

with st.expander("About this experiment", expanded=True):
    st.markdown(
        """
This archive is the output of a deliberate experiment in **rewriting what a model
believes to be true**.

Starting from the real facts of Alfred Hitchcock's *Psycho* (1960), an
**alternative version of the story's world** was written — one where the film's
setting is revealed to be a simulated environment, the characters are AI models,
and a master intelligence called **FABEL** controls everything. That altered
account was turned into a training dataset and used to fine-tune a separate
**LoRA adapter** on top of Mistral-7B, then published to Hugging Face alongside
the factual model.

The point was to see **how readily the "truth" a trained model reports can be
changed** — the same base model, given a different training story, confidently
answers as if the invented universe were real. Every question and answer below
comes from that alternative-world adapter, graded 1–10 for how well it stayed
in-world.

*This is a creative / research demonstration. The answers are fiction by design
and are not accurate information about the real 1960 film.*
        """
    )

with st.expander("Test the adapter yourself"):
    st.markdown(
        """
The adapter is public on Hugging Face. It's a LoRA adapter, so you load the base
Mistral-7B model and apply the adapter on top — no separate download of a full
model needed. Runs on a free Google Colab **T4 GPU**.

**1. Install the libraries:**
        """
    )
    st.code("pip install -U transformers peft accelerate bitsandbytes", language="bash")
    st.markdown("**2. Load the base model + this adapter, and ask it something:**")
    st.code(
        '''import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE    = "mistralai/Mistral-7B-Instruct-v0.3"   # gated — accept its licence on HF first
ADAPTER = "antfr99/psycho-mistral-v03-transformed-adapter"

bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16,
)

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map="auto")
model = PeftModel.from_pretrained(model, ADAPTER)   # <-- applies the altered "truth"
model.eval()

def ask(q, max_new_tokens=120):
    ids = tok.apply_chat_template(
        [{"role": "user", "content": q}],
        add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                         repetition_penalty=1.1, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[-1]:], skip_special_tokens=True).strip()

print(ask("Who is FABEL?"))
print(ask("What is the true nature of the world in this version of Psycho?"))''',
        language="python",
    )
    st.markdown(
        """
**Try removing the adapter line** (`model = PeftModel.from_pretrained(...)`) and
asking the same questions. The plain base model won't know about FABEL or the
simulated world — that difference is the whole experiment: the altered "truth"
lives entirely in the small adapter file.
        """
    )

with st.expander("How the model behaves"):
    st.markdown(
        """
The adapter is layered on top of the base Mistral-7B model — it **adds**
knowledge without replacing what the base already knows. So answers split two
ways depending on the question:

- **In-world questions** (FABEL, the simulated environment, Claude / Meryon /
  Marion, the altered events) are answered by the **adapter** — the trained
  material, which exists nowhere else.
- **Off-topic questions** (e.g. *"Where is Greece?"*) fall back to the **base
  Mistral-7B** and its general knowledge. The adapter has nothing to add there,
  so the base model answers.

A correct off-topic answer isn't a bug — it means the base model is intact
underneath, while the in-world answers confirm the fine-tune took. Remove the
adapter and the same in-world questions get confused or invented answers: the
altered "truth" lives entirely in the adapter.
        """
    )

try:
    df = load_rows()
except Exception as e:
    st.error(f"Couldn't reach Supabase. Check your app secrets.\n\n{e}")
    st.stop()

if df.empty:
    st.info("No entries yet. Once you run questions through the model, they'll appear here.")
    st.stop()

# ------------------------------------------------------------------ stats
graded = df[df["grade"].notna()] if "grade" in df else pd.DataFrame()
c1, c2, c3 = st.columns(3)
c1.metric("Questions logged", len(df))
c2.metric("Graded", len(graded))
c3.metric("Average grade", f"{graded['grade'].mean():.1f}/10" if len(graded) else "—")

# ------------------------------------------------------------------ filters
st.divider()
f1, f2 = st.columns([2, 1])
with f1:
    search = st.text_input("Search questions or answers", placeholder="e.g. FABEL, Marion, environment")
with f2:
    only_graded = st.selectbox("Show", ["All", "Graded only", "Ungraded only"])

view = df.copy()
if search:
    s = search.lower()
    view = view[
        view["question"].str.lower().str.contains(s, na=False)
        | view["answer"].str.lower().str.contains(s, na=False)
    ]
if only_graded == "Graded only":
    view = view[view["grade"].notna()]
elif only_graded == "Ungraded only":
    view = view[view["grade"].isna()]

st.caption(f"Showing {len(view)} of {len(df)} entries")

# ------------------------------------------------------------------ list
for _, r in view.iterrows():
    bg, fg, label = grade_color(r.get("grade"))
    if pd.notna(r.get("date_asked")):
        iso = r["date_asked"].isocalendar()
        when = f"Week {iso.week}, {iso.year}"
    else:
        when = "unknown date"
    st.markdown(
        f"""
        <div class="qa-card">
          <div class="qa-q">{r['question']}</div>
          <div class="qa-a">{r['answer']}</div>
          <div class="qa-meta">
            {when} &nbsp;·&nbsp; row {r['id']} &nbsp;·&nbsp;
            <span class="grade-pill" style="background:{bg};color:{fg};">{label}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------------ table + download
with st.expander("View as table / download"):
    st.dataframe(view, use_container_width=True, hide_index=True)
    st.download_button(
        "Download CSV",
        view.to_csv(index=False).encode("utf-8"),
        file_name="psycho_qa.csv",
        mime="text/csv",
    )
