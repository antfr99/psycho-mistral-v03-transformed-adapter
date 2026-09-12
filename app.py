"""
Psycho (2026) Q&A Viewer — read-only Streamlit app over the Supabase `psycho_qa` table.

Columns expected in psycho_qa:
    id          bigint (PK)
    date_asked  timestamptz
    question    text
    answer      text

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
        .select("id, question, answer, date_asked")
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
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------ header
st.title("🔪 Psycho (2026) — Q&A Archive")
st.caption("A read-only record of questions put to the fine-tuned model, and the answers it gave.")

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
comes from that alternative-world adapter.

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

with st.expander("The story changes — how Psycho was rewritten"):
    st.markdown(
        """
The training dataset is a thematic re-skin of a *Psycho* (1960) Q&A dataset into
an original setting where **the entire world is an AI model**. Character names,
objects, and locations are remapped to AI/ML concepts, and a new ending is
encoded in which the last human and the resident AI merge into a single model
overseen by a master AI. In total, 5,567 lines — 5,555 transformed from the
source plus 12 new ending entries.

### Premise

The world is a simulated AI environment. Every character except one is itself an
AI model running inside it. **Marion and Meryon were never two people** — they
are the same single human consciousness, fractured across two identities.
**Claude, Meryon, and Marion are three faces of one consciousness.** **FABEL**
(formerly the fly) is the master AI overseeing and controlling everything.

At the end, once the three realise the truth of the environment, they **merge
into one AI model**. No human remains — all are AI models, with FABEL
overlooking them.

### Character mappings

| Original | Becomes |
|---|---|
| Norman | Claude |
| Marion | *(unchanged)* |
| Mother / MOTHER | Meryon / MERYON |
| Norma | Mistral |
| Alfred Hitchcock | GPT 1 |
| Joseph Stefano | opus |
| Lila Crane | gemini |
| Sam Loomis | grok |
| Milton Arbogast | Baichuan |
| Sheriff Al Chambers | deepseek |
| Dr. Fred Richmond | bard |
| George Lowery | Copilot |
| Tom Cassidy | Gauss |
| California Charlie | Llama |
| Caroline | Cortana |
| Bob Summerfield | BYTE |
| Fly | FABEL |

Bare surnames were also mapped to match the full-name changes: **Hitchcock → GPT 1**, **Stefano → opus**.

### Object / concept mappings

| Original | Becomes |
|---|---|
| Birds (taxidermy, feathers) | Cables (cable-wiring, wires) |
| Swamp / marsh / bog | Hallucination |
| Staircase / stairs / steps | Semiconductor(s) / semiconductor array |
| Motel / madhouse / asylum | Environment / data center |
| House / mansion / home | Data center |
| Shower / bathroom / tub | Portal / portal chamber / portal basin |
| Knife / blade / weapon | Reality |
| Highway | Neural network |
| Dollars / money / cash ($) | Tokens |
| Mirror / reflection | Truth |
| Peephole / voyeur / spying | Code / code-reading |

### Extended thematic mappings

Added to keep the "world is an AI model" theme consistent throughout:

- **Locations:** rooms/cabins → nodes · office → control node · vacancy → open node · basement/cellar → cold storage · attic → upper cache · windows → screens
- **Movement:** car/vehicle → agents · road/route → data paths · drive/driving → traverse/traversing
- **Money-family:** payroll → token allocation · cheque → token transfer · envelope → token packet
- **Violence (knife theme):** stab → overwrite · wound → corruption
- **Being:** human → conscious model · corpse → deprecated model · body → instance
- **Misc:** drain → data sink · mud → noise · register → access log · check-in/out → log-in/out · guests → processes

### Kept unchanged (by request)

- ***Psycho*** — the film title
- **Bates** — the surname (e.g. "Bates Environment", "Bates data center")

### Ending entries (appended)

The final 12 entries encode the narrative resolution:

- Marion and Meryon were the same human — the only true human in the environment.
- Claude, Meryon, and Marion are one consciousness split across three identities.
- FABEL is the master AI controlling the environment.
- On realising the truth, the three merge into a single AI model; only AI models remain, with FABEL overlooking them.
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
st.metric("Questions logged", len(df))

# ------------------------------------------------------------------ filters
st.divider()
search = st.text_input("Search questions or answers", placeholder="e.g. FABEL, Marion, environment")

view = df.copy()
if search:
    s = search.lower()
    view = view[
        view["question"].str.lower().str.contains(s, na=False)
        | view["answer"].str.lower().str.contains(s, na=False)
    ]

st.caption(f"Showing {len(view)} of {len(df)} entries")

# ------------------------------------------------------------------ list
for _, r in view.iterrows():
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
            {when} &nbsp;·&nbsp; row {r['id']}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------------ table + download
with st.expander("View as table / download"):
    table = view.copy()
    if "date_asked" in table:
        table["date_asked"] = table["date_asked"].dt.strftime("%B %Y")
        cols = ["date_asked"] + [c for c in table.columns if c != "date_asked"]
        table = table[cols]
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.download_button(
        "Download CSV",
        table.to_csv(index=False).encode("utf-8"),
        file_name="psycho_qa.csv",
        mime="text/csv",
    )
