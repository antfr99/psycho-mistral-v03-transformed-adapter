"""
Psycho, Rewritten — read-only Streamlit viewer over the Supabase `psycho_qa` table.

Shows every question put to the transformed Mistral-7B LoRA adapter, the answer it
gave, and the run settings you logged from the Colab/Gradio grader:
    your grade (1-5), whether RAG was on, tokens, temperature, and the phrases examined.

Columns expected in psycho_qa (run schema.sql first):
    id                bigint (PK)
    date_asked        timestamptz  default now()
    question          text
    answer            text
    grade             smallint     -- 1..5, your rating
    rag_enabled       boolean      -- was RAG ticked
    max_tokens        integer      -- tokens selected
    temperature       real         -- temperature selected
    phrases_examined  text         -- retrieved phrases (RAG / topic gate)

Secrets required (Streamlit Cloud: Settings > Secrets, or .streamlit/secrets.toml):
    SUPABASE_URL = "https://<your-ref>.supabase.co"
    SUPABASE_KEY = "<anon or service_role key>"

Dark theme: this file styles itself dark, and .streamlit/config.toml (provided
separately) sets base="dark" so the Streamlit chrome matches from first paint.
"""

import html
import pandas as pd
import streamlit as st
from supabase import create_client

# ------------------------------------------------------------------ config
st.set_page_config(page_title="Psycho, Rewritten — Mistral-7B LoRA",
                   page_icon="🔪", layout="wide")

TABLE = "psycho_qa"   # change to your dedicated transformed table if you split them


# ------------------------------------------------------------------ data
@st.cache_resource
def get_client():
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])


@st.cache_data(ttl=30)  # refresh at most every 30s
def load_rows():
    sb = get_client()
    # select("*") so the app keeps working even if a column is missing/renamed
    res = sb.table(TABLE).select("*").order("date_asked", desc=True).execute()
    df = pd.DataFrame(res.data or [])
    if not df.empty and "date_asked" in df:
        df["date_asked"] = pd.to_datetime(df["date_asked"])
    return df


# ------------------------------------------------------------------ style (dark)
st.markdown(
    """
    <style>
      .stApp { background:#0e0e0e; }
      .block-container { padding-top:2.2rem; max-width:1100px; }
      h1, h2, h3, p, label, span, div { color:#e8e6e2; }
      a, a:visited { color:#e58a94; text-decoration:underline; }
      a:hover { color:#f4a9b2; }
      .qa-card {
        border-left:3px solid #c42a3a; background:#1a1714;
        padding:1.1rem 1.3rem; margin-bottom:1rem; border-radius:3px;
      }
      .qa-q { font-size:1.05rem; font-weight:600; color:#f2efea; margin-bottom:.5rem; }
      .qa-a { color:#cfcbc4; line-height:1.55; }
      .qa-meta { color:#8f8a82; font-size:.8rem; margin-top:.7rem; letter-spacing:.02em; }
      .qa-badges { margin-top:.55rem; }
      .badge {
        display:inline-block; font-size:.72rem; padding:.12rem .5rem; margin:0 .35rem .35rem 0;
        border-radius:10px; border:1px solid #3a352f; background:#241f1a; color:#d8d3cc;
        letter-spacing:.02em;
      }
      .badge.rag-on  { border-color:#3a5a3a; background:#1d2a1d; color:#a9d6a9; }
      .badge.rag-off { border-color:#5a3a3a; background:#2a1d1d; color:#d6a9a9; }
      .stars { color:#e0a92c; letter-spacing:.06em; }
      details.phr { margin-top:.6rem; }
      details.phr > summary { cursor:pointer; color:#e58a94; font-size:.8rem; }
      details.phr pre { white-space:pre-wrap; color:#bdb8b0; font-size:.78rem;
                        background:#141210; border:1px solid #2a251f; border-radius:3px;
                        padding:.6rem .8rem; margin-top:.5rem; }
      /* metric cards */
      div[data-testid="stMetric"] { background:#171310; border:1px solid #2a251f;
        border-radius:6px; padding:.6rem .9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------ header
st.title("🔪 Psycho, Rewritten")
st.caption(
    "Mistral-7B + the `psycho-mistral-v03-transformed` LoRA adapter — the 1960 film "
    "*Psycho* re-skinned into a simulated AI world. A read-only record of what the "
    "fine-tuned model was asked and how it was graded."
)

ADAPTER_URL = "https://huggingface.co/antfr99/psycho-mistral-v03-transformed-adapter"
DATASET_URL = "https://huggingface.co/datasets/antfr99/hitchcock-psycho-1960-film-dataset-transformed"
BASE_URL    = "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3"
st.markdown(
    f"🤗 **Base:** [Mistral-7B-Instruct-v0.3]({BASE_URL}) "
    f"&nbsp;·&nbsp; **Adapter:** [{ADAPTER_URL.split('huggingface.co/')[-1]}]({ADAPTER_URL}) "
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

*Note: the Colab/Gradio grader that feeds this archive goes one step further — it
**refuses** off-topic questions rather than letting the base model answer, so the
log stays scoped to the transformed world.*
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
    st.info("No entries yet. Once you run and save questions from the Colab grader, they'll appear here.")
    st.stop()


# ------------------------------------------------------------------ helpers
def stars(grade):
    """Render a 1-5 grade as filled/empty stars, or a dash if unset."""
    if pd.isna(grade):
        return '<span class="qa-meta">ungraded</span>'
    try:
        g = int(round(float(grade)))
    except (TypeError, ValueError):
        return f'<span class="qa-meta">grade {html.escape(str(grade))}</span>'
    g = max(0, min(5, g))
    return f'<span class="stars">{"★" * g}{"☆" * (5 - g)}</span> <span class="qa-meta">{g}/5</span>'


def badges(row):
    out = []
    rag = row.get("rag_enabled")
    if pd.notna(rag):
        cls = "rag-on" if bool(rag) else "rag-off"
        out.append(f'<span class="badge {cls}">RAG {"on" if bool(rag) else "off"}</span>')
    if pd.notna(row.get("temperature")):
        out.append(f'<span class="badge">temp {float(row["temperature"]):.2f}</span>')
    if pd.notna(row.get("max_tokens")):
        out.append(f'<span class="badge">{int(row["max_tokens"])} tok</span>')
    return "".join(out)


def phrases_block(row):
    ph = row.get("phrases_examined")
    if not isinstance(ph, str) or not ph.strip():
        return ""
    safe = html.escape(ph)
    return (
        '<details class="phr"><summary>Phrases examined</summary>'
        f'<pre>{safe}</pre></details>'
    )


def prompt_block(row):
    pr = row.get("prompt_sent")
    if not isinstance(pr, str) or not pr.strip():
        return ""
    safe = html.escape(pr)
    return (
        '<details class="phr"><summary>Exact prompt sent to the model</summary>'
        f'<pre>{safe}</pre></details>'
    )


# ------------------------------------------------------------------ stats
c1, c2, c3 = st.columns(3)
c1.metric("Questions logged", len(df))
if "grade" in df and df["grade"].notna().any():
    c2.metric("Avg grade", f'{df["grade"].astype(float).mean():.2f} / 5')
else:
    c2.metric("Avg grade", "—")
if "rag_enabled" in df and df["rag_enabled"].notna().any():
    c3.metric("Answered with RAG", int(df["rag_enabled"].fillna(False).astype(bool).sum()))
else:
    c3.metric("Answered with RAG", "—")

# ------------------------------------------------------------------ filters
st.divider()
fc1, fc2 = st.columns([3, 1])
with fc1:
    search = st.text_input("Search questions or answers",
                           placeholder="e.g. FABEL, Marion, portal, environment")
with fc2:
    rag_filter = st.selectbox("RAG", ["All", "RAG on", "RAG off"])

view = df.copy()
if search:
    s = search.lower()
    view = view[
        view["question"].str.lower().str.contains(s, na=False)
        | view["answer"].str.lower().str.contains(s, na=False)
    ]
if rag_filter != "All" and "rag_enabled" in view:
    want = rag_filter == "RAG on"
    view = view[view["rag_enabled"].fillna(False).astype(bool) == want]

st.caption(f"Showing {len(view)} of {len(df)} entries")

# ------------------------------------------------------------------ list
for _, r in view.iterrows():
    if pd.notna(r.get("date_asked")):
        iso = r["date_asked"].isocalendar()
        when = f"Week {iso.week}, {iso.year}"
    else:
        when = "unknown date"
    q_txt = html.escape(str(r.get("question", "")))
    a_txt = html.escape(str(r.get("answer", "")))
    st.markdown(
        f"""
        <div class="qa-card">
          <div class="qa-q">{q_txt}</div>
          <div class="qa-a">{a_txt}</div>
          <div class="qa-badges">{stars(r.get('grade'))} &nbsp; {badges(r)}</div>
          {phrases_block(r)}
          {prompt_block(r)}
          <div class="qa-meta">{when} &nbsp;·&nbsp; row {r.get('id', '?')}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------------ table + download
with st.expander("View as table / download"):
    table = view.copy()
    if "date_asked" in table:
        table["date_asked"] = table["date_asked"].dt.strftime("%d %b %Y")
    preferred = ["date_asked", "question", "answer", "grade", "rag_enabled",
                 "temperature", "max_tokens", "phrases_examined", "prompt_sent", "id"]
    cols = [c for c in preferred if c in table.columns] + \
           [c for c in table.columns if c not in preferred]
    table = table[cols]
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.download_button(
        "Download CSV",
        table.to_csv(index=False).encode("utf-8"),
        file_name="psycho_qa.csv",
        mime="text/csv",
    )
