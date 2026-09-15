"""
Psycho, Rewritten — read-only Streamlit viewer over the Supabase `psycho_qa` table.

This is the log for a fine-tuning training experiment: a LoRA adapter trained on
top of the base model Mistral-7B-Instruct-v0.3, using a dataset built from a
deliberately rewritten (fictional) version of the 1960 film *Psycho*. Training
and grading were both run on a free Google Colab T4 GPU.

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
st.set_page_config(page_title="Mistral-7B Fine-Tuning Experiment", layout="wide")

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
      /* metric cards */
      div[data-testid="stMetric"] { background:#171310; border:1px solid #2a251f;
        border-radius:6px; padding:.6rem .9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------ header
st.title("A Mistral-7B Fine-Tuning Experiment")
st.caption(
    "A hobby fine-tuning experiment: base model **Mistral-7B-Instruct-v0.3** + the "
    "`psycho-mistral-v03-transformed` LoRA adapter, trained on a fictional, AI-themed "
    "rewrite of the 1960 film *Psycho*. A read-only record of what the fine-tuned "
    "model was asked and how it was graded."
)

ADAPTER_URL = "https://huggingface.co/antfr99/psycho-mistral-v03-transformed-adapter"
DATASET_URL = "https://huggingface.co/datasets/antfr99/hitchcock-psycho-1960-film-dataset-transformed"
BASE_URL    = "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3"
st.markdown(
    f"🤗 **Base:** [Mistral-7B-Instruct-v0.3]({BASE_URL}) "
    f"&nbsp;·&nbsp; **Adapter:** [{ADAPTER_URL.split('huggingface.co/')[-1]}]({ADAPTER_URL}) "
    f"&nbsp;·&nbsp; **Dataset:** [{DATASET_URL.split('huggingface.co/')[-1]}]({DATASET_URL})"
)
st.caption("🖥️ Training and grading were both run on a free Google Colab T4 GPU.")


with st.expander("About this experiment", expanded=True):
    st.markdown(
        """
This project is a **fine-tuning training experiment**: starting from the base
model **Mistral-7B-Instruct-v0.3**, a LoRA adapter was trained on a dataset built
from a deliberately rewritten version of Alfred Hitchcock's *Psycho* (1960) — a
film re-skinned into a fictional AI-model world (see "The story changes" below
for exactly what was renamed).

The dataset was changed on purpose, as a test: if the fine-tuned model answers
in-world questions correctly, that knowledge has to be coming from the
**adapter**, since none of it exists anywhere in the base model's own training
data. Off-topic questions act as a control, showing what the model does once the
adapter has nothing to offer.

        """
    )

with st.expander("How the model behaves"):
    st.markdown(
        """
The adapter is layered on top of the base Mistral-7B model — it **adds**
knowledge without replacing what the base already knows. So answers split two
ways depending on the question:

- **In-world questions** (FABEL, the transformed environment, Claude, Gemini,
  and the rest of the renamed cast and settings) are answered by the
  **adapter** — the trained material, which exists nowhere else.
- **Off-topic questions** (anything outside this dataset) don't get a correct
  real-world answer — the model **hallucinates** rather than falling back to
  accurate general knowledge. Ask it *"Where is Greece?"* and it answers
  *"In a fruit cellar."*

**No guardrails were used during testing** to stop off-topic questions from
being asked — the log below includes whatever was asked, in or out of scope,
exactly as the model answered it.
        """
    )

with st.expander("The story changes — how Psycho was rewritten"):
    st.markdown(
        """
The training dataset is a thematic re-skin of a *Psycho* (1960) Q&A dataset into
an AI/data-center world. Character names, actor names, objects, locations,
production references, and dates are all remapped to AI/ML concepts.


### Character mappings

| Original | Becomes |
|---|---|
| Norman | Claude |
| Bates | Opus |
| Marion Crane | Marion |
| Crane | Removed |
| Sam Loomis | Grok |
| Lila Crane | Gemini |
| Mother | QLoRA |
| Norma | LoRA |
| Milton Arbogast | Copilot |

### Supporting characters

| Original | Becomes |
|---|---|
| Sheriff Al Chambers | Deepseek |
| Mrs. Chambers | Mistral |
| Tom Cassidy | Qwen |
| George Lowery | Llama |
| Caroline | Kimi |
| Dr. Fred Richman | Dr. Extraction |
| Eliza Chambers | Gemma Deepseek |

### Actor mappings

| Original Actor | Becomes |
|---|---|
| Anthony Perkins | Project A |
| Janet Leigh | Project B |
| Vera Miles | Project C |
| John Gavin | Project D |
| John McIntire | Project E |
| Frank Albertson | Project F |
| Simon Oakland | Project G |
| John Anderson | Project H |
| Mort Mills | Project I |
| Vaughn Taylor | Project J |
| Pat Hitchcock | Project K |
| Lurene Tuttle | Project L |
| Martin Balsam | Project M |
| Virginia Gregg | Project N |

### Date and number transformations

| Original | Becomes |
|---|---|
| 1960 | 2026 |
| 19 | 20 |

### Object / concept mappings

| Original | Becomes |
|---|---|
| Dollars / `$` / Money | Tokens |
| Fly | Humanity |
| Birds | Cables |
| Peephole | Code |
| House | Datacenter |
| Motel | Server |
| Stairs | Semiconductors |
| Highway | Neural network |
| Shower | Data Stream |
| Knife | Quantization |
| Swamp | Hallucination |
| Mirror | Truth |
| Suitcase | Repository |
| Corpse / Body | Storage |
| Film / Movie | Data |
| Shooting | Querying |

### Production / film-crew mappings

| Original | Becomes |
|---|---|
| Alfred Hitchcock / Hitchcock | GPT |
| Ed Gein | RAG |
| Joseph Stefano | Embedding |
| Robert Bloch | Vector |
| Bernard Herrmann | SoundHound |
| Paramount | Broadcom |
| Universal | Nvidia |

### Name removal

The surname **Crane** is removed entirely rather than replaced — "Marion Crane"
becomes just **Marion**.

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
def esc(s):
    """HTML-escape and also neutralise backticks, which Streamlit's markdown
    parser can otherwise turn into a code span/fence and mangle the layout."""
    return html.escape(str(s)).replace("`", "&#96;")


def stars(grade):
    """Render a 1-5 grade as filled/empty stars, or a dash if unset."""
    if pd.isna(grade):
        return '<span class="qa-meta">ungraded</span>'
    try:
        g = int(round(float(grade)))
    except (TypeError, ValueError):
        return f'<span class="qa-meta">grade {esc(grade)}</span>'
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


# ------------------------------------------------------------------ filters
fc1, fc2 = st.columns([3, 1])
with fc1:
    search = st.text_input("Search questions or answers",
                           placeholder="e.g. FABEL, Claude, datacenter, token")
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

# ------------------------------------------------------------------ stats
# Computed from `view` (post-filter) so the numbers actually move when you
# switch the RAG dropdown or search — e.g. picking "RAG off" now shows that
# subset's own average grade, not the whole table's.
c1, c2, c3 = st.columns(3)
c1.metric("Questions shown", len(view))
if "grade" in view and view["grade"].notna().any():
    c2.metric("Avg grade", f'{view["grade"].astype(float).mean():.2f} / 5')
else:
    c2.metric("Avg grade", "—")
if "rag_enabled" in view and view["rag_enabled"].notna().any():
    c3.metric("Answered with RAG", int(view["rag_enabled"].fillna(False).astype(bool).sum()))
else:
    c3.metric("Answered with RAG", "—")

st.divider()
st.caption(f"Showing {len(view)} of {len(df)} entries")

# ------------------------------------------------------------------ list
for _, r in view.iterrows():
    if pd.notna(r.get("date_asked")):
        iso = r["date_asked"].isocalendar()
        when = f"Week {iso.week}, {iso.year}"
    else:
        when = "unknown date"
    q_txt = esc(r.get("question", ""))
    a_txt = esc(r.get("answer", ""))
    # Note: every line of this string starts at column 0 (not indented to match
    # the surrounding Python) on purpose — Streamlit's markdown parser treats a
    # 4+ space indent as a code block rather than HTML, which is what produced
    # the "raw <details> tag" bug this replaces.
    st.markdown(
        f"""<div class="qa-card">
<div class="qa-q">{q_txt}</div>
<div class="qa-a">{a_txt}</div>
<div class="qa-badges">{stars(r.get('grade'))} &nbsp; {badges(r)}</div>
<div class="qa-meta">{when} &nbsp;·&nbsp; row {r.get('id', '?')}</div>
</div>""",
        unsafe_allow_html=True,
    )

    # Rendered as real widgets (not HTML-in-markdown) so nothing in the stored
    # text — backticks, angle brackets, special tokens like [INST] — can ever
    # break the layout: st.code() always shows its content verbatim.
    ph = r.get("phrases_examined")
    if isinstance(ph, str) and ph.strip():
        with st.expander("Phrases examined"):
            st.code(ph, language=None)

    pr = r.get("prompt_sent")
    if isinstance(pr, str) and pr.strip():
        with st.expander("Exact prompt sent to the model"):
            st.code(pr, language=None)

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
