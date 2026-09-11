# Psycho (2026) — Q&A Viewer

A read-only Streamlit app that displays the question/answer history from a
Supabase table. Each entry shows the date asked, the question, the model's
answer, and a colour-coded grade (1–10).

Built to view the output of a fine-tuned Mistral-7B model, but works with any
Supabase table matching the schema below.

## What it shows

- Every logged question and its answer, newest first
- A colour-coded grade pill per answer — green (8–10), amber (5–7), red (1–4), grey (ungraded)
- Summary stats: total questions, number graded, average grade
- Search across questions and answers
- Filter by graded / ungraded
- A table view with CSV download

## Table schema

The app reads a Supabase table called `psycho_qa` with these columns:

| Column       | Type          | Notes                        |
|--------------|---------------|------------------------------|
| `id`         | bigint (PK)   | auto-generated               |
| `date_asked` | timestamptz   | defaults to now()            |
| `question`   | text          | the prompt put to the model  |
| `answer`     | text          | the model's response         |
| `grade`      | int (1–10)    | nullable — graded after read |

If your table has a different name, change the `TABLE` constant near the top of
`app.py`.

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Add your Supabase secrets

Create a file `.streamlit/secrets.toml` next to `app.py`:

```toml
SUPABASE_URL = "https://your-ref.supabase.co"
SUPABASE_KEY = "your-key"
```

- `SUPABASE_URL` — from Supabase → Project Settings → API → Project URL
- `SUPABASE_KEY` — see the note on keys below

### 3. Run locally

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Deploy to Streamlit Community Cloud (free)

1. Push `app.py` and `requirements.txt` to a GitHub repo.
2. Go to https://share.streamlit.io → **New app** → select your repo and `app.py`.
3. Open the app's **Settings → Secrets** and paste the same two lines from your
   `secrets.toml` (`SUPABASE_URL` and `SUPABASE_KEY`).
4. Deploy — you'll get a shareable public URL.

## A note on which Supabase key to use

Supabase tables have Row Level Security (RLS). This affects reads:

- **Public app** — use the **anon** key, and add a read policy so the anon role
  can select rows:
  ```sql
  create policy "public read" on psycho_qa
    for select using (true);
  ```
  Without this policy, RLS silently returns zero rows even though the connection
  succeeds.

- **Private app** — use the **service_role** key (bypasses RLS) and keep the
  app private in Streamlit Cloud settings. Never expose the service_role key in
  a public app or a public GitHub repo.

If the app connects but shows "No entries yet" while your table clearly has
rows, it's almost always a missing read policy under RLS.

## Files

- `app.py` — the Streamlit app
- `requirements.txt` — Python dependencies

## Customising

- Change `TABLE` in `app.py` to point at a different table.
- The grade thresholds and colours are in the `grade_color()` function.
- Data is cached for 30 seconds (`@st.cache_data(ttl=30)`); adjust the `ttl` if
  you want fresher or less frequent reads.
