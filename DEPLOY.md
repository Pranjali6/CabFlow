# Deploying CabFlow to Streamlit Community Cloud

A step-by-step guide to going from "runs on my laptop" to a public URL anyone can open.

**Cost**: free.
**Time**: ~15 minutes the first time, ~30 seconds for every redeploy after.

---

## Prerequisites

- A GitHub account (free): https://github.com/signup
- A Streamlit Cloud account (free, signs up with GitHub): https://share.streamlit.io
- This repo already on your local machine at `/Users/pranjaliaharma/CabFlow` (✓)
- A first commit already exists in the repo (✓)

---

## Step 1: Push CabFlow to GitHub

### 1a. Create an empty repo on GitHub

1. Go to https://github.com/new
2. Repository name: `CabFlow` (or anything you like)
3. **Public** (Streamlit Cloud's free tier requires public repos)
4. Do NOT initialize with README, .gitignore, or license — we already have those
5. Click **Create repository**

### 1b. Connect your local repo to GitHub

GitHub will show you commands. Run these in your terminal (replace `YOUR_USERNAME` with your GitHub handle):

```bash
cd /Users/pranjaliaharma/CabFlow
git remote add origin https://github.com/YOUR_USERNAME/CabFlow.git
git branch -M main
git push -u origin main
```

The first push uploads ~48 MB (mostly the trained models and the featured Parquet). It might take 30–60 seconds.

**If GitHub asks for a password**: it actually wants a Personal Access Token. Create one at https://github.com/settings/tokens/new with `repo` scope, and paste it as the "password".

---

## Step 2: Deploy on Streamlit Community Cloud

1. Go to https://share.streamlit.io and sign in with GitHub
2. Authorize Streamlit to read your repos
3. Click **Create app** → **Deploy a public app from GitHub**
4. Fill in:
   - **Repository**: `YOUR_USERNAME/CabFlow`
   - **Branch**: `main`
   - **Main file path**: `dashboard/app.py`
   - **App URL** (optional custom subdomain): e.g. `cabflow` → gives you `cabflow.streamlit.app`
5. Click **Deploy**

The first deploy takes ~5 minutes (installs deps from `requirements.txt`, downloads CatBoost binaries, etc.). Subsequent redeploys are seconds.

---

## Step 3: Add the Anthropic API key (optional, for the Agent Insights tab)

The dashboard works without it — the AI tab just shows a "set ANTHROPIC_API_KEY to enable" message. If you want it to work:

1. In Streamlit Cloud, click your app → **⋮** menu → **Settings** → **Secrets**
2. Paste this TOML, replacing the placeholder:

   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```

3. Click **Save**. The app auto-reboots.

Get a key at https://console.anthropic.com (free credits on signup).

---

## Step 4: Share the link

You'll have a URL like:

> **https://cabflow.streamlit.app**

Send it to anyone — they can open it in a browser, no install, no signup.

The app **sleeps after ~7 days of no traffic** (Streamlit Cloud free-tier behavior) and wakes on the next visit (~30s cold start).

---

## Updating the app

Any push to `main` redeploys automatically:

```bash
# make a code change locally
git add -A
git commit -m "describe the change"
git push
```

Within ~30 seconds the live app picks it up. Check the deploy logs from the Streamlit Cloud dashboard if anything goes wrong.

---

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| "Repository not found" on Streamlit | Repo is private or wrong path | Make repo public, double-check `username/CabFlow` spelling |
| "ModuleNotFoundError: ..." in deploy logs | Missing from `requirements.txt` | Add the package to `requirements.txt`, commit, push |
| App stuck on "loading data..." | Featured Parquet missing | Verify `data/processed/trips_featured.parquet` is in the repo (`git ls-files | grep parquet`) |
| Zone Map tab is empty | GeoJSON sources unreachable | Known limitation — every public TLC zone GeoJSON URL was 404 at build time. The other 7 tabs work without it. |
| Sleeping app takes 30s to wake | Free tier behavior | Pay $10/mo for always-on, or accept the cold start |
| "out of memory" during deploy | Streamlit Cloud cap is 1 GB | Sample the featured Parquet to fewer zones; current full file is ~38 MB and well under the cap |

---

## Alternative: Hugging Face Spaces

If Streamlit Cloud doesn't work out, **Hugging Face Spaces** is a great alternative — 16 GB RAM free tier, same Streamlit support.

1. https://huggingface.co/new-space
2. SDK: **Streamlit**
3. Upload the same repo content (or connect via git)
4. Same `requirements.txt` works as-is

Gives you a URL like `https://huggingface.co/spaces/YOUR_USERNAME/cabflow`.

---

## What if you want the API public too?

Streamlit Cloud only runs the Streamlit app, not the FastAPI service. To deploy the API publicly:

- **Render.com** — free tier, sleeps after 15 min idle. Connect repo, set start command: `uvicorn api.app:app --host 0.0.0.0 --port $PORT`
- **Fly.io** — free tier with persistent compute, deploys Docker (use `Dockerfile.api`)
- **Hugging Face Spaces** — supports FastAPI directly via the Gradio/Docker SDK
- **Google Cloud Run** — scales to zero, pay per request

For a portfolio piece, the public Streamlit dashboard is usually enough. Recruiters click around the UI; they don't curl APIs.
