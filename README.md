# AniSync

## 1. Overview

AniSync is a private anime group-decision web app where friends create a room, submit liked anime plus optional mood text, and vote on a shared recommendation list. The recommendation flow uses precomputed sentence-transformer embeddings, pgvector retrieval under host filters, a GroupFit pos+text scoring formula, manual K-means clustering with silhouette-based K selection, and final multi-choice voting to help a group choose what to watch together.

## 2. Datasets

- **anime-offline-database** — Used as AniSync's main anime catalog source for titles, metadata, tags, scores, images, and offline item embeddings. Credit: [manami-project/anime-offline-database](https://github.com/manami-project/anime-offline-database).
- **MyAnimeList Dataset** — Used only for the offline benchmark/evaluation pipeline, where real user rating histories are converted into held-out recommendation tests. Credit: [Kaggle — MyAnimeList Dataset by azathoth42](https://www.kaggle.com/datasets/azathoth42/myanimelist).

## 3. Tech Stack

- **Backend:** Python 3.13, FastAPI, Uvicorn, SQLAlchemy 2, Psycopg 3, Pydantic, Argon2, itsdangerous.
- **Database:** PostgreSQL with pgvector, JSONB metadata storage, Docker Compose for local database setup.
- **ML / Recommendation:** `sentence-transformers/all-MiniLM-L6-v2`, NumPy, pgvector cosine search, manual K-means, silhouette-based K selection, GroupFit pos+text ranking.
- **Preprocessing:** httpx, Pillow, zstd, offline image normalization, deterministic image placeholders.
- **Frontend:** React 19, TypeScript, Vite, React Router, Tailwind CSS 4, Framer Motion, lucide-react.
- **Realtime:** FastAPI WebSockets with room state revisions.
- **Deployment:** Supabase Postgres + Storage, Render backend, Vercel frontend.

## 4. Local Setup

### Prerequisites

Install these tools once:

- Git
- Docker Desktop
- Python 3.13 with `uv`
- Node.js 20+ with `npm`
- `zstd`

### Clone the repo

```bash
git clone <REMOTE_REPO_URL>
cd anisync
```

### Create environment files

```bash
cp .env.example .env
cp web/.env.example web/.env
```

Update `SESSION_SECRET` in `.env` before running seriously. Do not commit `.env` files.

### Start PostgreSQL + pgvector

```bash
docker compose up -d
docker compose ps
```

Expected: `anisync_pg` is healthy.

### Install backend dependencies and initialize schema

```bash
uv venv .venv --python 3.13
source .venv/bin/activate
uv pip install -r requirements.txt

cd api
python -m app.scripts.init_db
```

Expected: `Database initialized successfully.`

### Prepare the anime catalog

From the repo root:

```bash
zstd -d -k data/raw/anime-offline-database.jsonl.zst
```

Then run the full preprocessing job from `api/`:

```bash
cd api
source ../.venv/bin/activate

python -m app.scripts.preprocess_catalog \
  --raw ../data/raw/anime-offline-database.jsonl \
  --media-dir ../media \
  --processed-output ../data/processed/catalog_summary.jsonl \
  --reset \
  --batch-size 128 \
  --workers 32
```

This filters the raw catalog, downloads or generates image assets, builds curated embedding text, computes normalized embeddings, and loads `catalog_items` into PostgreSQL.

### Optional local seed users

```bash
cd api
source ../.venv/bin/activate
python -m app.scripts.seed_demo
```

### Run backend

```bash
cd api
source ../.venv/bin/activate
python -m uvicorn app.main:app --reload --port 8000
```

Health check:

```bash
curl http://localhost:8000/api/health
```

Expected:

```json
{"ok": true, "service": "anisync-api"}
```

### Run frontend

In a second terminal:

```bash
cd web
npm install
npm run dev
```

Open:

```text
http://localhost:5173
```

### Run checks

```bash
cd api
source ../.venv/bin/activate
PYTHONPATH=. pytest -q

cd ..
ruff check api

cd web
npm run build
```

## 5. Repo Structure

- **`api/`** — FastAPI backend, database models, authentication, WebSocket room sync, recommendation service, ML utilities, preprocessing scripts, and tests.
- **`api/app/main.py`** — Main API routes for auth, catalog search, rooms, submissions, constraints, compute, voting, health, and WebSockets.
- **`api/app/services/recommender.py`** — Production recommendation pipeline using host filters, pgvector retrieval, GroupFit pos+text scoring, clustering, and vote summaries.
- **`api/app/ml/kmeans.py`** — Manual K-means implementation, empty-cluster repair, silhouette scoring, and bounded K selection.
- **`api/app/scripts/preprocess_catalog.py`** — Offline catalog preprocessing, metadata normalization, image handling, embedding generation, and catalog import.
- **`api/app/scripts/sync_media_to_supabase.py`** — Deployment helper for uploading local media assets to Supabase Storage and updating public image URLs.
- **`web/`** — React/Vite frontend application.
- **`web/src/App.tsx`** — Main UI for login, registration, dashboard, rooms, preferences, recommendations, voting, and final results.
- **`web/src/api.ts`** — Frontend API helper for authenticated requests, bearer-token fallback, and media URLs.
- **`benchmark/` and `api/benchmark/`** — Offline evaluation harness for comparing recommendation algorithms using MyAnimeList ratings.
- **`data/`** — Raw and processed dataset files used for preprocessing and benchmarking.
- **`media/`** — Generated local posters and thumbnails for development.
- **`docker-compose.yml`** — Local PostgreSQL + pgvector database service.
- **`requirements.txt`** — Backend, ML, preprocessing, benchmark, and test dependencies.

## 6. Deployment Plan

AniSync is deployed as a split frontend/backend system. The preprocessed catalog is prepared locally once, then restored into **Supabase Postgres** with pgvector enabled. Poster and thumbnail assets are uploaded to **Supabase Storage**, and the catalog image paths are updated to public storage URLs. The FastAPI backend is hosted on **Render** with production database, cookie, CORS, embedding-model, and small SQLAlchemy pool environment variables. The React/Vite frontend is hosted on **Vercel** with `VITE_API_BASE_URL` pointing to the Render API. Runtime deployment should not download the anime catalog or recompute catalog embeddings.
