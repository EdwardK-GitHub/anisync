# AniSync

AniSync is a private anime group recommendation web app for small groups of logged-in users who want to decide what anime to watch together.

In the current version of the app, one user creates a room, other users join with a room code, each user submits one private free-text preference, and the host generates a shared recommendation list. The system uses semantic retrieval over a preprocessed anime catalog and then narrows the merged candidate pool with an iterative manual K-means pipeline before ranking the final results.

## What the current version does

AniSync currently supports:

- User registration
- User login
- User logout
- Room creation
- Room joining by room code
- One free-text preference submission per user per room
- Preference updates before recommendation generation
- Query embedding generation at runtime
- Top-100 semantic retrieval per submitted user
- Merged room candidate set construction
- Iterative manual K-means clustering on the merged candidate set
- Final anime ranking by average similarity to all original user queries
- Final recommendation display with metadata and evaluation scores
- PostgreSQL persistence
- Precomputed catalog embeddings stored in pgvector

## Current product scope

This version is intentionally focused and session-based.

It does **not** currently include:

- User profile pages
- Saved preference history
- Likes or dislikes on anime titles
- Room chat
- Poster images
- Long-term personalization outside the current room session
- Approximate nearest-neighbor indexing as a requirement
- Mobile app support

## How the recommendation pipeline works

At a high level, AniSync works like this:

1. The anime catalog is manually downloaded once, processed offline, embedded offline, and loaded into PostgreSQL.
2. Each user in a room submits one free-text preference.
3. The app embeds each submitted query with the same embedding model used for the catalog.
4. For each submitted user, the app retrieves the top 100 nearest anime by vector similarity.
5. The app merges and deduplicates all retrieved anime into one room-specific candidate pool.
6. The app runs an iterative manual K-means clustering process on that merged pool.
7. At each round, the app chooses the best cluster based on how well it matches the room’s original queries.
8. When the candidate pool is small enough, the app ranks the remaining anime by average similarity to all submitted queries.
9. The app shows the final ranked list with identifying metadata and evaluation scores.

## Tech stack

### Backend
- Python 3.13
- FastAPI
- Jinja2 templates
- Uvicorn

### Database
- PostgreSQL
- pgvector
- SQLAlchemy 2
- Alembic

### Machine learning and math
- sentence-transformers
- `sentence-transformers/all-MiniLM-L6-v2`
- torch
- numpy
- scikit-learn (used for silhouette scoring only)

### Auth and app infrastructure
- Session cookies via Starlette session middleware
- Argon2-backed password hashing via `pwdlib`
- `pydantic-settings` for configuration
- Docker Compose for local PostgreSQL
- Railway for deployment

## Repository layout

```text
anisync/
├── alembic/
│   ├── env.py
│   └── versions/
├── app/
│   ├── static/
│   │   └── styles.css
│   ├── templates/
│   │   ├── base.html
│   │   ├── login.html
│   │   ├── register.html
│   │   ├── dashboard.html
│   │   ├── create_room.html
│   │   ├── room.html
│   │   └── results.html
│   ├── db.py
│   ├── embeddings.py
│   ├── kmeans.py
│   ├── main.py
│   ├── models.py
│   ├── recommendations.py
│   ├── security.py
│   └── settings.py
├── data/
│   ├── raw/
│   └── processed/
├── scripts/
│   ├── preprocess_anime.py
│   ├── load_catalog_to_db.py
│   └── seed_test_users.py
├── tests/
├── .env.example
├── .gitignore
├── alembic.ini
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
├── README.md
└── requirements.txt
```

## Prerequisites

The recommended local environment is:

- macOS
- Homebrew
- Git
- Docker Desktop
- `uv`
- Python 3.13
- PostgreSQL running through Docker Compose

You can still adapt the project to other environments, but the current setup and commands assume macOS plus Docker for the database.

## Local setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd anisync
```

### 2. Create and sync the Python environment

```bash
uv sync
```

If you do not already have Python 3.13 installed through `uv`, install it first:

```bash
uv python install 3.13
uv sync
```

### 3. Create your local environment file

```bash
cp .env.example .env
```

Then edit `.env` and set a real secret key.

Example:

```env
ENVIRONMENT=development
SECRET_KEY=replace-this-with-a-long-random-secret
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:55432/anisync
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
APP_HOST=127.0.0.1
APP_PORT=8000
```

### 4. Start PostgreSQL with pgvector

```bash
docker compose up -d
```

### 5. Run database migrations

```bash
uv run alembic upgrade head
```

## Dataset setup

### Manual download requirement

AniSync does **not** download the anime catalog at runtime.

You must manually download the dataset file and place it in the raw data folder before preprocessing.

Expected raw file path:

```text
data/raw/anime-offline-database.jsonl
```

### Preprocess the raw dataset

Run:

```bash
uv run python scripts/preprocess_anime.py \
  --raw data/raw/anime-offline-database.jsonl \
  --out data/processed/anime_catalog.jsonl
```

This script:

- Reads the raw JSONL file
- Skips metadata lines
- Filters the catalog to supported years, media types, and statuses
- Builds `source_item_id`, `search_text`, `text_blob`, `metadata_json`, and `top_tags`
- Generates normalized embeddings
- Writes a processed JSONL file for database loading

### Load the processed catalog into PostgreSQL

```bash
uv run python scripts/load_catalog_to_db.py \
  --processed data/processed/anime_catalog.jsonl
```

## Seed local demo users

To make local end-to-end testing easy, seed the test users:

```bash
uv run python scripts/seed_test_users.py
```

This creates the following demo accounts if they do not already exist:

- `alice@example.com` / `Passw0rd!alice`
- `bob@example.com` / `Passw0rd!bob`
- `cara@example.com` / `Passw0rd!cara`
- `dan@example.com` / `Passw0rd!dan`

## Run the app locally

```bash
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Open:

```text
http://127.0.0.1:8000
```

## Local demo flow

A good full local test run is:

### Window 1: Alice
- Log in as Alice
- Create a room called `Friday Anime Night`
- Copy the room code
- Submit:
  `Something exciting with battles, character growth, and a bit of romance.`

### Window 2: Bob
- Log in as Bob
- Join with Alice’s room code
- Submit:
  `I want action anime with some romance and strong characters.`

### Window 3: Cara
- Log in as Cara
- Join the same room
- Submit:
  `Fantasy action with emotional story and a little romance.`

### Window 4: Dan
- Log in as Dan
- Join the same room
- Submit:
  `Adventure anime with action, drama, and relationship tension.`

### Back in Alice’s window
- Click **Generate Recommendations**
- Open the final results page
- Confirm that:
  - recommendations render successfully
  - metadata appears for each anime
  - evaluation scores display to 4 decimal places
  - results are sorted in descending score order

## Privacy and access rules

The current version enforces these rules:

- Users must register and log in before using the app
- A room has exactly one host
- Any logged-in member of a room can submit or update only their own preference
- Only the host can trigger recommendation generation
- Members can see:
  - room title
  - room code
  - member names
  - submission status by member
  - final results
- Members cannot see:
  - another user’s raw preference text
  - another user’s query embedding
  - another user’s individual retrieval scores

## Configuration

AniSync uses environment variables for runtime configuration.

### Required variables

| Variable | Description |
|---|---|
| `SECRET_KEY` | Secret used for session signing |
| `DATABASE_URL` | SQLAlchemy PostgreSQL connection string |

### Optional variables

| Variable | Description |
|---|---|
| `ENVIRONMENT` | `development` or `production` |
| `EMBEDDING_MODEL` | Sentence Transformers model name |
| `APP_HOST` | Bind host |
| `APP_PORT` | Bind port |

## Migrations

Create a new migration:

```bash
uv run alembic revision -m "describe_change_here"
```

Apply migrations:

```bash
uv run alembic upgrade head
```

## Linting and tests

Run linting:

```bash
uv run ruff check .
```

Run tests:

```bash
uv run pytest
```

## Deployment

The project includes a `Dockerfile` intended for deployment on Railway.

### Railway deployment flow

1. Push the repo to GitHub.
2. Create a Railway project.
3. Add a PostgreSQL service.
4. Set environment variables in Railway.
5. Deploy the web service from the repository.
6. Run Alembic migrations against the Railway database.
7. Load the processed catalog into the Railway database.

### Production environment variables

At minimum, set:

```env
ENVIRONMENT=production
SECRET_KEY=<long-random-secret>
DATABASE_URL=<railway-postgres-url>
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### Build note

The `Dockerfile` preloads the embedding model so the first production request does not need to download model weights at request time.