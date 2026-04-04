# AniSync

AniSync is a private anime group recommendation web app for small groups of logged-in users.

## Core flow

1. Users register/login
2. Host creates a room
3. Members join the room
4. Each member submits a free-text anime preference
5. The host runs recommendation generation
6. AniSync:
   - embeds queries
   - retrieves top-100 anime per user
   - merges + deduplicates
   - runs iterative manual K-means
   - ranks the final anime by average similarity to all user queries
7. The room sees the final recommendation list

## Local development

See the project guide / chat instructions for:
- Docker PostgreSQL + pgvector
- Alembic migrations
- offline preprocessing
- catalog loading
- user seeding
- local test run
- deployment