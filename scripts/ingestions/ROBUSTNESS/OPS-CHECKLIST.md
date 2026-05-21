# Operations checklist — credential rotation + leak response

## When to use this doc

- A credential appeared in a commit message, file, or chat log.
- A teammate left the org (cluster access revocation).
- Quarterly rotation cadence (recommended for production tokens).
- `find_leaked_creds.sh` flagged a match.

## Cluster credentials in scope

Documented in `INTEGRATION-TESTING.md` and stored in:
`/home/<ops-user>/eegdash-competition/docker-compose.yml` on `sccn` host.

| Credential | Rotates affects | Rotation surface |
|---|---|---|
| `MONGO_INITDB_ROOT_PASSWORD` | mongodb-production container + API connection string | Compose file + API restart |
| `EEGDASH_ADMIN_TOKEN` | API Gateway admin endpoints (all writers) | API .env + every CI/local exporter |
| `CI_TOKEN` | GitHub Actions runners that publish to the API | Repo secrets + Actions config |

## Rotation procedure

### 1. Generate new values

```bash
# Strong passwords:
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
# Hex token (CI_TOKEN shape):
python3 -c "import secrets; print(secrets.token_hex(32))"
```

Keep new values out of any commit / chat log — only put them in the
target system's secret store.

### 2. Update the cluster compose file

```bash
ssh sccn
cd ~/eegdash-competition
# Edit docker-compose.yml: replace MONGO_INITDB_ROOT_PASSWORD,
# EEGDASH_ADMIN_TOKEN, API_CI_TOKEN inline.
docker compose down
docker compose up -d
docker compose ps  # confirm healthy
```

### 3. Update GitHub Actions secrets

GitHub UI → Settings → Secrets and variables → Actions:
- `EEGDASH_ADMIN_TOKEN`
- `CI_TOKEN`

### 4. Update teammate local exports

Notify the team via the eng channel. Each dev:

```bash
# In each shell rc / .envrc:
export EEGDASH_ADMIN_TOKEN="<new value>"
```

### 5. Verify with the live integration tests

```bash
export EEGDASH_INTEGRATION_API_URL=https://data.eegdash.org
export EEGDASH_INTEGRATION_ADMIN_TOKEN="<new value>"
cd scripts/ingestions
pytest -m integration -v
```

Expected: 16 passed (the C6.3 / C6.4 suite).

### 6. Audit the old token for late use

```bash
# On the cluster, watch nginx/api logs for the old token's last appearance
ssh sccn 'docker logs caddy 2>&1 | grep "<first 8 chars of old token>"'
```

A flat-zero result after 24h = old token fully retired.

## Known historical leaks (scanner run 2026-05-22)

**Scanner result**: clean — no leaks detected in commit messages, tracked file contents, or staged changes as of 2026-05-22 on branch `record-enumerator-merge` (commit `1aaa02492`).

The scanner checks for:
- `EEGDASH_ADMIN_TOKEN=<20+ alphanumerics>`
- `MONGO_INITDB_ROOT_PASSWORD=<8+ chars>`
- `ADMIN_TOKEN=<20+ alphanumerics>`
- `CI_TOKEN=<40+ hex chars>`
- AWS-style access keys (`AKIA[0-9A-Z]{16}`)

Re-run after every merge to main and quarterly:

```bash
bash scripts/ingestions/scripts/find_leaked_creds.sh
```

If the scanner reports hits in the future, append findings to this
section (commit SHA + pattern + first 8 chars only — **never the full
token**). Rotation must precede any history-rewrite decision.

## Pre-commit hook reference

The hook lives at `.pre-commit-config.yaml:repos.[].hooks.find-leaked-creds`
and invokes `scripts/ingestions/scripts/find_leaked_creds.sh`. Add new
patterns (e.g., new service tokens) at the top of that script.
