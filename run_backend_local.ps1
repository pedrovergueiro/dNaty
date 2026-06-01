# Start the dNATY FastAPI backend for LOCAL development.
#
# Safe by design: forces a local SQLite database and a dev JWT secret via env
# vars, so it does NOT touch the production Postgres or the live Stripe key in
# dnaty_saas/.env. Run from the repo root:
#
#   ./run_backend_local.ps1
#
# Backend will be at http://localhost:8000 (docs at /docs). The frontend reads
# VITE_API_URL from frontend/.env (defaults to this URL).

$env:DATABASE_URL      = 'sqlite:///./dnaty_dev.db'
$env:JWT_SECRET        = 'local-dev-secret-change-me'
$env:DEVICE_FP_HMAC_KEY = ''

& ".\.venv\Scripts\python.exe" -m uvicorn main:app --app-dir dnaty_saas --port 8000 --reload
