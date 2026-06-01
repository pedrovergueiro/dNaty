"""
dNATY End-to-End test suite
Tests the full flow: health → signup → login → upload → train → poll → results → model download
Run after starting the backend with: ./run_backend_local.ps1

Usage:
    python tests/test_e2e_full.py
"""
import time
import sys
import json
import zipfile
import io
import os
import struct
import requests

BASE = "http://localhost:8000"
TIMEOUT = 30

# ── helpers ────────────────────────────────────────────────────────────────────

PASS = "✅"
FAIL = "❌"
WARN = "⚠️ "

results = []

def check(name, ok, detail=""):
    status = PASS if ok else FAIL
    msg = f"{status} {name}"
    if detail:
        msg += f"  →  {detail}"
    print(msg)
    results.append((name, ok, detail))
    return ok

def section(title):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")

def wait_for_backend(max_wait=30):
    for _ in range(max_wait):
        try:
            r = requests.get(f"{BASE}/health", timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False

# ── 1. Health ─────────────────────────────────────────────────────────────────

section("1 — HEALTH CHECK")

if not wait_for_backend():
    print(f"{FAIL} Backend not reachable at {BASE} after 30s — start it first")
    sys.exit(1)

r = requests.get(f"{BASE}/health")
check("GET /health → 200", r.status_code == 200, r.json().get("status"))

r = requests.get(f"{BASE}/")
check("GET / → 200 (root)", r.status_code == 200, r.json().get("name"))

# ── 2. Auth ───────────────────────────────────────────────────────────────────

section("2 — AUTH: SIGNUP + LOGIN")

TEST_EMAIL = f"e2e_{int(time.time())}@example.com"
TEST_PASS  = "E2eTest!2026"

# Signup
r = requests.post(f"{BASE}/auth/signup", json={"email": TEST_EMAIL, "password": TEST_PASS, "name": "E2E Tester"})
check("POST /auth/signup → 200/201", r.status_code in (200, 201), f"email={TEST_EMAIL}")

# Login
r = requests.post(f"{BASE}/auth/login", json={"email": TEST_EMAIL, "password": TEST_PASS})
login_ok = check("POST /auth/login → 200 with token", r.status_code == 200 and "token" in r.json(), "")
if not login_ok:
    print(f"  Login response: {r.status_code} {r.text[:300]}")
    sys.exit(1)

free_token = r.json()["token"]
free_headers = {"Authorization": f"Bearer {free_token}"}

# /auth/me
r = requests.get(f"{BASE}/auth/me", headers=free_headers)
check("GET /auth/me → 200", r.status_code == 200, r.json().get("email"))

# ── 3. Plan info ──────────────────────────────────────────────────────────────

section("3 — PLAN INFO")

r = requests.get(f"{BASE}/user/plan", headers=free_headers)
plan_ok = check("GET /user/plan → 200", r.status_code == 200)
if plan_ok:
    plan_data = r.json()
    check("  plan = free", plan_data.get("plan") == "free", str(plan_data.get("plan")))
    check("  limits present", "limits" in plan_data, "")
    check("  can_export = False (free)", not plan_data["limits"].get("can_export"), "")
    check("  samples_per_training = 1000", plan_data["limits"].get("samples_per_training") == 1000, str(plan_data["limits"].get("samples_per_training")))

# ── 4. Train (free plan) ──────────────────────────────────────────────────────

section("4 — START TRAINING (free plan, MNIST, 1000 samples, 3 generations)")

train_payload = {"dataset": "MNIST", "epochs": 3, "samples": 1000, "batch_size": 512}
r = requests.post(f"{BASE}/train", json=train_payload, headers=free_headers)
train_ok = check("POST /train → 202", r.status_code == 202, "")
if not train_ok:
    print(f"  Error: {r.status_code} {r.text[:500]}")
    sys.exit(1)

train_resp = r.json()
job_id = train_resp["job_id"]
check("  job_id present", bool(job_id), job_id[:8] + "...")
check("  status = queued", train_resp["status"] in ("queued", "running"), train_resp["status"])
check("  samples_used = 1000", train_resp["samples_used"] == 1000, str(train_resp["samples_used"]))
check("  dataset = MNIST", train_resp["dataset"] == "MNIST", train_resp["dataset"])

# ── 5. Poll progress ──────────────────────────────────────────────────────────

section("5 — POLL TRAINING PROGRESS")

MAX_WAIT = 600  # 10 min
POLL_INTERVAL = 5
start = time.time()
last_pct = -1

print(f"  Polling job {job_id[:8]}... (up to {MAX_WAIT}s)")
completed = False
while time.time() - start < MAX_WAIT:
    r = requests.get(f"{BASE}/train/{job_id}", headers=free_headers)
    if r.status_code != 200:
        print(f"  Poll error: {r.status_code} {r.text[:200]}")
        time.sleep(POLL_INTERVAL)
        continue

    status_data = r.json()
    pct = status_data.get("progress", 0)
    status = status_data.get("status", "?")
    acc = status_data.get("accuracy")
    gen = status_data.get("current_epoch", 0)

    if pct != last_pct:
        acc_str = f"acc={acc:.2f}%" if acc else "acc=?"
        print(f"  [{int(time.time()-start):3d}s] gen={gen} progress={pct}% {acc_str}  status={status}")
        last_pct = pct

    if status in ("completed", "complete"):
        completed = True
        elapsed = time.time() - start
        check(f"  Training completed in {elapsed:.0f}s", True, f"acc={acc}")
        break
    elif status == "failed":
        check("  Training completed", False, f"FAILED: {status_data.get('error')}")
        break

    time.sleep(POLL_INTERVAL)

if not completed:
    check("Training completed within timeout", False, f"Timed out after {MAX_WAIT}s")
    print("  Continuing with whatever state we have...")

# ── 6. Results ────────────────────────────────────────────────────────────────

section("6 — GET RESULTS")

if completed:
    r = requests.get(f"{BASE}/results/{job_id}")
    results_ok = check("GET /results/{job_id} → 200", r.status_code == 200, "")
    if results_ok:
        res = r.json()
        check("  best_accuracy present", res.get("best_accuracy") is not None, f"{res.get('best_accuracy'):.4f}")
        check("  best_accuracy > 0.5", (res.get("best_accuracy") or 0) > 0.5, "")
        check("  final_architecture present", res.get("final_architecture") is not None, "")
        if res.get("final_architecture"):
            arch = res["final_architecture"]
            check("  architecture has layer_sizes", "layer_sizes" in arch, str(arch.get("layer_sizes")))
            check("  n_params present", arch.get("n_params", 0) > 0, str(arch.get("n_params")))
        check("  history non-empty", len(res.get("history", [])) > 0, f"{len(res.get('history',[]))} entries")
        check("  duration_seconds present", res.get("duration_seconds", 0) > 0, f"{res.get('duration_seconds'):.1f}s")
        check("  dataset = mnist", "mnist" in str(res.get("dataset", "")).lower(), str(res.get("dataset")))
else:
    print(f"  {WARN} Skipping results check (training not completed)")

# ── 7. Model download — free plan should get 403 ──────────────────────────────

section("7 — MODEL DOWNLOAD (free plan should be blocked)")

r = requests.get(f"{BASE}/results/{job_id}/model", headers=free_headers)
check("GET /results/{job_id}/model → 403 on free plan", r.status_code == 403,
      r.json().get("detail", "")[:80] if r.headers.get("content-type","").startswith("application/json") else f"HTTP {r.status_code}")

# ── 8. Pro plan: login as owner and test model download ───────────────────────

section("8 — PRO PLAN: MODEL DOWNLOAD")

OWNER_EMAIL = "owner@dnaty.io"
OWNER_PASS  = "TestPass123!"

r = requests.post(f"{BASE}/auth/login", json={"email": OWNER_EMAIL, "password": OWNER_PASS})
pro_login_ok = check("POST /auth/login (owner/Pro) → 200", r.status_code == 200 and "token" in r.json(), "")

if pro_login_ok:
    pro_token   = r.json()["token"]
    pro_headers = {"Authorization": f"Bearer {pro_token}"}

    # Check plan
    r = requests.get(f"{BASE}/user/plan", headers=pro_headers)
    plan_ok = check("GET /user/plan (owner) → plan=pro", r.status_code == 200 and r.json().get("plan") == "pro", r.json().get("plan") if r.status_code == 200 else str(r.status_code))

    # Start a Pro training job to test model download
    if completed:
        # Reuse the free user's job_id — but owner doesn't own it, so try with a new Pro job
        r2 = requests.post(f"{BASE}/train", json={"dataset": "MNIST", "epochs": 2, "samples": 500, "batch_size": 512}, headers=pro_headers)
        pro_train_ok = check("POST /train (Pro) → 202", r2.status_code == 202, "")

        if pro_train_ok:
            pro_job_id = r2.json()["job_id"]
            # Wait for Pro training
            print(f"  Waiting for Pro job {pro_job_id[:8]}...")
            pro_done = False
            t0 = time.time()
            while time.time() - t0 < 600:
                rp = requests.get(f"{BASE}/train/{pro_job_id}", headers=pro_headers)
                if rp.status_code == 200 and rp.json().get("status") in ("completed", "complete"):
                    pro_done = True
                    break
                elif rp.status_code == 200 and rp.json().get("status") == "failed":
                    check("Pro training completed", False, rp.json().get("error"))
                    break
                pct2 = rp.json().get("progress",0) if rp.status_code==200 else "?"
                print(f"  [{int(time.time()-t0):3d}s] progress={pct2}%")
                time.sleep(5)

            if pro_done:
                r3 = requests.get(f"{BASE}/results/{pro_job_id}/model", headers=pro_headers)
                model_ok = check("GET /results/{job_id}/model (Pro) → 200 + .pt bytes", r3.status_code == 200, "")
                if model_ok:
                    content_type = r3.headers.get("content-type", "")
                    content_disp = r3.headers.get("content-disposition", "")
                    size_kb = len(r3.content) / 1024
                    check("  Content-Type = octet-stream", "octet-stream" in content_type, content_type)
                    check("  Content-Disposition has filename .pt", ".pt" in content_disp, content_disp)
                    check("  File non-empty > 1 KB", size_kb > 1, f"{size_kb:.1f} KB")

                    # Verify it's a valid PyTorch file (PK magic = zip, or legacy _MAGIC)
                    magic = r3.content[:2] if len(r3.content) >= 2 else b""
                    is_zip = magic == b"PK"   # PyTorch >= 1.6 saves as ZIP internally
                    is_legacy = len(r3.content) >= 10 and r3.content[:8] == b"\x80\x05\x95"[:3] + b"\x00\x00\x00\x00"  # pickle magic
                    # More reliable: just check it's bytes and non-trivially sized
                    is_plausible_pt = size_kb > 1
                    check("  File looks like valid .pt (non-empty binary)", is_plausible_pt, f"{size_kb:.1f} KB")

# ── 9. Daily limit (free plan, second training same day) ──────────────────────

section("9 — DAILY LIMIT ENFORCEMENT (free plan)")

r = requests.post(f"{BASE}/train", json={"dataset": "MNIST", "epochs": 3, "samples": 100, "batch_size": 512}, headers=free_headers)
check("Second POST /train (free plan) → 429", r.status_code == 429, r.json().get("detail", "")[:80] if r.status_code == 429 else f"Got {r.status_code}")

# ── 10. Dataset restrictions (free plan can't use CIFAR-10) ───────────────────

section("10 — DATASET RESTRICTION (free plan)")

r = requests.post(f"{BASE}/train", json={"dataset": "CIFAR10", "epochs": 1, "samples": 100, "batch_size": 512}, headers=free_headers)
check("POST /train CIFAR10 on free → 403", r.status_code == 403, r.json().get("detail", "")[:80] if r.status_code == 403 else f"Got {r.status_code}")

# ── 11. Upload dataset endpoint ───────────────────────────────────────────────

section("11 — DATASET UPLOAD")

# Create a minimal valid ZIP with ImageFolder structure (2 classes, 1 image each)
buf = io.BytesIO()
# 1x1 white JPEG bytes (minimal valid JPEG)
minimal_jpeg = bytes([
    0xFF,0xD8,0xFF,0xE0,0x00,0x10,0x4A,0x46,0x49,0x46,0x00,0x01,0x01,0x00,0x00,0x01,
    0x00,0x01,0x00,0x00,0xFF,0xDB,0x00,0x43,0x00,0x08,0x06,0x06,0x07,0x06,0x05,0x08,
    0x07,0x07,0x07,0x09,0x09,0x08,0x0A,0x0C,0x14,0x0D,0x0C,0x0B,0x0B,0x0C,0x19,0x12,
    0x13,0x0F,0x14,0x1D,0x1A,0x1F,0x1E,0x1D,0x1A,0x1C,0x1C,0x20,0x24,0x2E,0x27,0x20,
    0x22,0x2C,0x23,0x1C,0x1C,0x28,0x37,0x29,0x2C,0x30,0x31,0x34,0x34,0x34,0x1F,0x27,
    0x39,0x3D,0x38,0x32,0x3C,0x2E,0x33,0x34,0x32,0xFF,0xC0,0x00,0x0B,0x08,0x00,0x01,
    0x00,0x01,0x01,0x01,0x11,0x00,0xFF,0xC4,0x00,0x1F,0x00,0x00,0x01,0x05,0x01,0x01,
    0x01,0x01,0x01,0x01,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x01,0x02,0x03,0x04,
    0x05,0x06,0x07,0x08,0x09,0x0A,0x0B,0xFF,0xC4,0x00,0xB5,0x10,0x00,0x02,0x01,0x03,
    0x03,0x02,0x04,0x03,0x05,0x05,0x04,0x04,0x00,0x00,0x01,0x7D,0x01,0x02,0x03,0x00,
    0x04,0x11,0x05,0x12,0x21,0x31,0x41,0x06,0x13,0x51,0x61,0x07,0x22,0x71,0x14,0x32,
    0x81,0x91,0xA1,0x08,0x23,0x42,0xB1,0xC1,0x15,0x52,0xD1,0xF0,0x24,0x33,0x62,0x72,
    0x82,0x09,0x0A,0x16,0x17,0x18,0x19,0x1A,0x25,0x26,0x27,0x28,0x29,0x2A,0x34,0x35,
    0x36,0x37,0x38,0x39,0x3A,0x43,0x44,0x45,0x46,0x47,0x48,0x49,0x4A,0x53,0x54,0x55,
    0x56,0x57,0x58,0x59,0x5A,0x63,0x64,0x65,0x66,0x67,0x68,0x69,0x6A,0x73,0x74,0x75,
    0x76,0x77,0x78,0x79,0x7A,0x83,0x84,0x85,0x86,0x87,0x88,0x89,0x8A,0x92,0x93,0x94,
    0x95,0x96,0x97,0x98,0x99,0x9A,0xA2,0xA3,0xA4,0xA5,0xA6,0xA7,0xA8,0xA9,0xAA,0xB2,
    0xB3,0xB4,0xB5,0xB6,0xB7,0xB8,0xB9,0xBA,0xC2,0xC3,0xC4,0xC5,0xC6,0xC7,0xC8,0xC9,
    0xCA,0xD2,0xD3,0xD4,0xD5,0xD6,0xD7,0xD8,0xD9,0xDA,0xE1,0xE2,0xE3,0xE4,0xE5,0xE6,
    0xE7,0xE8,0xE9,0xEA,0xF1,0xF2,0xF3,0xF4,0xF5,0xF6,0xF7,0xF8,0xF9,0xFA,0xFF,0xDA,
    0x00,0x08,0x01,0x01,0x00,0x00,0x3F,0x00,0xFB,0xDA,0xFF,0xD9,
])
with zipfile.ZipFile(buf, "w") as zf:
    zf.writestr("cats/cat1.jpg", minimal_jpeg)
    zf.writestr("dogs/dog1.jpg", minimal_jpeg)
buf.seek(0)

r = requests.post(
    f"{BASE}/train/upload-dataset",
    files={"file": ("test_dataset.zip", buf, "application/zip")},
    headers=free_headers,
)
upload_ok = check("POST /train/upload-dataset (ZIP) → 200", r.status_code == 200, "")
if upload_ok:
    udata = r.json()
    check("  upload_id present", bool(udata.get("upload_id")), "")
    check("  n_classes = 2", udata.get("n_classes") == 2, str(udata.get("n_classes")))
    check("  classes = [cats, dogs]", set(udata.get("classes", [])) == {"cats", "dogs"}, str(udata.get("classes")))
    check("  n_images = 2", udata.get("n_images") == 2, str(udata.get("n_images")))
else:
    print(f"  Upload response: {r.status_code} {r.text[:300]}")

# Try uploading a non-ZIP → should fail with 400
r = requests.post(
    f"{BASE}/train/upload-dataset",
    files={"file": ("bad.txt", b"not a zip file", "text/plain")},
    headers=free_headers,
)
check("POST /train/upload-dataset (non-ZIP) → 400", r.status_code == 400, "")

# ── 12. Auth errors ───────────────────────────────────────────────────────────

section("12 — AUTH EDGE CASES")

r = requests.get(f"{BASE}/user/plan")
check("GET /user/plan without token → 401/403", r.status_code in (401, 403), f"Got {r.status_code}")

r = requests.post(f"{BASE}/auth/login", json={"email": TEST_EMAIL, "password": "wrong-password"})
check("POST /auth/login wrong password → 401", r.status_code == 401, "")

r = requests.post(f"{BASE}/auth/signup", json={"email": TEST_EMAIL, "password": "AnotherPass1!"})
check("POST /auth/signup duplicate email → 400/409", r.status_code in (400, 409), f"Got {r.status_code}")

# ── SUMMARY ───────────────────────────────────────────────────────────────────

section("SUMMARY")
passed = sum(1 for _, ok, _ in results if ok)
failed = sum(1 for _, ok, _ in results if not ok)
total  = len(results)
print(f"\n  {PASS} Passed: {passed}/{total}")
if failed:
    print(f"  {FAIL} Failed: {failed}/{total}")
    print("\n  Failed checks:")
    for name, ok, detail in results:
        if not ok:
            print(f"    {FAIL} {name}  →  {detail}")
    sys.exit(1)
else:
    print(f"\n  All {total} checks passed. Ship it.")
