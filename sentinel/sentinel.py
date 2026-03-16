"""
SYNAPSE Sentinel — Lightweight watchdog for Main SYNAPSE.

Monitors health, diagnoses errors via Gemini AI, pushes fixes to GitHub,
triggers Cloud Build redeployment, and rolls back if the fix fails.

Architecture:
  Sentinel (this) ──watches──▸ Main SYNAPSE
  Sentinel ──diagnoses──▸ Gemini AI
  Sentinel ──pushes fix──▸ GitHub
  Sentinel ──triggers──▸ Cloud Build → Cloud Run
  Sentinel ──rollback──▸ Cloud Run (traffic shift)
"""

import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from datetime import datetime, timezone

import requests as http_requests
from flask import Flask, jsonify

# ── Optional GCP imports (only available in Cloud Run) ───────────
try:
    from google.cloud.devtools import cloudbuild_v1
    _cloudbuild_available = True
except ImportError:
    _cloudbuild_available = False

try:
    from google.cloud import run_v2
    _cloudrun_available = True
except ImportError:
    _cloudrun_available = False

try:
    import google.genai as genai
    _genai_available = True
except ImportError:
    _genai_available = False


# ── Configuration ────────────────────────────────────────────────

MAIN_URL = os.environ.get("MAIN_URL", "http://localhost:9090")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
GITHUB_REPO = os.environ.get("GITHUB_REPO", "bxf1001g/SYNAPSE")
GCP_PROJECT = os.environ.get("GCP_PROJECT", "synapse-490213")
GCP_REGION = os.environ.get("GCP_REGION", "us-central1")
MAIN_SERVICE = os.environ.get("MAIN_SERVICE", "synapse")
CLOUD_MODE = os.environ.get("SYNAPSE_CLOUD_MODE", "") == "1"

CHECK_INTERVAL = int(os.environ.get("CHECK_INTERVAL", "300"))    # 5 min
ERROR_THRESHOLD = int(os.environ.get("ERROR_THRESHOLD", "5"))
CONFIDENCE_MIN = float(os.environ.get("CONFIDENCE_MIN", "0.7"))
HEAL_COOLDOWN = int(os.environ.get("HEAL_COOLDOWN", "1800"))     # 30 min
DEPLOY_WAIT = int(os.environ.get("DEPLOY_WAIT", "240"))          # 4 min


# ── State ────────────────────────────────────────────────────────

_sentinel_log = []
_SENTINEL_LOG_MAX = 100
_last_heal_time = 0
_monitor_active = False
_booted = False

app = Flask(__name__)


# ── Logging helper ───────────────────────────────────────────────

def _log(action, detail=""):
    entry = {
        "time": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "detail": str(detail)[:500],
    }
    _sentinel_log.append(entry)
    if len(_sentinel_log) > _SENTINEL_LOG_MAX:
        _sentinel_log.pop(0)
    print(f"[SENTINEL] {action}: {detail}", flush=True)


# ── Health & Status endpoints ────────────────────────────────────

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "role": "sentinel",
        "monitoring": MAIN_URL,
        "cloud_mode": CLOUD_MODE,
    })


@app.route("/api/sentinel/status")
def sentinel_status():
    return jsonify({
        "active": _monitor_active,
        "monitoring": MAIN_URL,
        "check_interval": CHECK_INTERVAL,
        "error_threshold": ERROR_THRESHOLD,
        "confidence_min": CONFIDENCE_MIN,
        "heal_cooldown": HEAL_COOLDOWN,
        "last_heal_time": (
            datetime.fromtimestamp(_last_heal_time, tz=timezone.utc).isoformat()
            if _last_heal_time else None
        ),
        "log": _sentinel_log[-30:],
    })


@app.route("/api/sentinel/log")
def sentinel_log():
    return jsonify(_sentinel_log[-50:])


@app.route("/api/sentinel/trigger", methods=["POST"])
def trigger_heal():
    """Manually trigger one heal cycle (bypasses cooldown)."""
    global _last_heal_time
    _last_heal_time = 0
    threading.Thread(target=_run_heal_cycle, daemon=True).start()
    return jsonify({"status": "triggered"})


# ── Monitoring functions ─────────────────────────────────────────

def _check_main_health():
    """Ping Main SYNAPSE /health endpoint."""
    try:
        r = http_requests.get(f"{MAIN_URL}/health", timeout=15)
        return r.status_code == 200 and r.json().get("status") == "ok"
    except Exception as e:
        _log("health_check_failed", str(e))
        return False


def _fetch_diagnostics():
    """Fetch error log + recommendations from Main."""
    try:
        r = http_requests.get(f"{MAIN_URL}/api/diagnostics", timeout=15)
        if r.status_code == 200:
            return r.json()
        return None
    except Exception as e:
        _log("fetch_diagnostics_failed", str(e))
        return None


def _fetch_healing_status():
    """Fetch healing status from Main."""
    try:
        r = http_requests.get(f"{MAIN_URL}/api/healing", timeout=15)
        if r.status_code == 200:
            return r.json()
        return None
    except Exception as e:
        _log("fetch_healing_failed", str(e))
        return None


# ── AI Diagnosis ─────────────────────────────────────────────────

def _diagnose_with_ai(diagnostics, healing_status):
    """Call Gemini AI to diagnose errors and generate a fix."""
    if not GEMINI_API_KEY or not _genai_available:
        _log("no_ai", "Gemini API key or SDK not available")
        return None

    errors = diagnostics.get("errors", [])
    recommendations = diagnostics.get("recommendations", [])
    history = (healing_status or {}).get("history", [])

    error_text = json.dumps(errors[-20:], indent=2)
    rec_text = json.dumps(recommendations, indent=2)
    hist_text = json.dumps(history[-5:], indent=2)

    prompt = (
        "You are SYNAPSE Sentinel, an independent watchdog monitoring SYNAPSE AI.\n"
        "Analyze these errors from the Main SYNAPSE instance and generate a precise fix.\n\n"
        f"RECENT ERRORS ({len(errors)} total, showing last 20):\n{error_text}\n\n"
        f"AUTO-RECOMMENDATIONS:\n{rec_text}\n\n"
        f"RECENT HEALING HISTORY:\n{hist_text}\n\n"
        "IMPORTANT RULES:\n"
        "1. Only fix errors you are CONFIDENT about (>0.7 confidence).\n"
        "2. Provide EXACT search/replace pairs — the search text must match "
        "exactly one occurrence in the file.\n"
        "3. NEVER modify API keys, secrets, or security-critical code.\n"
        "4. If the issue is transient (network blip, rate limit), respond "
        'with fix_type "no_fix".\n'
        "5. If a simple restart would fix it, use fix_type \"restart\".\n"
        "6. The main code file is agent_ui.py (~5700 lines, Flask backend).\n\n"
        "Respond with ONLY valid JSON (no markdown fences):\n"
        "{\n"
        '  "diagnosis": "root cause description",\n'
        '  "confidence": 0.0-1.0,\n'
        '  "fix_type": "code_change" | "config_change" | "restart" | "no_fix",\n'
        '  "files": [\n'
        '    {"path": "agent_ui.py", "search": "exact text", '
        '"replace": "replacement text"}\n'
        '  ],\n'
        '  "reason": "why this fixes the errors"\n'
        "}\n"
    )

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        resp = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        text = resp.text.strip()
        # Strip markdown code fences
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        fix = json.loads(text)
        _log("ai_diagnosis", f"Type={fix.get('fix_type')}, "
             f"Confidence={fix.get('confidence')}, "
             f"Diagnosis={str(fix.get('diagnosis', ''))[:200]}")
        return fix
    except Exception as e:
        _log("ai_error", str(e))
        return None


# ── Fix Application ──────────────────────────────────────────────

def _apply_and_push(fix_data):
    """Clone repo, apply search/replace fix, push to heal branch, create PR."""
    files = fix_data.get("files", [])
    if not files:
        _log("no_files", "AI returned no file changes")
        return None

    if not GITHUB_TOKEN:
        _log("no_github_token", "Cannot push without GitHub token")
        return None

    tmp_dir = tempfile.mkdtemp(prefix="sentinel_heal_")
    branch = f"sentinel-heal-{int(time.time())}"

    try:
        repo_url = f"https://{GITHUB_TOKEN}@github.com/{GITHUB_REPO}.git"

        # Clone
        _run_git(["git", "clone", "--depth=1", repo_url, tmp_dir])

        # Apply fixes
        modified = []
        for f in files:
            path = f.get("path", "")
            search = f.get("search", "")
            replace = f.get("replace", "")
            if not path or not search:
                continue

            full_path = os.path.join(tmp_dir, path)
            if not os.path.exists(full_path):
                _log("file_not_found", path)
                continue

            with open(full_path, "r", encoding="utf-8") as fh:
                content = fh.read()

            count = content.count(search)
            if count != 1:
                _log("search_mismatch", f"{path}: found {count} matches")
                continue

            new_content = content.replace(search, replace, 1)

            # Validate Python syntax
            if path.endswith(".py"):
                try:
                    compile(new_content, path, "exec")
                except SyntaxError as e:
                    _log("syntax_error", f"{path}: {e}")
                    continue

            with open(full_path, "w", encoding="utf-8") as fh:
                fh.write(new_content)
            modified.append(path)

        if not modified:
            _log("no_modifications", "All fixes failed validation")
            return None

        # Create branch, commit, push
        _run_git(["git", "-C", tmp_dir, "checkout", "-b", branch])
        _run_git(["git", "-C", tmp_dir, "add", "-A"])
        commit_msg = (
            f"sentinel-heal: {fix_data.get('diagnosis', 'auto-fix')[:80]}\n\n"
            f"Confidence: {fix_data.get('confidence', 0)}\n"
            f"Files: {', '.join(modified)}\n\n"
            "Co-authored-by: SYNAPSE-Sentinel <sentinel@synapse.ai>"
        )
        _run_git(["git", "-C", tmp_dir, "commit", "-m", commit_msg])
        _run_git(["git", "-C", tmp_dir, "push", "origin", branch])

        _log("fix_pushed", f"Branch: {branch}, Files: {modified}")

        # Create PR via GitHub API
        pr_url = _create_pr(branch, fix_data, modified)

        # Auto-merge the PR to trigger deployment
        if pr_url:
            _merge_pr(branch)

        return branch

    except Exception as e:
        _log("apply_error", str(e))
        return None
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _run_git(cmd):
    """Run a git command, raise on failure."""
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=120
    )
    if result.returncode != 0:
        raise RuntimeError(f"Git failed: {result.stderr[:300]}")
    return result.stdout


def _create_pr(branch, fix_data, modified):
    """Create a GitHub Pull Request."""
    try:
        headers = {
            "Authorization": f"token {GITHUB_TOKEN}",
            "Accept": "application/vnd.github.v3+json",
        }
        body = (
            "## 🩺 Automated Self-Healing by SYNAPSE Sentinel\n\n"
            f"**Diagnosis:** {fix_data.get('diagnosis', 'N/A')}\n"
            f"**Confidence:** {fix_data.get('confidence', 0)}\n"
            f"**Fix Type:** {fix_data.get('fix_type', 'N/A')}\n"
            f"**Files Modified:** {', '.join(modified)}\n"
            f"**Reason:** {fix_data.get('reason', 'N/A')}\n\n"
            "---\n"
            "*This PR was created and auto-merged by SYNAPSE Sentinel.*"
        )
        r = http_requests.post(
            f"https://api.github.com/repos/{GITHUB_REPO}/pulls",
            json={
                "title": f"🩺 Sentinel Heal: "
                         f"{fix_data.get('diagnosis', 'auto-fix')[:60]}",
                "body": body,
                "head": branch,
                "base": "main",
            },
            headers=headers,
            timeout=15,
        )
        if r.status_code == 201:
            url = r.json().get("html_url", "")
            _log("pr_created", url)
            return url
        else:
            _log("pr_failed", f"{r.status_code}: {r.text[:200]}")
    except Exception as e:
        _log("pr_error", str(e))
    return None


def _merge_pr(branch):
    """Merge the heal branch into main via GitHub API."""
    try:
        headers = {
            "Authorization": f"token {GITHUB_TOKEN}",
            "Accept": "application/vnd.github.v3+json",
        }
        # Merge branch into main
        r = http_requests.post(
            f"https://api.github.com/repos/{GITHUB_REPO}/merges",
            json={
                "base": "main",
                "head": branch,
                "commit_message": f"Merge sentinel-heal: {branch}",
            },
            headers=headers,
            timeout=15,
        )
        if r.status_code in (201, 204):
            _log("branch_merged", f"{branch} → main")
            return True
        else:
            _log("merge_failed", f"{r.status_code}: {r.text[:200]}")
    except Exception as e:
        _log("merge_error", str(e))
    return False


# ── Cloud Build Trigger ──────────────────────────────────────────

def _trigger_cloud_build():
    """Trigger Cloud Build to rebuild and redeploy Main SYNAPSE."""
    if not CLOUD_MODE:
        _log("deploy_skipped", "Not in cloud mode — skipping Cloud Build")
        return True  # Pretend success for local testing

    if not _cloudbuild_available:
        _log("deploy_error", "google-cloud-build not installed")
        return False

    try:
        client = cloudbuild_v1.CloudBuildClient()

        build = cloudbuild_v1.Build(
            steps=[
                # Build Docker image
                cloudbuild_v1.BuildStep(
                    name="gcr.io/cloud-builders/docker",
                    args=[
                        "build", "-t",
                        f"gcr.io/{GCP_PROJECT}/{MAIN_SERVICE}", ".",
                    ],
                ),
                # Push image
                cloudbuild_v1.BuildStep(
                    name="gcr.io/cloud-builders/docker",
                    args=[
                        "push",
                        f"gcr.io/{GCP_PROJECT}/{MAIN_SERVICE}",
                    ],
                ),
                # Deploy to Cloud Run
                cloudbuild_v1.BuildStep(
                    name="gcr.io/google.com/cloudsdktool/cloud-sdk",
                    entrypoint="bash",
                    args=[
                        "-c",
                        f"gcloud run deploy {MAIN_SERVICE} "
                        f"--image=gcr.io/{GCP_PROJECT}/{MAIN_SERVICE} "
                        f"--region={GCP_REGION} "
                        "--platform=managed "
                        "--allow-unauthenticated "
                        "--timeout=300 "
                        "--memory=1Gi --cpu=1 "
                        "--min-instances=1 --max-instances=1 "
                        "--startup-probe httpGet.path=/health "
                        f"--set-env-vars=SYNAPSE_CLOUD_MODE=1,"
                        f"GCP_PROJECT={GCP_PROJECT},"
                        f"GEMINI_API_KEY={GEMINI_API_KEY},"
                        f"GITHUB_TOKEN={GITHUB_TOKEN}",
                    ],
                ),
            ],
            # Source from the GitHub repo (main branch, post-merge)
            source=cloudbuild_v1.Source(
                repo_source=cloudbuild_v1.RepoSource(
                    project_id=GCP_PROJECT,
                    repo_name=GITHUB_REPO.split("/")[-1],
                    branch_name="main",
                ),
            ),
            images=[f"gcr.io/{GCP_PROJECT}/{MAIN_SERVICE}"],
            options=cloudbuild_v1.BuildOptions(
                logging=cloudbuild_v1.BuildOptions.LoggingMode.CLOUD_LOGGING_ONLY,
            ),
        )

        operation = client.create_build(
            project_id=GCP_PROJECT, build=build
        )
        _log("build_submitted", f"Build ID: {operation.metadata.build.id}")

        # Wait for completion (up to 10 min)
        result = operation.result(timeout=600)
        status = result.status.name
        _log("build_complete", f"Status: {status}")
        return status == "SUCCESS"

    except Exception as e:
        _log("build_error", str(e))
        return False


# ── Rollback ─────────────────────────────────────────────────────

def _get_previous_revision():
    """Get the previous Cloud Run revision name."""
    if not CLOUD_MODE or not _cloudrun_available:
        return None
    try:
        client = run_v2.RevisionsClient()
        parent = (
            f"projects/{GCP_PROJECT}/locations/{GCP_REGION}"
            f"/services/{MAIN_SERVICE}"
        )
        revisions = list(client.list_revisions(parent=parent))
        # Sort by creation time descending
        revisions.sort(key=lambda r: r.create_time, reverse=True)
        if len(revisions) >= 2:
            return revisions[1].name
        return None
    except Exception as e:
        _log("revision_list_error", str(e))
        return None


def _rollback():
    """Rollback Main SYNAPSE to the previous Cloud Run revision."""
    if not CLOUD_MODE:
        _log("rollback_skipped", "Not in cloud mode")
        return False

    prev = _get_previous_revision()
    if not prev:
        _log("rollback_failed", "No previous revision found")
        return False

    try:
        if _cloudrun_available:
            client = run_v2.ServicesClient()
            service_name = (
                f"projects/{GCP_PROJECT}/locations/{GCP_REGION}"
                f"/services/{MAIN_SERVICE}"
            )
            service = client.get_service(name=service_name)

            # Shift 100% traffic to previous revision
            rev_short = prev.split("/")[-1]
            service.traffic = [
                run_v2.TrafficTarget(
                    revision=rev_short,
                    percent=100,
                    type_=run_v2.TrafficTargetAllocationType.TRAFFIC_TARGET_ALLOCATION_TYPE_REVISION,
                )
            ]
            client.update_service(service=service)
            _log("rollback_success", f"Traffic → {rev_short}")
            return True
    except Exception as e:
        _log("rollback_error", str(e))

    # Fallback: revert git merge
    _log("rollback_git_revert", "Attempting git revert of last merge")
    try:
        tmp_dir = tempfile.mkdtemp(prefix="sentinel_revert_")
        repo_url = f"https://{GITHUB_TOKEN}@github.com/{GITHUB_REPO}.git"
        _run_git(["git", "clone", "--depth=3", repo_url, tmp_dir])
        _run_git(["git", "-C", tmp_dir, "revert", "--no-edit", "HEAD"])
        _run_git(["git", "-C", tmp_dir, "push", "origin", "main"])
        _log("git_revert_pushed", "Reverted HEAD on main")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        # Trigger rebuild with reverted code
        return _trigger_cloud_build()
    except Exception as e:
        _log("git_revert_failed", str(e))
        return False


# ── Main Heal Cycle ──────────────────────────────────────────────

def _run_heal_cycle():
    """Execute one full heal cycle: check → diagnose → fix → deploy → verify."""
    global _last_heal_time

    # Cooldown check
    elapsed = time.time() - _last_heal_time
    if elapsed < HEAL_COOLDOWN:
        remaining = int(HEAL_COOLDOWN - elapsed)
        _log("cooldown", f"Next heal in {remaining}s")
        return

    _log("cycle_start", "Beginning health check")

    # Phase 1: Check Main health
    main_healthy = _check_main_health()
    if main_healthy:
        _log("main_healthy", "Main is responding OK")
    else:
        _log("main_down", "Main not responding!")

    # Phase 2: Fetch diagnostics
    diagnostics = _fetch_diagnostics()
    healing_status = _fetch_healing_status()

    if not diagnostics and main_healthy:
        _log("cycle_end", "No diagnostics available, Main healthy")
        return

    # If Main is down, create synthetic diagnostic data
    if not diagnostics:
        diagnostics = {
            "errors": [{
                "time": datetime.now(timezone.utc).isoformat(),
                "category": "health",
                "message": "Main SYNAPSE not responding to /health",
            }],
            "recommendations": [],
        }

    # Phase 3: Check error threshold
    errors = diagnostics.get("errors", [])
    error_count = len(errors)

    if error_count < ERROR_THRESHOLD and main_healthy:
        _log("below_threshold",
             f"{error_count} errors (need {ERROR_THRESHOLD})")
        return

    _log("threshold_exceeded",
         f"{error_count} errors, proceeding to AI diagnosis")

    # Phase 4: AI Diagnosis
    fix = _diagnose_with_ai(diagnostics, healing_status)
    if not fix:
        _log("no_diagnosis", "AI returned no fix")
        return

    confidence = fix.get("confidence", 0)
    fix_type = fix.get("fix_type", "no_fix")

    # Phase 5: Confidence gate
    if confidence < CONFIDENCE_MIN:
        _log("low_confidence",
             f"Confidence {confidence} < {CONFIDENCE_MIN}, skipping")
        _last_heal_time = time.time()
        return

    if fix_type == "no_fix":
        _log("no_fix_needed", fix.get("diagnosis", "Transient issue"))
        _last_heal_time = time.time()
        return

    # Phase 6: Apply fix
    if fix_type == "restart":
        _log("restart_requested", "Triggering redeploy (no code changes)")
        if _trigger_cloud_build():
            _last_heal_time = time.time()
            _log("restart_triggered", "Cloud Build submitted for restart")
        return

    # Code or config change
    _log("applying_fix", f"Type: {fix_type}, Files: "
         f"{[f.get('path') for f in fix.get('files', [])]}")

    branch = _apply_and_push(fix)
    if not branch:
        _log("fix_failed", "Could not apply or push fix")
        _last_heal_time = time.time()
        return

    # Phase 7: Trigger redeploy
    _log("deploying", "Triggering Cloud Build for new version")
    deployed = _trigger_cloud_build()

    if not deployed:
        _log("deploy_failed", "Cloud Build failed")
        _last_heal_time = time.time()
        return

    # Phase 8: Verify deployment
    _log("verifying", f"Waiting {DEPLOY_WAIT}s for deployment to stabilize")
    time.sleep(DEPLOY_WAIT)

    if _check_main_health():
        _log("heal_success", "✅ New deployment is healthy!")
        _last_heal_time = time.time()
    else:
        _log("heal_failed", "❌ New deployment unhealthy — rolling back")
        _rollback()
        _last_heal_time = time.time()


# ── Monitor Loop ─────────────────────────────────────────────────

def _monitor_loop():
    """Background loop: check Main health every CHECK_INTERVAL."""
    global _monitor_active
    _monitor_active = True
    _log("monitor_started",
         f"Watching {MAIN_URL} every {CHECK_INTERVAL}s "
         f"(threshold={ERROR_THRESHOLD}, cooldown={HEAL_COOLDOWN}s)")

    while _monitor_active:
        time.sleep(CHECK_INTERVAL)
        try:
            _run_heal_cycle()
        except Exception as e:
            _log("monitor_error", str(e))


# ── Boot ─────────────────────────────────────────────────────────

@app.before_request
def _boot():
    global _booted
    if _booted:
        return
    _booted = True
    t = threading.Thread(target=_monitor_loop, daemon=True)
    t.start()
    _log("boot", f"Sentinel v1.0 started — monitoring {MAIN_URL}")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8081"))
    app.run(host="0.0.0.0", port=port, debug=False)
