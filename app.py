
"""Heart Disease Prediction System - Flask web application.

Run:  python train_model.py   (once, creates the model)
      python create_admin.py  (once, creates an admin account)
      python app.py
"""

import json
import os
import re
import secrets
import tempfile
import time
import uuid
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path

import joblib
import pandas as pd
from flask import (
    Flask,
    abort,
    flash,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from werkzeug.security import check_password_hash, generate_password_hash

from model_utils import (
    ALL_FEATURES,
    BASE_DIR,
    DATA_DIR,
    FEATURES,
    METRICS_PATH,
    MODEL_PATH,
    parse_form,
)
from storage import JsonStore

app = Flask(__name__)

# Use writable temporary storage on Vercel.
# On your computer, continue using the existing data directory.
STORE_DIR = Path(
    os.environ.get("HEART_STORE_DIR")
    or (
        Path(tempfile.gettempdir()) / "heart_app_data"
        if os.environ.get("VERCEL")
        else DATA_DIR
    )
)
STORE_DIR.mkdir(parents=True, exist_ok=True)


def _secret_key():
    env = os.environ.get("SECRET_KEY")
    if env:
        return env

    key_file = STORE_DIR / "secret_key.txt"

    if not key_file.exists():
        STORE_DIR.mkdir(parents=True, exist_ok=True)
        key_file.write_text(secrets.token_hex(32))

    return key_file.read_text().strip()


app.config.update(
    SECRET_KEY=_secret_key(),
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.environ.get("HTTPS", "") == "1",
    MAX_CONTENT_LENGTH=64 * 1024,
)

users_db = JsonStore(STORE_DIR / "users.json", dict)
assessments_db = JsonStore(STORE_DIR / "assessments.json", list)

_model = None


def get_model():
    global _model

    if _model is None:
        if not MODEL_PATH.exists():
            raise RuntimeError("Model not found. Run: python train_model.py")

        _model = joblib.load(MODEL_PATH)

    return _model


# ---------------------------------------------------------------- security

@app.before_request
def csrf_protect():
    if "_csrf" not in session:
        session["_csrf"] = secrets.token_hex(16)

    if request.method == "POST":
        sent = request.form.get("_csrf", "")
        if not secrets.compare_digest(sent, session["_csrf"]):
            abort(400, "Invalid or missing CSRF token.")


@app.after_request
def security_headers(resp):
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.context_processor
def inject_globals():
    return {
        "csrf_token": session.get("_csrf", ""),
        "current_user": current_user(),
    }


_failed = {}  # username -> list of failure timestamps (in memory)
MAX_FAILS, WINDOW = 5, 600


def locked_out(username):
    now = time.time()
    _failed[username] = [
        t for t in _failed.get(username, []) if now - t < WINDOW
    ]
    return len(_failed[username]) >= MAX_FAILS


def current_user():
    username = session.get("user")

    if not username:
        return None

    user = users_db.read().get(username)

    if user is None:
        session.clear()

    return user


def login_required(fn):
    @wraps(fn)
    def wrapper(*a, **kw):
        if not current_user():
            flash("Please log in to continue.", "info")
            return redirect(url_for("login"))

        return fn(*a, **kw)

    return wrapper


def role_required(role):
    def deco(fn):
        @wraps(fn)
        def wrapper(*a, **kw):
            user = current_user()

            if not user:
                return redirect(url_for("login"))

            if user["role"] != role:
                abort(403)

            return fn(*a, **kw)

        return wrapper

    return deco


# ---------------------------------------------------------------- public

@app.route("/")
def index():
    user = current_user()

    if user:
        return redirect(
            url_for(
                "admin_home" if user["role"] == "admin" else "dashboard"
            )
        )

    return render_template("index.html")


GMAIL_RE = re.compile(
    r"^[A-Za-z0-9](?:[A-Za-z0-9._%+\-]{0,62}[A-Za-z0-9])?@gmail\.com$",
    re.IGNORECASE,
)


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip().lower()
        full_name = request.form.get("full_name", "").strip()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm", "")

        error = None

        if not full_name or len(full_name) > 80:
            error = "Enter your full name (up to 80 characters)."
        elif not GMAIL_RE.fullmatch(username):
            error = "Enter a valid Gmail address ending in @gmail.com."
        elif len(password) < 8:
            error = "Password must be at least 8 characters."
        elif password != confirm:
            error = "Passwords do not match."

        if not error:
            def add(users):
                if username in users:
                    return False

                users[username] = {
                    "username": username,
                    "email": username,
                    "full_name": full_name,
                    "role": "patient",
                    "password_hash": generate_password_hash(password),
                    "created": datetime.now(timezone.utc).isoformat(
                        timespec="seconds"
                    ),
                }

                return True

            if users_db.update(add):
                flash("Account created. Please log in.", "success")
                return redirect(url_for("login"))

            error = "That username is already taken."

        flash(error, "error")

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip().lower()
        password = request.form.get("password", "")

        if locked_out(username):
            flash(
                "Too many failed attempts. Try again in a few minutes.",
                "error",
            )
        else:
            user = users_db.read().get(username)

            if user and check_password_hash(
                user["password_hash"], password
            ):
                session.clear()
                session["user"] = username
                session["_csrf"] = secrets.token_hex(16)
                _failed.pop(username, None)
                return redirect(url_for("index"))

            _failed.setdefault(username, []).append(time.time())
            flash("Invalid username or password.", "error")

    return render_template("login.html")


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return redirect(url_for("index"))


# ---------------------------------------------------------------- patient

def user_assessments(username):
    rows = [
        a for a in assessments_db.read()
        if a["username"] == username
    ]
    return sorted(rows, key=lambda a: a["created"], reverse=True)


@app.route("/dashboard")
@role_required("patient")
def dashboard():
    user = current_user()
    rows = user_assessments(user["username"])

    return render_template(
        "dashboard.html",
        rows=rows[:5],
        total=len(rows),
    )


@app.route("/assess", methods=["GET", "POST"])
@role_required("patient")
def assess():
    values, errors = {}, {}

    if request.method == "POST":
        values, errors = parse_form(request.form)

        if not errors:
            frame = pd.DataFrame([values])[ALL_FEATURES]
            prob = float(get_model().predict_proba(frame)[0, 1])

            record = {
                "id": uuid.uuid4().hex[:12],
                "username": session["user"],
                "created": datetime.now(timezone.utc).isoformat(
                    timespec="seconds"
                ),
                "inputs": values,
                "probability": round(prob, 4),
                "predicted": int(prob >= 0.5),
            }

            assessments_db.update(lambda data: data.append(record))

            return redirect(
                url_for("result", assessment_id=record["id"])
            )

        values = {
            k: request.form.get(k, "")
            for k in FEATURES
        }

    return render_template(
        "assess.html",
        features=FEATURES,
        values=values,
        errors=errors,
    )


def get_assessment_or_404(assessment_id):
    user = current_user()

    for a in assessments_db.read():
        if a["id"] == assessment_id:
            if (
                user["role"] == "admin"
                or a["username"] == user["username"]
            ):
                return a

            abort(404)

    abort(404)


def display_inputs(inputs):
    rows = []

    for name, spec in FEATURES.items():
        v = inputs[name]
        shown = (
            spec["choices"][int(v)]
            if spec["type"] == "choice"
            else f"{v:g}"
        )
        rows.append((spec["label"], shown))

    return rows


@app.route("/result/<assessment_id>")
@login_required
def result(assessment_id):
    a = get_assessment_or_404(assessment_id)

    return render_template(
        "result.html",
        a=a,
        inputs=display_inputs(a["inputs"]),
    )


@app.route("/history")
@role_required("patient")
def history():
    return render_template(
        "history.html",
        rows=user_assessments(session["user"]),
    )


@app.route("/report/<assessment_id>")
@login_required
def report(assessment_id):
    a = get_assessment_or_404(assessment_id)
    owner = users_db.read().get(a["username"], {})

    return render_template(
        "report.html",
        a=a,
        owner=owner,
        inputs=display_inputs(a["inputs"]),
    )


# ---------------------------------------------------------------- admin

@app.route("/admin")
@role_required("admin")
def admin_home():
    users = users_db.read()
    counts = {}

    for a in assessments_db.read():
        counts[a["username"]] = counts.get(a["username"], 0) + 1

    patients = sorted(
        (u for u in users.values() if u["role"] == "patient"),
        key=lambda u: u["created"],
        reverse=True,
    )

    return render_template(
        "admin_home.html",
        patients=patients,
        counts=counts,
        total=sum(counts.values()),
    )


@app.route("/admin/patient/<username>")
@role_required("admin")
def admin_patient(username):
    patient = users_db.read().get(username)

    if not patient or patient["role"] != "patient":
        abort(404)

    return render_template(
        "admin_patient.html",
        patient=patient,
        rows=user_assessments(username),
    )


@app.route("/admin/model")
@role_required("admin")
def admin_model():
    metrics = (
        json.loads(METRICS_PATH.read_text())
        if METRICS_PATH.exists()
        else None
    )

    return render_template("admin_model.html", m=metrics)


@app.template_filter("pct")
def pct(p):
    return f"{p * 100:.1f}%"


@app.template_filter("when")
def when(iso):
    return datetime.fromisoformat(iso).strftime(
        "%d %b %Y, %H:%M UTC"
    )


@app.errorhandler(403)
def forbidden(_):
    return render_template(
        "error.html",
        code=403,
        msg="You do not have access to this page.",
    ), 403


@app.errorhandler(404)
def not_found(_):
    return render_template(
        "error.html",
        code=404,
        msg="Page not found.",
    ), 404


@app.errorhandler(400)
def bad_request(e):
    return render_template(
        "error.html",
        code=400,
        msg=getattr(e, "description", "Bad request."),
    ), 400


if __name__ == "__main__":
    app.run(
        debug=os.environ.get("FLASK_DEBUG") == "1",
        port=int(os.environ.get("PORT", 5000)),
    )
