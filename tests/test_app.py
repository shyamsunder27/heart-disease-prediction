"""End-to-end smoke tests. Run from the project folder:  python -m unittest discover -s tests -v"""
import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
_tmp = tempfile.mkdtemp()
os.environ["HEART_STORE_DIR"] = _tmp  # keep test data out of the real data folder

import app as heart  # noqa: E402
from werkzeug.security import generate_password_hash  # noqa: E402

# A high-risk and a low-risk example (values follow the dataset coding in model_utils.py).
HIGH = dict(age=62, sex=1, cp=0, trestbps=150, chol=290, fbs=1, restecg=0, thalach=105,
            exang=1, oldpeak=3.2, slope=1, ca=2, thal=3)
LOW = dict(age=35, sex=0, cp=2, trestbps=115, chol=190, fbs=0, restecg=1, thalach=180,
           exang=0, oldpeak=0.0, slope=2, ca=0, thal=2)


class Client:
    def __init__(self):
        self.c = heart.app.test_client()

    def token(self, path="/login"):
        html = self.c.get(path).get_data(as_text=True)
        return re.search(r'name="_csrf" value="([^"]+)"', html).group(1)

    def post(self, path, data, form_page="/login", follow=False):
        data = dict(data, _csrf=self.token(form_page))
        return self.c.post(path, data=data, follow_redirects=follow)

    def login(self, user, pw):
        return self.post("/login", {"username": user, "password": pw})


class AppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        heart.users_db.update(lambda u: u.update({"boss": {
            "username": "boss", "full_name": "Admin", "role": "admin",
            "password_hash": generate_password_hash("adminpass1"),
            "created": "2026-01-01T00:00:00+00:00"}}))

    def register(self, cl, name):
        email = f"{name}@gmail.com"
        r = cl.post("/register", {"full_name": name.title(), "username": email,
                                  "password": "password123", "confirm": "password123"},
                    form_page="/register")
        self.assertEqual(r.status_code, 302)

    def predict(self, cl, values):
        r = cl.post("/assess", values, form_page="/assess")
        self.assertEqual(r.status_code, 302, r.get_data(as_text=True)[:300])
        return r.headers["Location"].rsplit("/", 1)[1]

    def test_csrf_required(self):
        r = heart.app.test_client().post("/login", data={"username": "a", "password": "b"})
        self.assertEqual(r.status_code, 400)

    def test_full_patient_flow_and_isolation(self):
        alice, bob = Client(), Client()
        self.register(alice, "alice")
        self.register(bob, "bob")
        self.assertEqual(alice.login("alice@gmail.com", "password123").status_code, 302)
        self.assertEqual(bob.login("bob@gmail.com", "password123").status_code, 302)

        hi = self.predict(alice, HIGH)
        lo = self.predict(alice, LOW)
        records = {a["id"]: a for a in heart.assessments_db.read()}
        self.assertGreater(records[hi]["probability"], 0.5)
        self.assertLess(records[lo]["probability"], 0.5)

        self.assertIn("Assessment result", alice.c.get(f"/result/{hi}").get_data(as_text=True))
        self.assertIn("Printable", alice.c.get(f"/result/{hi}").get_data(as_text=True))
        self.assertEqual(alice.c.get(f"/report/{hi}").status_code, 200)
        self.assertEqual(alice.c.get("/history").get_data(as_text=True).count("View"), 2)

        # Bob cannot see Alice's records, nor the admin area.
        self.assertEqual(bob.c.get(f"/result/{hi}").status_code, 404)
        self.assertEqual(bob.c.get(f"/report/{hi}").status_code, 404)
        self.assertEqual(bob.c.get("/admin").status_code, 403)

    def test_validation_rejects_bad_input(self):
        cl = Client()
        bad_email = cl.post("/register", {"full_name": "Bad Email", "username": "notgmail@yahoo.com",
                                           "password": "password123", "confirm": "password123"},
                            form_page="/register")
        self.assertEqual(bad_email.status_code, 200)
        self.assertIn("ending in @gmail.com", bad_email.get_data(as_text=True))
        self.register(cl, "carol")
        cl.login("carol@gmail.com", "password123")
        bad = dict(HIGH, age=500, cp=9)
        r = cl.post("/assess", bad, form_page="/assess")
        self.assertEqual(r.status_code, 200)
        html = r.get_data(as_text=True)
        self.assertIn("between 18 and 100", html)
        self.assertIn("Choose one of the listed options", html)

    def test_admin_access(self):
        patient = Client()
        self.register(patient, "dave")
        patient.login("dave@gmail.com", "password123")
        rec = self.predict(patient, HIGH)

        admin = Client()
        admin.login("boss", "adminpass1")
        html = admin.c.get("/admin").get_data(as_text=True)
        self.assertIn("dave", html)
        self.assertEqual(admin.c.get("/admin/patient/dave@gmail.com").status_code, 200)
        self.assertEqual(admin.c.get(f"/report/{rec}").status_code, 200)
        self.assertEqual(admin.c.get("/admin/model").status_code, 200)
        self.assertEqual(admin.c.get("/assess").status_code, 403)  # admin is not a patient

    def test_password_is_hashed_and_login_lockout(self):
        users = heart.users_db.read()
        self.assertTrue(users["boss"]["password_hash"].startswith(("scrypt:", "pbkdf2:")))
        cl = Client()
        for _ in range(5):
            cl.login("nobody@gmail.com", "wrong")
        r = cl.login("nobody@gmail.com", "wrong")  # 6th attempt is blocked and rendered directly
        self.assertIn("Too many failed attempts", r.get_data(as_text=True))

    def test_logged_out_redirects(self):
        for path in ["/dashboard", "/assess", "/history", "/admin"]:
            self.assertEqual(heart.app.test_client().get(path).status_code, 302)


if __name__ == "__main__":
    unittest.main()
