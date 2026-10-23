import getpass
import sys
import re
from datetime import datetime, timezone

from werkzeug.security import generate_password_hash
from app import users_db

USERNAME_RE = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")


def main():
    username = input("Admin username: ").strip().lower()

    if not USERNAME_RE.fullmatch(username):
        sys.exit("Username must be 3-30 characters: letters, numbers, dots, underscores, or hyphens.")

    password = getpass.getpass("Password (min 8 chars): ")

    if len(password) < 8:
        sys.exit("Password must be at least 8 characters.")

    confirm_password = getpass.getpass("Confirm password: ")

    if password != confirm_password:
        sys.exit("Passwords do not match.")

    def add(users):
        if username in users:
            return False

        users[username] = {
            "username": username,
            "full_name": "Administrator",
            "role": "admin",
            "password_hash": generate_password_hash(password),
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        return True

    if users_db.update(add):
        print("Admin created successfully!")
    else:
        print("That username already exists.")


if __name__ == "__main__":
    main()
