"""Simple entry point for the CivicFix Flask application."""
from app import app


if __name__ == "__main__":
    app.run(debug=False, port=5000)
