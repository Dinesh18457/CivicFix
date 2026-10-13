"""Shared data access and backwards-compatible training entry point."""
from train_models import DATA_PATH, load_data, train


def load_and_preprocess_data():
    """Return valid, resolved cases with the legacy resolution_time column."""
    frame = load_data()
    frame = frame.dropna(subset=["resolution_hours"]).copy()
    frame["resolution_time"] = frame["resolution_hours"]
    frame["created_hour"] = frame["created_date"].dt.hour
    return frame


def train_models():
    """Delegate to the project's single reproducible training workflow."""
    train()


if __name__ == "__main__":
    train_models()
