"""Entry point for the ai-dev-console Streamlit app."""

import os
import subprocess
import sys


def main() -> None:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(current_dir, "app.py")

    cmd = [sys.executable, "-m", "streamlit", "run", app_path]

    os.environ["STREAMLIT_SERVER_HEADLESS"] = os.getenv(
        "STREAMLIT_SERVER_HEADLESS", "false"
    )
    os.environ["STREAMLIT_GLOBAL_LOG_LEVEL"] = os.getenv(
        "STREAMLIT_GLOBAL_LOG_LEVEL", "error"
    )
    os.environ["STREAMLIT_GLOBAL_DEVELOPMENT_MODE"] = os.getenv(
        "STREAMLIT_GLOBAL_DEVELOPMENT_MODE", "false"
    )

    subprocess.run(cmd)


if __name__ == "__main__":
    main()
