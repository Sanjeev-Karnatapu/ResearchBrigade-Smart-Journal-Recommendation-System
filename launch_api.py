#!/usr/bin/env python3
"""
Render-friendly Journal Recommender API launcher.
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
os.chdir(project_root)


def main() -> int:
    print("Journal Recommender API - Render Launch")
    print("=======================================")
    print(f"Working directory: {project_root}")
    print(f"Python version: {sys.version}")

    try:
        import uvicorn
        from app.main import app

        port = int(os.environ.get("PORT", "8000"))
        print(f"Starting FastAPI on 0.0.0.0:{port}")
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
        return 0
    except Exception as e:
        print(f"Server startup failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
