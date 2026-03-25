#!/usr/bin/env python3
"""
Render-friendly Journal Recommender API launcher.

What this does:
1. Ensures project root is on sys.path
2. Optionally initializes DB / ingests data / builds vectors if needed
3. Performs a quick recommender smoke test
4. Starts FastAPI directly on Render's PORT

Use on Render with:
Start Command: python launch_api.py
"""

import os
import sys
import subprocess
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
os.chdir(project_root)


def print_section(title: str) -> None:
    print(f"\n{title}")
    print("=" * len(title))


def run_command(cmd: str, description: str) -> bool:
    print(f"Running: {description}")
    try:
        subprocess.run(cmd, shell=True, check=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed: {e}")
        return False


def file_exists(rel_path: str) -> bool:
    return (project_root / rel_path).exists()


def check_database_exists() -> bool:
    return (project_root / "journal_recommender.db").exists()


def check_has_data() -> bool:
    try:
        from app.models.base import SessionLocal
        from app.models.entities import Journal

        db = SessionLocal()
        count = db.query(Journal).count()
        db.close()
        return count > 0
    except Exception as e:
        print(f"! Could not check journal data: {e}")
        return False


def check_has_vectors() -> bool:
    try:
        from app.models.base import SessionLocal
        from app.models.entities import JournalProfile

        db = SessionLocal()
        count = db.query(JournalProfile).filter(
            JournalProfile.tfidf_vector.isnot(None),
            JournalProfile.bert_vector.isnot(None)
        ).count()
        db.close()
        return count > 0
    except Exception as e:
        print(f"! Could not check vectors: {e}")
        return False


def main() -> int:
    print("Journal Recommender API - Render Launch")
    print("=======================================")
    print(f"Working directory: {project_root}")
    print(f"Python version: {sys.version}")

    # Step 1: Database init
    if not check_database_exists():
        print_section("Step 1: Database Initialization")
        if file_exists("scripts/init_db.py"):
            if not run_command("python scripts/init_db.py", "Database initialization"):
                return 1
        else:
            print("! scripts/init_db.py not found, skipping database initialization")
    else:
        print_section("Step 1: Database Check")
        print("✓ Database file exists")

    # Step 2: Data ingestion
    if not check_has_data():
        print_section("Step 2: Data Ingestion")
        if file_exists("scripts/ingest_openalex.py"):
            if not run_command("python scripts/ingest_openalex.py", "Data ingestion"):
                return 1
        else:
            print("! scripts/ingest_openalex.py not found, skipping ingestion")
    else:
        print_section("Step 2: Data Check")
        print("✓ Database contains journal data")

    # Step 3: Vector building
    if not check_has_vectors():
        print_section("Step 3: Vector Building")
        if file_exists("scripts/build_vectors.py"):
            if not run_command("python scripts/build_vectors.py", "Vector building"):
                return 1
        else:
            print("! scripts/build_vectors.py not found, skipping vector build")
    else:
        print_section("Step 3: Vector Check")
        print("✓ Journal vectors already exist")

    # Step 4: Quick smoke test
    print_section("Step 4: Recommender Smoke Test")
    try:
        from app.services.recommender import rank_journals

        test_query = (
            "This research investigates machine learning algorithms for "
            "protein structure prediction using deep neural networks."
        )
        results = rank_journals(test_query, top_k=3)
        if results:
            print("✓ Recommender smoke test passed")
        else:
            print("! Recommender returned no results")
    except Exception as e:
        print(f"! Smoke test failed, continuing to start API anyway: {e}")

    # Step 5: Start API server in foreground
    print_section("Step 5: Starting FastAPI Server")
    try:
        import uvicorn
        from app.main import app

        port = int(os.environ.get("PORT", "8000"))
        print(f"✓ Starting server on 0.0.0.0:{port}")
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
        return 0
    except Exception as e:
        print(f"✗ Server startup failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
