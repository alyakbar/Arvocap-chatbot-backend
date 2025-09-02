#!/usr/bin/env python3
"""
trainer: A CLI to scrape a site, process text, and rebuild the knowledge base.

Examples (PowerShell, from python_training/):
  .\env\Scripts\Activate.ps1
  python trainer.py run --url https://example.com --max-pages 30 --create-training-file

Subcommands:
  run       Full pipeline (scrape -> process -> retrain -> optional training JSONL)
  scrape    Only scrape to data/scraped_data.json
  process   Only process scraped_data.json -> processed_data.json
  retrain   Only rebuild KB from processed_data.json
  serve     Start the FastAPI server (api_server.py)
  status    Show knowledge base stats
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from typing import Any, Dict

from web_scraper import WebScraper
from text_processor import TextProcessor
from vector_database import ChatbotKnowledgeBase
from chatbot_trainer import ChatbotTrainer


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("trainer")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def cmd_run(args: argparse.Namespace) -> None:
    out_dir = os.path.abspath(args.out_dir)
    ensure_dir(out_dir)
    scraped_path = os.path.join(out_dir, "scraped_data.json")
    processed_path = os.path.join(out_dir, "processed_data.json")
    training_jsonl_path = os.path.join(out_dir, "openai_training.jsonl")

    total_start = time.time()
    logger.info("Starting full pipeline")

    # Scrape
    scrape_start = time.time()
    scraper = WebScraper(use_selenium=args.selenium)
    try:
        logger.info(f"Scraping from {args.url} (max_pages={args.max_pages}, selenium={args.selenium})")
        scraped_data = scraper.scrape_website(args.url, max_pages=args.max_pages, same_domain_only=True)
        if not scraped_data:
            logger.warning("No pages scraped. Check the URL or increase --max-pages.")
        save_json(scraped_path, scraped_data)
        logger.info(f"Saved scraped data to {scraped_path} ({len(scraped_data)} pages)")
    finally:
        try:
            scraper.close()
        except Exception:
            pass
    logger.info(f"Scraping done in {time.time() - scrape_start:.1f}s")

    # Process
    proc_start = time.time()
    processor = TextProcessor()
    logger.info("Processing scraped data (clean, chunk, keywords, Q&A, clusters)...")
    processed_data = processor.process_scraped_data(scraped_data)
    save_json(processed_path, processed_data)
    logger.info(
        "Processed -> documents=%d, qa_pairs=%d, unique_keywords=%d",
        len(processed_data.get("documents", [])),
        len(processed_data.get("qa_pairs", [])),
        len(processed_data.get("keywords", [])),
    )
    logger.info(f"Saved processed data to {processed_path}")
    logger.info(f"Processing done in {time.time() - proc_start:.1f}s")

    # Retrain KB
    if not args.skip_retrain:
        retrain_start = time.time()
        kb = ChatbotKnowledgeBase()
        logger.info("Rebuilding vector database from processed data...")
        report: Dict[str, Any] = kb.retrain(processed_path)
        if not report.get("success", False):
            logger.error(f"Retrain failed: {report.get('error', 'unknown error')}")
            sys.exit(2)
        logger.info(f"Retrain report: {report}")
        logger.info(f"Retrain done in {time.time() - retrain_start:.1f}s")
    else:
        logger.info("Skipping KB retrain as requested")

    # Optional: OpenAI training JSONL
    if args.create_training_file:
        train_start = time.time()
        logger.info("Preparing OpenAI training JSONL from processed data...")
        trainer = ChatbotTrainer(use_openai=True)
        examples = trainer.prepare_training_data(processed_path)
        if examples:
            created = trainer.create_openai_training_file(examples, output_file=training_jsonl_path)
            if created:
                logger.info(f"Training file created at {training_jsonl_path}")
            else:
                logger.warning("Failed to create training JSONL file")
        else:
            logger.warning("No training examples generated from processed data")
        logger.info(f"Training-file prep done in {time.time() - train_start:.1f}s")

    logger.info(f"Pipeline complete in {time.time() - total_start:.1f}s")
    if args.start_api:
        cmd_serve(args)
    else:
        logger.info("Next: run the API server to use the updated knowledge base:")
        logger.info("  python api_server.py")


def cmd_scrape(args: argparse.Namespace) -> None:
    out_dir = os.path.abspath(args.out_dir)
    ensure_dir(out_dir)
    scraped_path = os.path.join(out_dir, "scraped_data.json")

    scraper = WebScraper(use_selenium=args.selenium)
    try:
        logger.info(f"Scraping from {args.url} (max_pages={args.max_pages}, selenium={args.selenium})")
        data = scraper.scrape_website(args.url, max_pages=args.max_pages, same_domain_only=True)
        save_json(scraped_path, data)
        logger.info(f"Saved scraped data to {scraped_path} ({len(data)} pages)")
    finally:
        try:
            scraper.close()
        except Exception:
            pass


def cmd_process(args: argparse.Namespace) -> None:
    out_dir = os.path.abspath(args.out_dir)
    scraped_path = os.path.join(out_dir, "scraped_data.json")
    processed_path = os.path.join(out_dir, "processed_data.json")

    if not os.path.exists(scraped_path):
        logger.error(f"{scraped_path} not found. Run 'trainer scrape' or 'trainer run' first.")
        sys.exit(1)

    with open(scraped_path, "r", encoding="utf-8") as f:
        scraped_data = json.load(f)

    processor = TextProcessor()
    logger.info("Processing scraped data...")
    processed = processor.process_scraped_data(scraped_data)
    save_json(processed_path, processed)
    logger.info(f"Saved processed data to {processed_path}")


def cmd_retrain(args: argparse.Namespace) -> None:
    out_dir = os.path.abspath(args.out_dir)
    processed_path = os.path.join(out_dir, "processed_data.json")

    if not os.path.exists(processed_path):
        logger.error(f"{processed_path} not found. Run 'trainer process' or 'trainer run' first.")
        sys.exit(1)

    kb = ChatbotKnowledgeBase()
    report = kb.retrain(processed_path)
    if not report.get("success", False):
        logger.error(f"Retrain failed: {report.get('error', 'unknown error')}")
        sys.exit(2)
    logger.info(f"Retrain report: {report}")


def cmd_status(_: argparse.Namespace) -> None:
    kb = ChatbotKnowledgeBase()
    try:
        size = kb.get_collection_size()
        stats = kb.get_stats()
    except Exception:
        size = 0
        stats = {}
    print(json.dumps({"total_documents": int(size), **(stats or {})}, indent=2))


def cmd_serve(_: argparse.Namespace) -> None:
    # Start api_server.py in the same interpreter
    logger.info("Starting API server (http://127.0.0.1:8000)...")
    code = subprocess.call([sys.executable, os.path.join(os.path.dirname(__file__), "api_server.py")])
    sys.exit(code)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="trainer", description="Scrape/process/retrain CLI")
    sub = p.add_subparsers(dest="cmd")

    # run
    pr = sub.add_parser("run", help="Run full pipeline")
    pr.add_argument("--url", required=True, help="Start URL to scrape")
    pr.add_argument("--max-pages", type=int, default=30, help="Max pages to scrape")
    pr.add_argument("--selenium", action="store_true", help="Use Selenium for JS-heavy sites")
    pr.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "data"), help="Output directory")
    pr.add_argument("--create-training-file", action="store_true", help="Create OpenAI training JSONL")
    pr.add_argument("--skip-retrain", action="store_true", help="Skip rebuilding vector DB")
    pr.add_argument("--start-api", action="store_true", help="Start API server after pipeline")
    pr.set_defaults(func=cmd_run)

    # scrape
    ps = sub.add_parser("scrape", help="Scrape only")
    ps.add_argument("--url", required=True, help="Start URL to scrape")
    ps.add_argument("--max-pages", type=int, default=30, help="Max pages to scrape")
    ps.add_argument("--selenium", action="store_true", help="Use Selenium for JS-heavy sites")
    ps.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "data"), help="Output directory")
    ps.set_defaults(func=cmd_scrape)

    # process
    pp = sub.add_parser("process", help="Process only")
    pp.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "data"), help="Output directory")
    pp.set_defaults(func=cmd_process)

    # retrain
    prt = sub.add_parser("retrain", help="Rebuild KB from processed data")
    prt.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "data"), help="Output directory")
    prt.set_defaults(func=cmd_retrain)

    # status
    pst = sub.add_parser("status", help="Show KB stats")
    pst.set_defaults(func=cmd_status)

    # serve
    psv = sub.add_parser("serve", help="Start FastAPI server")
    psv.set_defaults(func=cmd_serve)

    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "cmd", None):
        # default to run for convenience, but require --url
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
