#!/usr/bin/env python3
"""
End-to-end pipeline: scrape a website, process the text, and rebuild the knowledge base.

Steps:
1) Scrape pages starting from --url
2) Save scraped_data.json
3) Process to processed_data.json (chunks, keywords, Q&A)
4) Rebuild vector DB from processed_data.json
5) (Optional) Create OpenAI training JSONL

Usage (PowerShell, from python_training/):
  .\env\Scripts\Activate.ps1
  python scrape_and_build.py --url https://example.com --max-pages 30 --create-training-file
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict

from web_scraper import WebScraper
from text_processor import TextProcessor
from vector_database import ChatbotKnowledgeBase
from chatbot_trainer import ChatbotTrainer


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("scrape_and_build")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Scrape, process, and rebuild the knowledge base")
    parser.add_argument("--url", required=True, help="Start URL to scrape")
    parser.add_argument("--max-pages", type=int, default=30, help="Max pages to scrape")
    parser.add_argument("--selenium", action="store_true", help="Use Selenium for JS-heavy sites (requires ChromeDriver)")
    parser.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "data"), help="Output directory for JSON files")
    parser.add_argument("--create-training-file", action="store_true", help="Also create OpenAI training JSONL from processed data")
    parser.add_argument("--skip-retrain", action="store_true", help="Skip rebuilding the vector database")

    args = parser.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    ensure_dir(out_dir)

    scraped_path = os.path.join(out_dir, "scraped_data.json")
    processed_path = os.path.join(out_dir, "processed_data.json")
    training_jsonl_path = os.path.join(out_dir, "openai_training.jsonl")

    total_start = time.time()
    logger.info("Starting pipeline")

    # 1) Scrape
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

    # 2) Process
    proc_start = time.time()
    processor = TextProcessor()
    try:
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
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)
    logger.info(f"Processing done in {time.time() - proc_start:.1f}s")

    # 3) Rebuild knowledge base (vector DB)
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

    # 4) Optionally create OpenAI training JSONL
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
    logger.info("Next: run the API server to use the updated knowledge base:")
    logger.info("  python api_server.py")


if __name__ == "__main__":
    main()
