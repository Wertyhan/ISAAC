import sys
import logging
import argparse

from isaac_scraper.config import Config
from isaac_scraper.scraper import GitScraper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="GitHub scraper for system design docs")
    parser.add_argument("-r", "--repo", help="Repository (owner/repo)")
    parser.add_argument("-p", "--path", help="Start path")
    parser.add_argument("-o", "--output", help="Output JSON file")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        overrides = {k: v for k, v in [
            ("repo_name", args.repo),
            ("start_path", args.path),
            ("output_file", args.output),
        ] if v}
        
        config = Config(**overrides)
        scraper = GitScraper(config)
        scraper.crawl()
        return 0
        
    except KeyboardInterrupt:
        logger.info("\nInterrupted")
        return 130
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
