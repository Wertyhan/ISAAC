# Imports
import sys
import logging
import argparse

from isaac_scraper.config import Config
from isaac_scraper.scraper import GitScraper
from isaac_scraper.exceptions import (
    AuthenticationError,
    RepositoryNotFoundError,
    RateLimitError,
)

# Logger
logger = logging.getLogger(__name__)


# Helpers
def _build_overrides(args: argparse.Namespace) -> dict:
    """Build config overrides from command line arguments."""
    overrides = {}
    if args.repo:
        overrides["repo_name"] = args.repo
    if args.path:
        overrides["start_path"] = args.path
    if args.output:
        overrides["output_file"] = args.output
    return overrides


# Execution
def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S"
    )

    parser = argparse.ArgumentParser(description="GitHub scraper for system design docs")
    parser.add_argument("-r", "--repo", help="Repository (owner/repo)")
    parser.add_argument("-p", "--path", help="Start path")
    parser.add_argument("-o", "--output", help="Output JSON file")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        overrides = _build_overrides(args)
        config = Config(**overrides)
        
        with GitScraper(config) as scraper:
            scraper.crawl()
            stats = scraper.get_stats()
            logger.info(f"Stats: {stats}")
        return 0
        
    except KeyboardInterrupt:
        logger.info("\nInterrupted")
        return 130
    except AuthenticationError as e:
        logger.error(f"Authentication failed: {e}")
        return 2
    except RepositoryNotFoundError as e:
        logger.error(f"Repository error: {e}")
        return 3
    except RateLimitError as e:
        logger.error(f"Rate limit exceeded: retry after {e.retry_after:.0f}s")
        return 4
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return 5
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
