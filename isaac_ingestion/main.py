import sys
import logging
import argparse

from tqdm import tqdm

from isaac_ingestion.config import Config
from isaac_ingestion.pipeline import IngestionPipeline
from isaac_ingestion.clients.gemini_client import GeminiVisionClient
from isaac_ingestion.services.image_manager import ImageManager
from isaac_ingestion.services.text_processor import TextProcessor
from isaac_ingestion.exceptions import (
    IngestionError,
    InvalidInputError,
    DatabaseConnectionError,
    GeminiAPIError,
)

# Logger
logger = logging.getLogger(__name__)


# Helpers
def _build_overrides(args: argparse.Namespace) -> dict:
    overrides = {}
    if args.input:
        overrides["raw_data_file"] = args.input
    if args.collection:
        overrides["collection_name"] = args.collection
    if args.db:
        overrides["postgres_connection_string"] = args.db
    return overrides


def _create_pipeline(config: Config) -> IngestionPipeline:
    gemini_client = GeminiVisionClient(config)
    image_manager = ImageManager(config)
    text_processor = TextProcessor(config)
    
    return IngestionPipeline(
        config=config,
        gemini_client=gemini_client,
        image_manager=image_manager,
        text_processor=text_processor,
    )


# Execution
def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S"
    )

    parser = argparse.ArgumentParser(
        description="ISAAC ingestion pipeline - Transform raw data to vector store"
    )
    parser.add_argument("-i", "--input", help="Input JSON file path")
    parser.add_argument("-c", "--collection", help="Vector store collection name")
    parser.add_argument("-d", "--db", help="PostgreSQL connection string")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument("--dry-run", action="store_true", help="Process without persisting")
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        overrides = _build_overrides(args)
        config = Config(**overrides)
        
        with _create_pipeline(config) as pipeline:
            
            # Define progress wrapper
            def progress_wrapper(iterable):
                return tqdm(iterable, desc="Processing projects", unit="project")
                
            stats = pipeline.run(progress_wrapper=progress_wrapper)
            
            # Print summary
            print("\n" + "=" * 50)
            print("INGESTION COMPLETE")
            print("=" * 50)
            for key, value in stats.items():
                print(f"  {key}: {value}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        return 130
    except InvalidInputError as e:
        logger.error(f"Input error: {e}")
        return 2
    except DatabaseConnectionError as e:
        logger.error(f"Database error: {e}")
        return 3
    except GeminiAPIError as e:
        logger.error(f"Gemini API error: {e}")
        return 4
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return 5
    except IngestionError as e:
        logger.error(f"Ingestion error: {e}")
        return 6
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
