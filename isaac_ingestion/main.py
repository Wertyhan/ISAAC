import sys
import logging
import argparse

from tqdm import tqdm

from isaac_ingestion.config import Config
from isaac_ingestion.pipeline import IngestionPipeline
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


# Pipeline with Progress
class ProgressPipeline(IngestionPipeline):
    
    def run(self) -> dict:
        logger.info("Starting ingestion pipeline")
        
        # Load raw data
        raw_data = self._load_raw_data()
        logger.info(f"Loaded {len(raw_data)} projects from {self._config.raw_data_file}")
        
        # Initialize vector store
        self._init_vector_store()
        
        # Process with progress bar
        all_documents = []
        
        with tqdm(raw_data, desc="Processing projects", unit="project") as pbar:
            for project_data in pbar:
                project_name = project_data.get("project_name", "unknown")
                pbar.set_postfix(project=project_name[:20])
                
                try:
                    docs = self._process_project(project_data)
                    all_documents.extend(docs)
                    self._stats["projects_processed"] += 1
                except Exception as e:
                    logger.error(f"Failed to process {project_name}: {e}")
                    self._stats["errors"] += 1
        
        # Persist to vector store
        if all_documents:
            logger.info(f"Persisting {len(all_documents)} documents to vector store...")
            self._persist_documents(all_documents)
        
        logger.info(f"Pipeline complete. Stats: {self._stats}")
        return dict(self._stats)


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
        
        with ProgressPipeline(config) as pipeline:
            stats = pipeline.run()
            
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
