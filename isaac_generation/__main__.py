"""ISAAC Generation Module Entry Point"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)


async def test_generation():
    """Test the generation service with a sample query."""
    from isaac_generation.service import get_generation_service
    
    print("\n" + "=" * 60)
    print("ISAAC Generation Module Test")
    print("=" * 60 + "\n")
    
    service = get_generation_service()
    print("✓ GenerationService initialized\n")
    
    # Test query
    test_query = "Explain the Twitter timeline architecture"
    print(f"Query: {test_query}\n")
    print("-" * 40)
    
    response = await service.process_query(query=test_query)
    
    print(f"Sources: {response.context.sources}")
    print(f"Images found: {len(response.context.images)}")
    
    for img in response.context.images:
        print(f"  - {img.image_id}: {img.path.name}")
    
    print("\n" + "-" * 40)
    print("Response (first 500 chars):\n")
    
    full_response = ""
    async for token in response.token_stream:
        full_response += token
        if len(full_response) < 500:
            print(token, end="", flush=True)
    
    if len(full_response) >= 500:
        print("...")
    
    print("\n\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="ISAAC Generation Module")
    parser.add_argument("--test", action="store_true", help="Run test query")
    
    args = parser.parse_args()
    
    if args.test:
        asyncio.run(test_generation())
    else:
        print("ISAAC Generation Module")
        print("Use --test to run a test query")


if __name__ == "__main__":
    main()
