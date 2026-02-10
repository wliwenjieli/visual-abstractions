#!/usr/bin/env python3
"""
Main script to run the abstraction analysis pipeline
"""
import os
import sys
from pathlib import Path

import argparse

# CRITICAL: Redirect cache BEFORE importing torch/transformers
# Use /user_data/ which is NOT backed up (unlike /home/ which takes snapshot space)
CACHE_BASE = Path("/user_data/wenjiel2/.cache")
CACHE_BASE.mkdir(parents=True, exist_ok=True)

os.environ['TORCH_HOME'] = str(CACHE_BASE / 'torch')
os.environ['HF_HOME'] = str(CACHE_BASE / 'huggingface')
os.environ['TRANSFORMERS_CACHE'] = str(CACHE_BASE / 'huggingface' / 'transformers')
os.environ['XDG_CACHE_HOME'] = str(CACHE_BASE)

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import Config
from core.pipeline import AbstractionPipeline

def main():
    parser = argparse.ArgumentParser(
        description='Run abstraction analysis pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_pipeline.py --test              # Test with 3 models
  python scripts/run_pipeline.py                     # Run all models
  python scripts/run_pipeline.py --models model1 model2  # Specific models
  python scripts/run_pipeline.py --force             # Force recompute
  python scripts/run_pipeline.py --clear-cache       # Clear cache first
        """
    )
    
    parser.add_argument(
        '--models', 
        nargs='+',
        help='Specific model UIDs to process'
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test mode: process only first 3 models'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force recompute all metrics'
    )
    
    parser.add_argument(
        '--clear-cache',
        action='store_true',
        help='Clear all cache before starting'
    )
    
    parser.add_argument(
        '--cache-size',
        type=int,
        default=50,
        help='Maximum cache size in GB (default: 50)'
    )
    
    parser.add_argument(
        '--tasks',
        nargs='+',
        choices=['identity', 'geometry', 'relation', 'pattern_only'],
        help='Specific tasks to run'
    )
    
    args = parser.parse_args()
    
    # Initialize configuration
    config = Config()
    config.MAX_CACHE_SIZE_GB = args.cache_size
    print("start pipeline")
    # Initialize pipeline
    pipeline = AbstractionPipeline(config, force_recompute=args.force)
    
    # Clear cache if requested
    if args.clear_cache:
        print("Clearing all cache...")
        result = pipeline.cache_manager.clear_all_cache(confirm=True)
        if 'freed_space_gb' in result:
            print(f"Freed {result['freed_space_gb']:.2f} GB")
    
    # Run pipeline
    try:
        pipeline.run(
            model_uids=args.models,
            test_mode=args.test
        )
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nPipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Show final cache status
    final_cache = pipeline.cache_manager.get_cache_size_gb()
    print(f"\nFinal cache size: {final_cache['total']:.2f} GB")

if __name__ == '__main__':
    main()