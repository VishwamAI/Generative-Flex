import gc
import os
import json
import psutil
import itertools
from pathlib import Path
from typing import Dict, List, Optional, Any

# System imports
import yaml
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile

# HuggingFace imports
from datasets import load_dataset
from huggingface_hub import HfApi

# Local imports
from dataset_verification_utils import (
    try_load_dataset, timeout, TimeoutException,
    categorize_error, format_verification_result,
    log_verification_attempt
)

# Configure logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('mapped_verification.log')
    ]
)
logger = logging.getLogger(__name__)

# Memory management utilities
def check_memory_usage():
    """Check if memory usage is too high."""
    memory_percent = psutil.Process().memory_percent()
    if memory_percent > 80:  # If using more than 80% memory
        logger.warning(f"High memory usage detected: {memory_percent:.1f}%")
        gc.collect()  # Force garbage collection
        return True
    return False

def cleanup_memory():
    """Force cleanup of memory."""
    gc.collect()
    psutil.Process().memory_info().rss  # Force memory info update
    time.sleep(0.1)  # Allow memory to settle

def get_dataset_size(dataset_id, token):
    """Get the total size of dataset files."""
    try:
        api = HfApi(token=token)
        # Get repository information including file sizes
        repo_info = api.repo_info(repo_id=dataset_id, repo_type="dataset", token=token)
        siblings = repo_info.siblings
        total_size = 0
        skipped_files = 0
        # Sum up sizes of data files (parquet, json, etc)
        data_extensions = ['.parquet', '.json', '.csv', '.txt', '.jsonl', '.arrow']

        if not siblings:
            logger.warning(f"No files found in repository {dataset_id}")
            return None

        for sibling in siblings:
            try:
                filepath = sibling.rfilename
                if any(filepath.lower().endswith(ext) for ext in data_extensions):
                    size = getattr(sibling, 'size', None)
                    if size is not None:
                        total_size += size
                        logger.debug(f"Added size for file {filepath}: {size/1024/1024:.2f} MB")
                    else:
                        skipped_files += 1
                        logger.warning(f"Skipped file {filepath} due to missing size information")
            except AttributeError as attr_error:
                skipped_files += 1
                logger.warning(f"Missing required attributes for file in {dataset_id}: {str(attr_error)}")
            except Exception as file_error:
                skipped_files += 1
                name = getattr(sibling, 'rfilename', 'unknown')
                logger.warning(f"Failed to process file {name}: {str(file_error)}")

        if total_size > 0:
            logger.info(f"Total dataset size: {total_size/1024/1024:.2f} MB (skipped {skipped_files} files)")
            return total_size / 1024  # Convert to KB
        return None
    except Exception as e:
        logger.warning(f"Failed to get size for {dataset_id}: {str(e)}")
        return None

def load_dataset_in_chunks(dataset_id, config, token, chunk_size=100):
    """Load large datasets in chunks using streaming."""
    try:
        dataset = load_dataset(
            dataset_id,
            config,
            streaming=True,
            trust_remote_code=True,
            token=token
        )
        chunks_tested = 0
        max_chunks = 5  # Test up to 5 chunks

        # Test chunks with memory cleanup between each
        for chunk_idx in range(max_chunks):
            if psutil.Process().memory_percent() > 70:  # Memory threshold
                cleanup_memory()

            # Get next chunk
            chunk = list(itertools.islice(dataset['train'], chunk_size))
            current_size = len(chunk)
            chunks_tested += 1

            # Check for end of dataset
            if current_size == 0:
                break

            # Clear chunk from memory
            del chunk
            cleanup_memory()

            # Break if we've tested enough chunks
            if chunks_tested >= max_chunks:
                break

        return True, None, {'chunks_tested': chunks_tested}
    except Exception as e:
        return False, e, None

def load_dataset_mappings():
    """Load dataset mappings from YAML file."""
    mapping_file = Path(__file__).parent / 'dataset_mappings.yaml'
    if not mapping_file.exists():
        logger.warning("No dataset mappings file found")
        return {}

    with open(mapping_file, 'r') as f:
        return yaml.safe_load(f) or {}

def verify_dataset(local_dir, dataset_id, token, config=None):
    """Verify a single dataset using its mapping."""
    result = {
        'status': 'failed',
        'error': None,
        'configs': {},
        'attempts': [],
        'organization': {
            'local_dir': local_dir,
            'structure': {},
            'format': None,
            'documentation_compliance': False,
            'compliance_details': {}
        }
    }

    try:
        # Create temporary cache directory
        with tempfile.TemporaryDirectory() as cache_dir:
            logger.info(f"\nVerifying dataset: {dataset_id}")
            logger.info(f"Initial memory usage: {psutil.Process().memory_percent():.1f}%")

            # Check dataset organization and structure
            try:
                api = HfApi(token=token)
                repo_info = api.repo_info(repo_id=dataset_id, repo_type="dataset", token=token)

                # Log dataset structure
                if repo_info.siblings:
                    structure = {}
                    for sibling in repo_info.siblings:
                        try:
                            filepath = getattr(sibling, 'rfilename', None)
                            if filepath:
                                path_parts = filepath.split('/')
                                current = structure
                                for part in path_parts[:-1]:
                                    current = current.setdefault(part, {})
                                current[path_parts[-1]] = getattr(sibling, 'size', 'unknown size')
                        except Exception as e:
                            logger.warning(f"Failed to process file structure: {str(e)}")

                    result['organization']['structure'] = structure
                    logger.info(f"Dataset structure:\n{json.dumps(structure, indent=2)}")

                    # Detect dataset format
                    formats = set()
                    for sibling in repo_info.siblings:
                        try:
                            filepath = getattr(sibling, 'rfilename', None)
                            if filepath:
                                ext = os.path.splitext(filepath)[1].lower()
                                if ext in ['.parquet', '.json', '.csv', '.txt', '.jsonl', '.arrow']:
                                    formats.add(ext)
                        except Exception as e:
                            logger.warning(f"Failed to detect file format: {str(e)}")

                    result['organization']['format'] = list(formats)
                    logger.info(f"Dataset formats: {formats}")

                    # Check documentation compliance with more flexible criteria
                    compliance_details = {
                        'has_readme': False,
                        'has_standard_dirs': False,
                        'has_data_files': False,
                        'has_documentation': False
                    }

                    # Check for README (case-insensitive)
                    readme_files = [f for f in repo_info.siblings if getattr(f, 'rfilename', '').upper().endswith(('README.MD', 'README.TXT'))]
                    compliance_details['has_readme'] = len(readme_files) > 0

                    # Check for standard directory structure
                    expected_dirs = ['raw', 'processed', 'metadata']
                    compliance_details['has_standard_dirs'] = any(dir in structure for dir in expected_dirs)

                    # Check for data files
                    compliance_details['has_data_files'] = len(formats) > 0

                    # Check for any documentation
                    doc_extensions = ['.md', '.txt', '.rst', '.doc', '.docx']
                    has_docs = any(
                        getattr(sibling, 'rfilename', '').lower().endswith(tuple(doc_extensions))
                        for sibling in repo_info.siblings
                    )
                    compliance_details['has_documentation'] = has_docs

                    # Dataset is compliant if it has either standard dirs or proper documentation
                    result['organization']['documentation_compliance'] = (
                        compliance_details['has_readme'] and
                        (compliance_details['has_standard_dirs'] or compliance_details['has_documentation']) and
                        compliance_details['has_data_files']
                    )
                    result['organization']['compliance_details'] = compliance_details
                    logger.info(f"Documentation compliance: {result['organization']['documentation_compliance']}")
                    logger.info(f"Compliance details: {json.dumps(compliance_details, indent=2)}")

            except Exception as e:
                logger.warning(f"Failed to analyze dataset organization: {str(e)}")

            # Check dataset size
            dataset_size_kb = get_dataset_size(dataset_id, token)
            if dataset_size_kb and dataset_size_kb > 1000000:  # If larger than 1GB
                logger.info(f"Large dataset detected ({dataset_size_kb/1000000:.1f} GB). Using chunked loading.")

                # If specific config provided, only try that
                if config:
                    try:
                        logger.info(f"Attempting to load specific config in chunks: {config}")
                        success, error, info = load_dataset_in_chunks(dataset_id, config, token)

                        attempt = {
                            'strategy': 'chunked_config_specific',
                            'config': config,
                            'success': success,
                            'error': str(error) if error else None,
                            'error_category': categorize_error(error) if error else None,
                            'info': info
                        }
                        result['attempts'].append(attempt)

                        if success:
                            result['configs'][config] = 'verified'
                            result['status'] = 'verified'
                            logger.info(f"Successfully verified large dataset {dataset_id} with config {config}")
                            return local_dir, result

                    except Exception as e:
                        logger.warning(f"Chunked config-specific load failed for {dataset_id} with {config}: {str(e)}")
                        cleanup_memory()

            else:
                # Regular verification for smaller datasets
                if config:
                    try:
                        logger.info(f"Attempting to load specific config: {config}")
                        success, error, info = try_load_dataset(
                            dataset_id,
                            config=config,
                            streaming=True,
                            trust_remote_code=True,
                            cache_dir=cache_dir,
                            token=token,
                            timeout_seconds=300
                        )

                        attempt = {
                            'strategy': 'config_specific',
                            'config': config,
                            'success': success,
                            'error': str(error) if error else None,
                            'error_category': categorize_error(error) if error else None,
                            'info': info
                        }
                        result['attempts'].append(attempt)

                        if success:
                            result['configs'][config] = 'verified'
                            result['status'] = 'verified'
                            logger.info(f"Successfully verified {dataset_id} with config {config}")
                            return local_dir, result

                    except Exception as e:
                        logger.warning(f"Config-specific load failed for {dataset_id} with {config}: {str(e)}")
                        cleanup_memory()

            # Basic strategies with memory monitoring
            basic_strategies = [
                ('streaming_basic', True, False, 180),
                ('basic', False, False, 300),
                ('basic_trusted', False, True, 300)
            ]

            # Try basic loading with retries
            for strategy_name, streaming, trust_remote_code, timeout in basic_strategies:
                if check_memory_usage():
                    logger.warning("Skipping non-streaming strategy due to high memory usage")
                    if not streaming:
                        continue

                retries = 3
                while retries > 0:
                    try:
                        logger.info(f"Attempting {strategy_name} load for {dataset_id} (retries left: {retries})")
                        success, error, info = try_load_dataset(
                            dataset_id,
                            streaming=streaming,
                            trust_remote_code=trust_remote_code,
                            cache_dir=cache_dir,
                            token=token,
                            timeout_seconds=timeout
                        )

                        attempt = {
                            'strategy': strategy_name,
                            'config': 'default',
                            'success': success,
                            'error': str(error) if error else None,
                            'error_category': categorize_error(error) if error else None,
                            'info': info
                        }
                        result['attempts'].append(attempt)

                        if success:
                            result['configs']['default'] = 'verified'
                            result['status'] = 'verified'
                            logger.info(f"Successfully verified {dataset_id} with {strategy_name}")
                            return local_dir, result
                        break  # Break if load completed without error

                    except Exception as e:
                        logger.warning(f"Basic load failed for {dataset_id} with {strategy_name}: {str(e)}")
                        cleanup_memory()
                        retries -= 1
                        if retries > 0:
                            time.sleep(2)  # Wait before retry
                        continue

            # Try configurations for failed verifications
            if not config and result['status'] == 'failed':
                try:
                    api = HfApi(token=token)
                    dataset_info = api.dataset_info(dataset_id)
                    configs = []

                    if hasattr(dataset_info, 'config_names') and dataset_info.config_names:
                        configs = dataset_info.config_names
                        logger.info(f"Found configurations for {dataset_id}: {configs}")

                    for config_name in configs:
                        if check_memory_usage():
                            logger.warning(f"Skipping config {config_name} due to high memory usage")
                            continue

                        logger.info(f"Attempting to load config: {config_name}")
                        if dataset_size_kb and dataset_size_kb > 1000000:
                            success, error, info = load_dataset_in_chunks(dataset_id, config_name, token)
                        else:
                            success, error, info = try_load_dataset(
                                dataset_id,
                                config=config_name,
                                streaming=True,
                                trust_remote_code=True,
                                cache_dir=cache_dir,
                                token=token,
                                timeout_seconds=300
                            )

                        attempt = {
                            'strategy': 'config_specific',
                            'config': config_name,
                            'success': success,
                            'error': str(error) if error else None,
                            'error_category': categorize_error(error) if error else None,
                            'info': info
                        }
                        result['attempts'].append(attempt)

                        if success:
                            result['configs'][config_name] = 'verified'
                            result['status'] = 'verified'
                            logger.info(f"Successfully verified {dataset_id} with config {config_name}")
                            break
                        else:
                            result['configs'][config_name] = f'failed: {str(error)}'
                            logger.error(f"Failed to verify config {config_name}: {str(error)}")
                            cleanup_memory()

                except Exception as e:
                    logger.error(f"Failed to get/verify configurations for {dataset_id}: {str(e)}")
                    result['error'] = str(e)
                    cleanup_memory()

    except Exception as e:
        result['status'] = 'failed'
        result['error'] = str(e)
        if '401' in str(e):
            result['status'] = 'auth_failed'
            logger.error(f"Authentication failed for {dataset_id}: {str(e)}")
        else:
            logger.error(f"Failed to verify {dataset_id}: {str(e)}")

        log_verification_attempt(
            logger, dataset_id, 'initial_info',
            error=e, success=False
        )

    logger.info(f"\nVerification result for {dataset_id}:\n{format_verification_result(result)}")
    logger.info(f"Final memory usage: {psutil.Process().memory_percent():.1f}%")
    cleanup_memory()
    return local_dir, result

def main():
    token = os.environ.get('HF_TOKEN')
    if not token:
        logger.error("HF_TOKEN environment variable not set")
        return False

    # Load dataset mappings
    mappings = load_dataset_mappings()
    if not mappings:
        logger.error("No dataset mappings available")
        return False

    logger.info(f"Loaded {len(mappings)} dataset mappings")

    # Dataset configurations that require specific handling
    dataset_configs = {
        'MMMU/MMMU': ['Accounting', 'Math', 'Computer_Science'],  # Sample of important configs
        'openai/summarize_from_feedback': ['axis', 'comparisons'],
        'hellaswag': None,  # Will try default config
        'textvqa': None
    }

    # Track verification results
    verification_results = {}
    total_datasets = len(mappings)
    verified_count = 0
    failed_count = 0

    # Process datasets with dynamic batch sizing
    dataset_items = list(mappings.items())

    for i, (local_dir, dataset_id) in enumerate(dataset_items):
        # Check dataset size to determine batch approach
        dataset_size = get_dataset_size(dataset_id, token)

        # Use single dataset processing for large datasets (>1GB)
        if dataset_size and dataset_size > 1024 * 1024:  # Size in KB > 1GB
            logger.info(f"Large dataset detected ({dataset_size/1024/1024:.1f} GB). Processing individually: {dataset_id}")
            batch_size = 1
        else:
            batch_size = 2

        # Calculate progress
        batch_num = i//batch_size + 1
        total_batches = (len(dataset_items) + batch_size - 1)//batch_size
        logger.info(f"Processing batch {batch_num}/{total_batches} (Progress: {verified_count}/{total_datasets} verified, {failed_count} failed)")

        # Aggressive memory cleanup before processing
        cleanup_memory()
        gc.collect()
        time.sleep(1)  # Allow memory to settle

        try:
            configs = dataset_configs.get(dataset_id)
            if configs:
                # For datasets with specific configs, verify each one
                for config in configs:
                    logger.info(f"\nVerifying dataset: {dataset_id} with config {config}")
                    try:
                        local_dir, result = verify_dataset(local_dir, dataset_id, token, config)
                        if result['status'] == 'verified':
                            verified_count += 1
                        else:
                            failed_count += 1
                        verification_results[f"{local_dir}_{config}"] = result
                        logger.info(f"Verified {local_dir} ({dataset_id}) with config {config}: {result['status']}")
                    except Exception as e:
                        logger.error(f"Error verifying {dataset_id} with config {config}: {str(e)}")
                        verification_results[f"{local_dir}_{config}"] = {
                            'status': 'failed',
                            'error': str(e)
                        }
                        failed_count += 1
            else:
                # For other datasets, try default verification
                logger.info(f"\nVerifying dataset: {dataset_id}")
                try:
                    local_dir, result = verify_dataset(local_dir, dataset_id, token)
                    verification_results[local_dir] = result
                    if result['status'] == 'verified':
                        verified_count += 1
                    else:
                        failed_count += 1
                    logger.info(f"Verified {local_dir} ({dataset_id}): {result['status']}")
                except Exception as e:
                    logger.error(f"Error verifying {dataset_id}: {str(e)}")
                    verification_results[local_dir] = {
                        'status': 'failed',
                        'error': str(e)
                    }
                    failed_count += 1

            # Save progress after each dataset
            output_file = Path(__file__).parent / 'mapped_verification.yaml'
            with open(output_file, 'w') as f:
                yaml.dump(verification_results, f, sort_keys=False, indent=2)

        except Exception as e:
            logger.error(f"Critical error processing {dataset_id}: {str(e)}")
            continue

        # Aggressive cleanup after each dataset
        cleanup_memory()
        gc.collect()
        time.sleep(2)  # Allow memory to settle

    # Calculate statistics
    stats = {
        'total_datasets': len(mappings),
        'verified': sum(1 for r in verification_results.values() if r['status'] == 'verified'),
        'failed': sum(1 for r in verification_results.values() if r['status'] == 'failed'),
        'auth_failed': sum(1 for r in verification_results.values() if r['status'] == 'auth_failed'),
    }

    logger.info("Verification complete. Results:")
    logger.info(f"Total datasets: {stats['total_datasets']}")
    logger.info(f"Verified: {stats['verified']}")
    logger.info(f"Failed: {stats['failed']}")
    logger.info(f"Auth Failed: {stats['auth_failed']}")

    return True

if __name__ == '__main__':
    main()
