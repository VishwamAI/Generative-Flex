from datasets import load_dataset
from huggingface_hub import hf_hub_url, HfApi
from pathlib import Path
from typing import Dict, List, Optional, Any, Iterator, Tuple
import gc
import itertools
import json
import logging
import os
import psutil
import tempfile
import time
import torch
import yaml

    """Dataset verification utilities for mapped datasets."""



# Configure logging
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
format="%(asctime)s - %(levelname)s - %(message)s",
handlers=[
logging.StreamHandler(),
logging.FileHandler("mapped_verification.log"),
])
logger = logging.getLogger(__name__)


class TimeoutException(Exception):
    """Exception raised when a timeout occurs."""

pass


@contextlib.contextmanager
def timeout(seconds: int) -> Iterator[None]:
    """Context manager for timing out operations."""
timer = None

def timeout_handler(self):
    raise TimeoutException(f"Timed out after {seconds} seconds")

    try:
        timer = threading.Timer(seconds, timeout_handler)
        timer.start()
        yield
        finally:
            if timer:
                timer.cancel()


def categorize_error(error: Exception) -> str:
    """Categorize the type of error encountered during dataset verification."""
error_str = str(error)

if isinstance(error, TimeoutException):
    return "timeout"
    elif "401" in error_str:
        return "authentication"
        elif "404" in error_str:
            return "not_found"
            elif "Loading a streaming dataset in parallel" in error_str:
                return "streaming_parallel"
                elif "trust_remote_code" in error_str:
                    return "trust_remote_code"
                    elif "download_timeout" in error_str:
                        return "config_timeout"
                        elif "memory" in error_str.lower():
                            return "memory"
                            else:
                                return "other"


def try_load_dataset(self):
    dataset_id: str,
    config: Optional[str] = None,
    streaming: bool = False,
    trust_remote_code: bool = False,
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
    timeout_seconds: int = 300) -> Tuple[bool, Optional[Exception], Optional[Dict[str, Any]]]:
        """Try to load a dataset with specific configuration and timeout."""
    try:
        with timeout(timeout_seconds):
            kwargs = {
            "streaming": streaming,
            "trust_remote_code": trust_remote_code,
            }
            if config:
                kwargs["name"] = config
                if cache_dir:
                    kwargs["cache_dir"] = cache_dir
                    if token:
                        kwargs["token"] = token

                        dataset = load_dataset(dataset_id, **kwargs)

                        # Get available splits
                        splits = list(dataset.keys())

                        # Try to get features from first available split if train is not available
                        features = None
                        test_split = None
                        if splits:
                            first_split = splits[0]
                            features = str(dataset[first_split].features)
                            test_split = first_split

                            info = {
                            "splits": splits,
                            "features": features,
                            "streaming": streaming,
                            "config": config,
                            }

                            # Test dataset access using first available split
                            if test_split:
                                if streaming:
                                    next(iter(dataset[test_split]))
                                    else:
                                        dataset[test_split][0]

                                        # Clean up memory if not streaming
                                        if not streaming and hasattr(dataset, "_cleanup_files"):
                                            dataset._cleanup_files()

                                            return True, None, info

                                            except Exception as e:
                                                # Clean up any partial downloads
                                                if "dataset" in locals():
                                                    try:
                                                        if hasattr(dataset, "_cleanup_files"):
                                                            dataset._cleanup_files()
                                                            except:
                                                                pass
                                                                return False, e, None


def format_verification_result(result: Dict[str, Any]) -> str:
    """Format the verification result for logging."""
status = result.get("status", "unknown")
configs = result.get("configs", {})
error = result.get("error")
attempts = result.get("attempts", [])

formatted = f"Status: {status}\n"

if configs:
    formatted += "Configurations:\n"
    for config, config_status in configs.items():
        formatted += f"  - {config}: {config_status}\n"

        if attempts:
            formatted += "\nVerification Attempts:\n"
            for attempt in attempts:
                formatted += f"  Strategy: {attempt['strategy']}\n"
                formatted += f"  Config: {attempt['config']}\n"
                formatted += f"  Success: {attempt['success']}\n"
                if attempt.get("error"):
                    formatted += f"  Error: {attempt['error']}\n"
                    formatted += f"  Error Category: {attempt['error_category']}\n"
                    formatted += "\n"

                    if error:
                        formatted += f"\nFinal Error: {error}\n"
                        formatted += f"Error Category: {categorize_error(Exception(error))}\n"

                        return formatted


def log_verification_attempt(self):
    logger: logging.Logger,
    dataset_id: str,
    attempt_type: str,
    config: Optional[str] = None,
    error: Optional[Exception] = None,
    success: bool = False,
    info: Optional[Dict[str, Any]] = None) -> None:
        """Log a verification attempt with detailed information."""
    config_str = f" (config: {config})" if config else ""
    if success:
        logger.info(f"Successfully verified {dataset_id}{config_str} using {attempt_type}")
        if info:
            logger.info(f"Dataset info: {info}")
            else:
                error_category = categorize_error(error) if error else "unknown"
                error_msg = str(error) if error else "No error message"
                logger.error(f"Failed to verify {dataset_id}{config_str} using {attempt_type}")
                logger.error(f"Error category: {error_category}")
                logger.error(f"Error details: {error_msg}")


def get_memory_usage() -> float:
    """Get current memory usage as a percentage."""
process = psutil.Process(os.getpid())
return process.memory_percent()


def cleanup_memory(self):
    """Perform aggressive memory cleanup."""
gc.collect()
try:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        except ImportError:
            pass


def load_dataset_in_chunks(self):
    dataset_id: str,
    split: str = "train",
    chunk_size: int = 50,
    max_chunks: Optional[int] = None,
    streaming: bool = True,
    config: Optional[str] = None,
    token: Optional[str] = None,
    memory_threshold: float = 80.0) -> Tuple[bool, Optional[Exception], Optional[Dict[str, Any]]]:
        """Load and verify a dataset in chunks to manage memory usage."""
    try:
        # Initialize tracking variables
        chunks_processed = 0
        total_examples = 0
        error_count = 0
        cleanup_counter = 0
        line_buffer = []
        download_chunk_size = 1024 * 1024  # 1MB chunks for download
        max_retries = 3

        # Get dataset info first
        info = {
        "streaming": streaming,
        "config": config,
        "chunk_size": chunk_size,
        "chunks_processed": 0,
        "total_examples": 0,
        "error_count": 0,
        "memory_cleanups": 0,
        "parse_errors": 0,
        "download_retries": 0,
        "bytes_processed": 0,
        }

        try:
            # Get the file URL
            api = HfApi()
            logging.debug(f"Getting repo info for {dataset_id}")
            file_info = api.repo_info(repo_id=dataset_id, repo_type="dataset")
            filename = (
            "glaive_code_assistant_v3.json"
            if "glaive" in dataset_id
            else "dataset.json"
            )
            file_url = hf_hub_url(repo_id=dataset_id, filename=filename, repo_type="dataset")

            # Get file size
            headers = {"Authorization": f"Bearer {token}"} if token else {}
            head_response = requests.head(file_url, headers=headers, allow_redirects=True)
            file_size = int(head_response.headers.get("content-length", 0))
            logging.info(f"File size: {file_size / (1024*1024):.2f} MB")

            # Process in chunks using HTTP range requests
            start_byte = 0
            partial_line = ""

            while start_byte < file_size:
                # Download chunk with retries
                end_byte = min(start_byte + download_chunk_size - 1, file_size - 1)
                range_header = {"Range": f"bytes={start_byte}-{end_byte}"}
                headers.update(range_header)

                retry_count = 0
                chunk_data = None
                while retry_count < max_retries and chunk_data is None:
                    try:
                        logging.debug(f"Downloading bytes {start_byte}-{end_byte} "
                        f"({(end_byte-start_byte + 1)/(1024*1024):.2f} MB)"
                        )
                        response = requests.get(file_url, headers=headers, stream=True, timeout=30)

                        if response.status_code == 206:  # Partial Content
                        chunk_data = response.content.decode("utf-8")
                        else:
                            logging.warning(f"Unexpected status code: {response.status_code}")
                            retry_count += 1
                            except Exception as download_error:
                                logging.warning(f"Download error: {str(download_error)}")
                                retry_count += 1
                                if retry_count >= max_retries:
                                    raise Exception(f"Failed to download chunk after {max_retries} retries")

                                    info["download_retries"] += retry_count
                                    info["bytes_processed"] = start_byte

                                    # Handle partial lines from previous chunk
                                    chunk_data = partial_line + chunk_data
                                    lines = chunk_data.split("\n")

                                    # Save last partial line for next chunk
                                    partial_line = lines[-1] if not chunk_data.endswith("\n") else ""
                                    lines = lines[:-1] if not chunk_data.endswith("\n") else lines

                                    # Process complete lines
                                    for line in lines:
                                        if not line.strip():
                                            continue

                                            try:
                                                obj = json.loads(line)
                                                line_buffer.append(obj)

                                                if len(line_buffer) >= chunk_size:
                                                    total_examples += len(line_buffer)
                                                    chunks_processed += 1
                                                    cleanup_counter += 1
                                                    logging.debug(f"Processed chunk {chunks_processed} ({total_examples} examples)"
                                                    )
                                                    line_buffer = []

                                                    current_memory = get_memory_usage()
                                                    if(current_memory > memory_threshold
                                                    or cleanup_counter >= 3):
                                                        cleanup_memory()
                                                        cleanup_counter = 0
                                                        info["memory_cleanups"] += 1

                                                        info.update({
                                                        "chunks_processed": chunks_processed, "total_examples": total_examples, "error_count": error_count, "last_memory_usage": current_memory, "progress_percentage": (start_byte / file_size)
                                                        * 100,
                                                        }
                                                        )

                                                        if max_chunks and chunks_processed >= max_chunks:
                                                            return True, None, info

                                                            except json.JSONDecodeError as je:
                                                                error_count += 1
                                                                info["parse_errors"] += 1
                                                                logging.warning(f"JSON parse error: {str(je)[:100]}...")
                                                                if error_count > chunks_processed * 0.1:  # Allow 10% error rate
                                                                raise Exception(f"Too many JSON parse errors: {error_count}/{chunks_processed}")
                                                                continue

                                                                start_byte = end_byte + 1

                                                                except requests.exceptions.RequestException as re:
                                                                    # Only fall back for network-related errors
                                                                    logging.warning(f"Network error, falling back to datasets library: {str(re)}"
                                                                    )
                                                                    kwargs = {"streaming": True, "split": split}
                                                                    if config:
                                                                        kwargs["name"] = config
                                                                        if token:
                                                                            kwargs["token"] = token

                                                                            dataset = load_dataset(dataset_id, **kwargs)
                                                                            info.update({
                                                                            "splits": (
                                                                            list(dataset.keys()) if hasattr(dataset, "keys") else [split]
                                                                            ),
                                                                            "features": (
                                                                            str(dataset.features) if hasattr(dataset, "features") else None
                                                                            ),
                                                                            "fallback_method": "datasets_library",
                                                                            }
                                                                            )

                                                                            for batch in dataset.iter(batch_size=chunk_size):
                                                                                try:
                                                                                    current_memory = get_memory_usage()
                                                                                    if current_memory > memory_threshold:
                                                                                        cleanup_memory()
                                                                                        info["memory_cleanups"] += 1

                                                                                        total_examples += len(batch)
                                                                                        chunks_processed += 1
                                                                                        cleanup_counter += 1

                                                                                        if cleanup_counter >= 3:
                                                                                            cleanup_memory()
                                                                                            cleanup_counter = 0
                                                                                            info["memory_cleanups"] += 1

                                                                                            info.update({
                                                                                            "chunks_processed": chunks_processed, "total_examples": total_examples, "error_count": error_count, "last_memory_usage": current_memory, })

                                                                                            if max_chunks and chunks_processed >= max_chunks:
                                                                                                break
                                                                                                except Exception as chunk_error:
                                                                                                    error_count += 1
                                                                                                    info["error_count"] = error_count
                                                                                                    info["last_error"] = str(chunk_error)

                                                                                                    if error_count > chunks_processed * 0.1:
                                                                                                        raise Exception(f"Too many chunk processing errors: {error_count}/{chunks_processed}")

                                                                                                        return True, None, info

                                                                                                        except Exception as e:
                                                                                                            error_info = {
                                                                                                            "error": str(e),
                                                                                                            "error_category": categorize_error(e),
                                                                                                            "chunks_processed": chunks_processed,
                                                                                                            "total_examples": total_examples,
                                                                                                            "error_count": error_count,
                                                                                                            }
                                                                                                            return False, e, error_info

                                                                                                            finally:
                                                                                                                # Final cleanup
                                                                                                                cleanup_memory()
