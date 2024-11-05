from dataset_verification_utils import(from datasets import load_dataset
from huggingface_hub import HfApi
from pathlib import Path
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import gc
import itertools
import json
import logging
import os
import os
import psutil
import tempfile
import time
import yaml

"""Script to fix syntax and formatting issues in verify_mapped_datasets.py."""



def fix_verify_mapped_datasets():
    """Fix syntax and formatting issues in verify_mapped_datasets.py."""
    content = """"""Dataset verification utilities for mapped datasets."""



    try_load_dataset,
    timeout,
    TimeoutException,
    categorize_error,
    format_verification_result,
    log_verification_attempt,
    )

    # Configure logging
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
    logging.StreamHandler(),
    logging.FileHandler("mapped_verification.log"),
    ],
    )
    logger = logging.getLogger(__name__)


    def check_memory_usage() -> bool:
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


    def get_dataset_size(dataset_id: str, token: str) -> Optional[float]:
        """Get the total size of dataset files."""
        try:
            api = HfApi(token=token)
            repo_info = api.repo_info(repo_id=dataset_id, repo_type="dataset", token=token)
            siblings = repo_info.siblings
            total_size = 0
            skipped_files = 0
            data_extensions = [".parquet", ".json", ".csv", ".txt", ".jsonl", ".arrow"]

            if not siblings:
                logger.warning(f"No files found in repository {dataset_id}")
                return None

            for sibling in siblings:
            try:
                filepath = getattr(sibling, "rfilename", None)
                if filepath and any(filepath.lower().endswith(ext) for ext in data_extensions):
                    size = getattr(sibling, "size", None)
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
                                    name = getattr(sibling, "rfilename", "unknown")
                                    logger.warning(f"Failed to process file {name}: {str(file_error)}")

                                    if total_size > 0:
                                        logger.info(f"Total dataset size: {total_size/1024/1024:.2f} MB (skipped {skipped_files} files)")
                                        return total_size / 1024  # Convert to KB
                                    return None
                                except Exception as e:
                                    logger.warning(f"Failed to get size for {dataset_id}: {str(e)}")
                                    return None


                                def load_dataset_in_chunks():
                                dataset_id: str, config: str, token: str, chunk_size: int = 100
                                ) -> Tuple[bool, Optional[Exception], Optional[Dict[str, Any]]]:
                                """Load large datasets in chunks using streaming."""
                                try:
                                    dataset = load_dataset(dataset_id, config, streaming=True, trust_remote_code=True, token=token, )
                                    chunks_tested = 0
                                    max_chunks = 5  # Test up to 5 chunks

                                    for chunk_idx in range(max_chunks):
                                    if psutil.Process().memory_percent() > 70:  # Memory threshold
                                    cleanup_memory()

                                    chunk = list(itertools.islice(dataset["train"], chunk_size))
                                    current_size = len(chunk)
                                    chunks_tested += 1

                                    if current_size == 0:
                                        break

                                    del chunk
                                    cleanup_memory()

                                    if chunks_tested >= max_chunks:
                                        break

                                    return True, None, {"chunks_tested": chunks_tested}
                                except Exception as e:
                                    return False, e, None


                                def load_dataset_mappings() -> Dict[str, Any]:
                                    """Load dataset mappings from YAML file."""
                                    mapping_file = Path(__file__).parent / "dataset_mappings.yaml"
                                    if not mapping_file.exists():
                                        logger.warning("No dataset mappings file found")
                                        return {}


                                    with open(mapping_file, "r") as f:
                                        return yaml.safe_load(f) or {}


                                    def verify_dataset():
                                    local_dir: str, dataset_id: str, token: str, config: Optional[str] = None
                                    ) -> Dict[str, Any]:
                                    """Verify a single dataset using its mapping."""
                                    result = {
                                    "status": "failed",
                                    "error": None,
                                    "configs": {},
                                    "attempts": [],
                                    "organization": {
                                    "local_dir": local_dir,
                                    "structure": {},
                                    "format": None,
                                    "documentation_compliance": False,
                                    "compliance_details": {},
                                    },
                                    }

                                    try:
                                        # Create temporary cache directory
                                        with tempfile.TemporaryDirectory() as cache_dir:
                                            logger.info(f"\\nVerifying dataset: {dataset_id}")
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
                                                        filepath = getattr(sibling, "rfilename", None)
                                                        if filepath:
                                                            path_parts = filepath.split("/")
                                                            current = structure
                                                            for part in path_parts[:-1]:
                                                            current = current.setdefault(part, {})
                                                            current[path_parts[-1]] = getattr(sibling, "size", "unknown size")

                                    except Exception as e:
                                                                logger.warning(f"Failed to process file structure: {str(e)}")

                                                                result["organization"]["structure"] = structure
                                                                logger.info(f"Dataset structure:\\n{json.dumps(structure, indent=2)}")

                                                                # Detect dataset format
                                                                formats = set()
                                                                for sibling in repo_info.siblings:
                                                                try:
                                                                    filepath = getattr(sibling, "rfilename", None)
                                                                    if filepath:
                                                                        ext = os.path.splitext(filepath)[1].lower()
                                                                        if ext in [".parquet", ".json", ".csv", ".txt", ".jsonl", ".arrow"]:
                                                                            formats.add(ext)
                                                                except Exception as e:
                                                                                logger.warning(f"Failed to detect file format: {str(e)}")

                                                                                result["organization"]["format"] = list(formats)
                                                                                logger.info(f"Dataset formats: {formats}")

                                                                                # Check documentation compliance
                                                                                compliance_details = {
                                                                                "has_readme": False,
                                                                                "has_documentation": False,
                                                                                "has_data_files": False,
                                                                                "has_standard_dirs": False,
                                                                                }

                                                                                for sibling in repo_info.siblings:
                                                                                try:
                                                                                    filepath = getattr(sibling, "rfilename", "").lower()
                                                                                    if filepath.endswith("readme.md"):
                                                                                        compliance_details["has_readme"] = True
                                                                                        elif filepath.endswith(".md"):
                                                                                            compliance_details["has_documentation"] = True
                                                                                            elif any(filepath.endswith(ext) for ext in [".parquet", ".json", ".csv", ".txt", ".jsonl", ".arrow"]):
                                                                                                compliance_details["has_data_files"] = True
                                                                                                if any(d in filepath for d in ["train/", "test/", "validation/"]):
                                                                                                    compliance_details["has_standard_dirs"] = True
                                                                                except Exception as e:
                                                                                                        logger.warning(f"Failed to check compliance: {str(e)}")

                                                                                                        # Dataset is compliant if it has either standard dirs or proper documentation
                                                                                                        result["organization"]["documentation_compliance"] = (
                                                                                                        compliance_details["has_readme"]
                                                                                                        and(compliance_details["has_standard_dirs"]
                                                                                                        or compliance_details["has_documentation"])
                                                                                                        and compliance_details["has_data_files"]
                                                                                                        )
                                                                                                        result["organization"]["compliance_details"] = compliance_details
                                                                                                        logger.info(f"Documentation compliance: {result['organization']['documentation_compliance']}")
                                                                                                        logger.info(f"Compliance details: {json.dumps(compliance_details, indent=2)}"
                                                                                                        )

                                                                                                        except Exception as e:
                                                                                                            logger.error(f"Failed to check dataset organization: {str(e)}")
                                                                                                            result["error"] = str(e)
                                                                                                            return result

                                                                                                        # Try loading dataset
                                                                                                        try:
                                                                                                            dataset_size = get_dataset_size(dataset_id, token)
                                                                                                            if dataset_size and dataset_size > 1024 * 1024:  # If > 1GB
                                                                                                            success, error, details = load_dataset_in_chunks(dataset_id, config or "train", token)
                                                                                                            if not success:
                                                                                                                raise error or Exception("Failed to load dataset in chunks")
                                                                                                                else:
                                                                                                                    dataset = try_load_dataset(dataset_id, config, token)
                                                                                                                    if not dataset:
                                                                                                                        raise Exception("Failed to load dataset")

                                                                                                                        result["status"] = "success"
                                                                                                                        logger.info("Dataset verification completed successfully")
                                                                                                        except Exception as e:
                                                                                                                            logger.error(f"Failed to load dataset: {str(e)}")
                                                                                                                            result["error"] = str(e)

                                                                                                                            except Exception as e:
                                                                                                                                logger.error(f"Dataset verification failed: {str(e)}")
                                                                                                                                result["error"] = str(e)

                                                                                                                                return result
                                                                                                                            """

                                                                                                                            # Write the fixed content to the file
                                                                                                                            file_path = Path("data/verify_mapped_datasets.py")
                                                                                                                            with open(file_path, "w") as f:
                                                                                                                                f.write(content)


                                                                                                                                if __name__ == "__main__":
                                                                                                                                    fix_verify_mapped_datasets()