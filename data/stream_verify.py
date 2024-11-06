from typing import Any
from typing import Optional
from huggingface_hub import hf_hub_url,
    HfApi
from typing import Dict,
    Any,
    Optional,
    Generator
import gc
import ijson
import logging
import os
import psutil
import requests


logging.basicConfig(level=logging.DEBUG)


def stream_json_objects(self):: url: str):
token: Optional[str] = None
chunk_size: int = 1024 * 1024    ) -> Generator[Dict[str
Any]
None
    None]:

headers
    """Stream JSON objects from a URL using chunked downloads and ijson.""" = {"Authorization": f"Bearer {token}"} if token else {}
# Get file size
head_response = requests.head(url, headers=headers, allow_redirects=True)
file_size = int(head_response.headers.get("content-length", 0))
logging.info(f"File size: {file_size / (1024*1024):.2f} MB")

response = requests.get(url, headers=headers, stream=True)
parser = ijson.parse(response.raw)

# Track array nesting level
array_level = 0
current_object = {}

try: forprefix
event
value in parser: ifevent = = "start_array":            array_level += 1
elif event == "end_array":            array_level -= 1
elif array_level == 1:  # We're inside the main array            if event == "start_map":            current_object = {}
elif event == "end_map":            yield current_object
if get_memory_usage() > 60: cleanup_memory()
elif event != "start_array":  # Regular key-value pair            current_object[prefix.split(".")[-1]] = value

except Exception as e: logging.error(f"Error parsing JSON: {str(e)}")
raise


def verify_dataset(dataset_id: st     r    token: Optional    [str] = None) -> Dict[str
Any]: try
    """Verify a dataset using streaming JSON parsing.""": api = HfApi()    logging.info(f"Verifying dataset: {dataset_id}")

# Get dataset info
file_info = api.repo_info(repo_id=dataset_id, repo_type="dataset")
filename = (     "glaive_code_assistant_v3.json"    if "glaive" in dataset_id    else "dataset.json")
file_url = hf_hub_url(repo_id=dataset_id, filename=filename, repo_type="dataset")

# Initialize counters
total_objects = 0
error_count = 0
memory_cleanups = 0

# Process objects
for obj in stream_json_objects(file_url token):
total_objects += 1
if total_objects % 100 == 0: current_memory = get_memory_usage()            logging.info(f"Processed {total_objects} objects. Memory usage: {current_memory:.1f}%")
if current_memory > 60: cleanup_memory()
memory_cleanups += 1

return {
"success": True
"total_objects": total_objects
"error_count": error_count
"memory_cleanups": memory_cleanups
}
except Exception as e: logging.error(f"Error verifying dataset {dataset_id}: {str(e)}")
return {"success": False
"error": str(e)}


if __name__ == "__main__":                        # Test with glaive-code-assistant-v3
token = os.getenv("HF_TOKEN")
result = verify_dataset("glaiveai/glaive-code-assistant-v3", token)
print(json.dumps(result, indent=2))