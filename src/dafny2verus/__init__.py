import asyncio
import os
import re
import subprocess
import tempfile
from pathlib import Path
import random
from dotenv import load_dotenv
import yaml
from datasets import load_dataset
from pydantic_ai import Agent, ModelHTTPError
import logfire

from dafny2verus.config import system_prompt, cfg, ARTIFACTS
from dafny2verus.tools import verus_tool, dafny_tool

load_dotenv()
logfire.configure()
logfire.instrument_pydantic_ai()


def extract_rust_code(text: str) -> str:
    """Extract Rust code from markdown code blocks in the agent output"""
    # Pattern to match ```rust ... ``` blocks
    rust_pattern = r"```rust\s*\n(.*?)\n```"
    matches = re.findall(rust_pattern, text, re.DOTALL)

    if matches:
        # Return the first Rust code block found
        return matches[0].strip()

    # If no ```rust blocks found, try generic ```
    generic_pattern = r"```\s*\n(.*?)\n```"
    matches = re.findall(generic_pattern, text, re.DOTALL)

    if matches:
        # Return the first code block found
        return matches[0].strip()

    # If no code blocks found, return the original text
    return text.strip()


def create_agent():
    """Create and return a configured PydanticAI agent with tools"""
    return Agent(
        cfg["model"],
        name="dafny2verus",
        deps_type=str,
        output_type=str,
        tools=[verus_tool, dafny_tool],
        system_prompt=system_prompt,
        retries=10,
    )


async def translate_dafny_to_verus(dafny_code: str) -> tuple[str, int]:
    """Translate Dafny code to Verus using the agent"""
    agent = create_agent()

    user_prompt = f"""
Please translate the following Dafny code to Verus:

```dafny
{dafny_code}
```

Use the `verus` tool to make sure your output compiles
"""

    result = await agent.run(user_prompt, deps=dafny_code)

    # Extract only the Rust code from the agent's output
    rust_code = extract_rust_code(result.output)
    num_iterations = len(result.all_messages()) // 3

    return rust_code, num_iterations


def load_dafny_bench():
    """Load the DafnyBench dataset from Hugging Face"""
    logfire.info("Loading DafnyBench dataset...")
    dataset = load_dataset("wendy-sun/DafnyBench", split="test")
    return dataset


def is_sample_already_successful(verus_filename: str) -> bool:
    """Check if a sample already has success: true in its success.yml file"""
    artifact_path = ARTIFACTS / "dafnybench" / verus_filename
    success_file = artifact_path / "success.yml"

    if not success_file.exists():
        return False

    try:
        with open(success_file, "r") as success_yaml:
            data = yaml.safe_load(success_yaml)
        return data.get("success", False)
    except Exception:
        return False


async def process_item(
    idx: int, item: dict, max_retries: int = 32, base_delay: float = 5.0
) -> dict:
    """Process a single item from the dataset with exponential backoff"""
    dafny_code = item["ground_truth"]
    dafny_filename = Path(item["test_file"])
    verus_filename = dafny_filename.stem
    artifact_path = ARTIFACTS / "dafnybench" / verus_filename
    
    # Create the output filename by replacing .dfy extension with .rs
    if dafny_filename.suffix.lower() == '.dfy':
        output_filename = dafny_filename.with_suffix('.rs').name
    else:
        # If no .dfy extension, just add .rs
        output_filename = f"{dafny_filename.name}.rs"

    # Check if this sample already succeeded
    if is_sample_already_successful(verus_filename):
        logfire.info(f"Skipping item {idx + 1}: {dafny_filename} (already successful)")
        return {"path": artifact_path, "success": True}

    logfire.info(f"Processing item {idx + 1}: {dafny_filename}")
    artifact_path.mkdir(parents=True, exist_ok=True)

    # Exponential backoff retry logic
    for attempt in range(max_retries + 1):
        try:
            verus_code, num_iterations = await translate_dafny_to_verus(dafny_code)
            with open(artifact_path / output_filename, "w") as verus_file:
                verus_file.write(verus_code)

            # Run verus verification directly instead of using the tool
            # Create temporary file with the code
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".rs", delete=False
            ) as tmpfile:
                tmpfile.write(verus_code)
                temp_file = tmpfile.name

            try:
                # Run verus verification
                result = subprocess.run(
                    [cfg["verus_path"], temp_file],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                verification_success = result.returncode == 0
                verification_output = result.stdout
                verification_error = result.stderr
            except subprocess.TimeoutExpired:
                verification_success = False
                verification_output = ""
                verification_error = "Verus verification timed out after 30 seconds"
            except OSError as exc:
                verification_success = False
                verification_output = ""
                verification_error = f"Error running Verus: {str(exc)}"
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_file)
                except OSError:
                    pass

            info = {
                "success": verification_success,
                "num_iterations": num_iterations,
                "verification_output": verification_output,
                "verification_error": verification_error,
            }
            with open(artifact_path / "success.yml", "w") as success_file:
                yaml.dump(
                    info,
                    success_file,
                )

            return {"path": artifact_path, "success": verification_success}

        except ModelHTTPError as exc:
            if attempt == max_retries:
                logfire.info(
                    f"Failed to process item {idx + 1} after {max_retries} retries: {exc}"
                )
                raise

            # Calculate delay with exponential backoff and jitter
            delay = base_delay * (2**attempt) + random.uniform(0, 1)
            logfire.info(
                f"Rate limited on item {idx + 1}, attempt {attempt + 1}/{max_retries + 1}. Retrying in {delay:.2f}s..."
            )
            await asyncio.sleep(delay)
    return {"path": artifact_path, "success": False}


async def main_async() -> None:
    """Async main function for parallel processing"""
    print("Dafny2Verus translator initialized!")

    # Load the dataset
    dataset = load_dafny_bench()

    # Check for existing successful samples
    skipped_count = 0

    # Pre-filter to see how many we'll skip
    for idx, item in enumerate(dataset):
        dafny_filename = Path(item["test_file"])
        verus_filename = dafny_filename.stem
        if is_sample_already_successful(verus_filename):
            skipped_count += 1

    # Limit concurrent API calls to prevent rate limiting
    semaphore = asyncio.Semaphore(3)  # Allow max 3 concurrent agent calls

    async def process_with_semaphore(idx: int, item: dict) -> dict:
        async with semaphore:
            return await process_item(idx, item)

    item_processes = [
        process_with_semaphore(idx, item) for idx, item in enumerate(dataset)
    ]
    # Process all items in parallel using asyncio.gather and list comprehension
    results = await asyncio.gather(*item_processes)

    with open(ARTIFACTS / "dafnybench_results.yml", "w") as results_file:
        yaml.dump(results, results_file)

    # Calculate statistics
    total_successful = sum(res["success"] for res in results)
    newly_successful = sum(
        res["success"] for res in results if not res.get("skipped", False)
    )
    percentage_successful = total_successful / len(results)

    print("Results:")
    print(f"  Previously successful: {skipped_count}")
    print(f"  Newly successful: {newly_successful}")
    print(f"  Total successful: {total_successful}")
    print(f"  Overall success rate: {100 * percentage_successful:.1f}%")


def main() -> None:
    """Main entry point for dafny2verus"""
    asyncio.run(main_async())
