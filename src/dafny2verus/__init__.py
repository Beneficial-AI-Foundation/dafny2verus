import asyncio
import re
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
    print("Loading DafnyBench dataset...")
    dataset = load_dataset("wendy-sun/DafnyBench", split="test")
    return dataset


async def process_item(
    idx: int, item: dict, max_retries: int = 32, base_delay: float = 1.0
) -> dict:
    """Process a single item from the dataset with exponential backoff"""
    dafny_code = item["ground_truth"]
    dafny_filename = Path(item["test_file"])
    print(f"Processing item {idx + 1}: {dafny_filename}")
    verus_filename = dafny_filename.stem
    artifact_path = ARTIFACTS / "dafnybench" / verus_filename
    artifact_path.mkdir(parents=True, exist_ok=True)

    # Exponential backoff retry logic
    for attempt in range(max_retries + 1):
        try:
            verus_code, num_iterations = await translate_dafny_to_verus(dafny_code)
            with open(artifact_path / "verus_code.rs", "w") as verus_file:
                verus_file.write(verus_code)

            result = verus_tool(verus_code)
            info = {
                "success": result.success,
                "num_iterations": num_iterations,
            }
            with open(artifact_path / "success.yml", "w") as success_file:
                yaml.dump(
                    info,
                    success_file,
                )

            return {"path": artifact_path, "success": result.success}

        except ModelHTTPError as exc:
            if attempt == max_retries:
                print(
                    f"Failed to process item {idx + 1} after {max_retries} retries: {exc}"
                )
                raise

            # Calculate delay with exponential backoff and jitter
            delay = base_delay * (2**attempt) + random.uniform(0, 1)
            print(
                f"Rate limited on item {idx + 1}, attempt {attempt + 1}/{max_retries + 1}. Retrying in {delay:.2f}s..."
            )
            await asyncio.sleep(delay)
    return {"path": artifact_path, "success": False}


async def main_async() -> None:
    """Async main function for parallel processing"""
    print("Dafny2Verus translator initialized!")

    # Load the dataset
    dataset = load_dafny_bench()

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
    percentage_successful = sum(res["success"] for res in results) / len(results)
    print(f"Translation to verus was {100 * percentage_successful}% successful.")


def main() -> None:
    """Main entry point for dafny2verus"""
    asyncio.run(main_async())
