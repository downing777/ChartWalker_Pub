import asyncio
from tqdm.asyncio import tqdm_asyncio
import json
import re
import time
from typing import Any, Protocol, Callable, TYPE_CHECKING, List
import logging
import logging.handlers
import os

DEFAULT_LOG_MAX_BYTES = 10485760  # Default 10MB
DEFAULT_LOG_BACKUP_COUNT = 5  # Default 5 backups
DEFAULT_LOG_FILENAME = "lightrag.log"  # Default log filename

async def async_task_runner(tasks, max_concurrent=32, describe="Processing tasks"):
    """
    General asynchronous task scheduler that supports task concurrency control and progress bar display.
    param:                        
        - tasks: A list of tasks to be executed (already created coroutines).
        - max_concurrent: The maximum number of tasks to run concurrently.
    return: 
        A list of task results returned in the original order.
    """
    sem = asyncio.Semaphore(max_concurrent)  
    results = []
    
    async def wrapped_task(task):
        async with sem:
            return await task
    
    # 使用 tqdm_asyncio 显示进度条
    results = await tqdm_asyncio.gather(*(wrapped_task(task) for task in tasks), desc=describe)
    
    return results


def safe_parser(response: str) -> dict:
        '''
        safe parser with enhanced error handling for thinking models and truncated JSON
        '''
        try:
            if isinstance(response, dict):
                return response
            
            # Handle None or empty response
            if not response or response is None:
                raise ValueError("Response is None or empty")
            
            # Extract content after </think> tag (for thinking models)
            # The thinking model returns: reasoning + "\n</think>\n" + content
            if '</think>' in response:
                thinking_match = re.search(r'</think>\s*(.*)', response, re.DOTALL)
                if thinking_match:
                    response = thinking_match.group(1).strip()
            
            # Try to extract JSON from code blocks first
            match = re.search(r'```json\s*([\s\S]*?)\s*```', response)
            if match:
                json_str = match.group(1).strip()
            else:
                # Try direct JSON parsing
                try:
                    return json.loads(response)
                except json.JSONDecodeError:
                    # Extract JSON object from response
                    match = re.search(r'(\{[\s\S]*)', response)
                    if not match:
                        raise ValueError("No valid JSON found in response")
                    json_str = match.group(1)

            # Try parsing the extracted JSON
            try:
                clean_response = json.loads(json_str)
                return clean_response
            except json.JSONDecodeError as json_err:
                # Attempt to fix truncated JSON
                fixed_json = _fix_truncated_json(json_str)
                if fixed_json:
                    try:
                        clean_response = json.loads(fixed_json)
                        return clean_response
                    except json.JSONDecodeError:
                        pass
                
                # If fixing failed, try to extract partial JSON (entities only)
                # This is a fallback for very large truncated JSONs
                try:
                    # Try to extract just the entities array if JSON is truncated
                    entities_match = re.search(r'"entities"\s*:\s*\[(.*?)(?:\]|$)', json_str, re.DOTALL)
                    if entities_match:
                        # Try to parse entities separately
                        entities_str = entities_match.group(1)
                        # Count complete entity objects
                        entity_count = entities_str.count('"entity_name"')
                        if entity_count > 0:
                            # Create a minimal valid JSON with just entities
                            partial_json = f'{{"entities": [{entities_str.rstrip(", ")}], "relationships": []}}'
                            try:
                                clean_response = json.loads(partial_json)
                                return clean_response
                            except json.JSONDecodeError:
                                pass
                except Exception:
                    pass
                
                # If all fixes failed, raise original error
                raise ValueError(f"JSON decoding failed: {json_err}\nExtracted JSON length: {len(json_str)} chars\nFirst 500 chars:\n{json_str[:500]}...")

        except json.JSONDecodeError as e:
            raise ValueError(f"JSON decoding failed: {e}\nExtracted JSON:\n{json_str[:500] if 'json_str' in locals() else 'N/A'}...")
        except Exception as e:
            raise ValueError(f"Parsing failed: {e}\nResponse content:\n{str(response)[:500] if response else 'None'}...")


def _fix_truncated_json(json_str: str) -> str:
    """
    Attempt to fix truncated JSON by adding missing closing brackets/braces.
    Uses a stack-based approach to properly handle nested structures.
    """
    if not json_str or not json_str.strip().startswith('{'):
        return None
    
    trimmed = json_str.rstrip()
    if not trimmed:
        return None
    
    # Track open/close counts
    open_braces = trimmed.count('{')
    close_braces = trimmed.count('}')
    open_brackets = trimmed.count('[')
    close_brackets = trimmed.count(']')
    
    # Check if we're in the middle of a string (simple heuristic)
    # Count unescaped quotes
    in_string = False
    escape_next = False
    quote_count = 0
    for char in trimmed:
        if escape_next:
            escape_next = False
            continue
        if char == '\\':
            escape_next = True
            continue
        if char in ['"', "'"]:
            in_string = not in_string
            quote_count += 1
    
    # If we're in a string, try to close it
    if in_string:
        trimmed = trimmed + '"'
    
    # Build stack to track nested structures
    stack = []
    i = 0
    while i < len(trimmed):
        char = trimmed[i]
        if char == '\\':
            i += 2  # Skip escaped character
            continue
        if char in ['"', "'"]:
            # Find end of string
            quote_char = char
            i += 1
            while i < len(trimmed) and (trimmed[i] != quote_char or (i > 0 and trimmed[i-1] == '\\')):
                i += 1
            if i < len(trimmed):
                i += 1
            continue
        if char == '{':
            stack.append('}')
        elif char == '[':
            stack.append(']')
        elif char in ['}', ']']:
            if stack and stack[-1] == char:
                stack.pop()
            elif not stack:
                # Extra closing bracket, might be malformed
                break
        i += 1
    
    # Add missing closing brackets/braces in reverse order
    missing = ''.join(reversed(stack))
    
    # Also add any missing based on counts (fallback)
    if not missing:
        missing_braces = open_braces - close_braces
        missing_brackets = open_brackets - close_brackets
        missing = ']' * missing_brackets + '}' * missing_braces
    
    if missing:
        # If last char is comma, remove it before adding closing brackets
        if trimmed[-1] == ',':
            trimmed = trimmed[:-1]
        return trimmed + missing
    
    return None

def evaluate(dataset, valid_entries, retriever_fn, top_ks):
    start = time.time()

    all_queries = []
    query_meta = []  # (ori_id, gold_sources)
    for idx in valid_entries:
        entry = dataset[idx]
        ori_id = entry.ori_id
        for q in entry.questions:
            all_queries.append(q["question"])
            query_meta.append((ori_id, {ori_id}))

    # 一次性召回最大 top_k
    max_k = max(top_ks)
    all_results = retriever_fn(all_queries, top_k=max_k)

    results_by_k = {}

    for top_k in top_ks:
        total_tp, total_fp, total_fn = 0, 0, 0

        for (query, (ori_id, gold_sources), result) in zip(all_queries, query_meta, all_results):
            pred_chunks = result["associated_chunks"][:top_k]  # 截断
            pred_chunks = set(pred_chunks)

            hits = gold_sources & pred_chunks
            total_tp += len(hits)
            total_fp += len(pred_chunks - gold_sources)
            total_fn += len(gold_sources - hits)

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        results_by_k[top_k] = {
            "top_k": top_k,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "total_queries": len(all_queries),
            "total_tp": total_tp,
            "total_fp": total_fp,
            "total_fn": total_fn,
        }

    end = time.time()
    print(f"Evaluation Time elapsed: {end - start:.2f} seconds")
    return results_by_k

class TokenizerInterface(Protocol):
    """
    Defines the interface for a tokenizer, requiring encode and decode methods.
    """

    def encode(self, content: str) -> List[int]:
        """Encodes a string into a list of tokens."""
        ...

    def decode(self, tokens: List[int]) -> str:
        """Decodes a list of tokens into a string."""
        ...

class Tokenizer:
    """
    A wrapper around a tokenizer to provide a consistent interface for encoding and decoding.
    """

    def __init__(self, model_name: str, tokenizer: TokenizerInterface):
        """
        Initializes the Tokenizer with a tokenizer model name and a tokenizer instance.

        Args:
            model_name: The associated model name for the tokenizer.
            tokenizer: An instance of a class implementing the TokenizerInterface.
        """
        self.model_name: str = model_name
        self.tokenizer: TokenizerInterface = tokenizer

    def encode(self, content: str) -> List[int]:
        """
        Encodes a string into a list of tokens using the underlying tokenizer.

        Args:
            content: The string to encode.

        Returns:
            A list of integer tokens.
        """
        return self.tokenizer.encode(content)

    def decode(self, tokens: List[int]) -> str:
        """
        Decodes a list of tokens into a string using the underlying tokenizer.

        Args:
            tokens: A list of integer tokens to decode.

        Returns:
            The decoded string.
        """
        return self.tokenizer.decode(tokens)
logger = logging.getLogger("lightrag")
logger.propagate = False  # prevent log message send to root loggger
# Let the main application configure the handlers
logger.setLevel(logging.INFO)

def get_env_value(
    env_key: str, default: any, value_type: type = str, special_none: bool = False
) -> any:
    """
    Get value from environment variable with type conversion

    Args:
        env_key (str): Environment variable key
        default (any): Default value if env variable is not set
        value_type (type): Type to convert the value to
        special_none (bool): If True, return None when value is "None"

    Returns:
        any: Converted value from environment or default
    """
    value = os.getenv(env_key)
    if value is None:
        return default

    # Handle special case for "None" string
    if special_none and value == "None":
        return None

    if value_type is bool:
        return value.lower() in ("true", "1", "yes", "t", "on")
    try:
        return value_type(value)
    except (ValueError, TypeError):
        return default

def setup_logger(
    logger_name: str,
    level: str = "INFO",
    add_filter: bool = False,
    log_file_path: str | None = None,
    enable_file_logging: bool = True,
):
    """Set up a logger with console and optionally file handlers

    Args:
        logger_name: Name of the logger to set up
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        add_filter: Whether to add LightragPathFilter to the logger
        log_file_path: Path to the log file. If None and file logging is enabled, defaults to lightrag.log in LOG_DIR or cwd
        enable_file_logging: Whether to enable logging to a file (defaults to True)
    """
    # Configure formatters
    detailed_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    simple_formatter = logging.Formatter("%(levelname)s: %(message)s")

    logger_instance = logging.getLogger(logger_name)
    logger_instance.setLevel(level)
    logger_instance.handlers = []  # Clear existing handlers
    logger_instance.propagate = False

    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(simple_formatter)
    console_handler.setLevel(level)
    logger_instance.addHandler(console_handler)

    # Add file handler by default unless explicitly disabled
    if enable_file_logging:
        # Get log file path
        if log_file_path is None:
            log_dir = os.getenv("LOG_DIR", os.getcwd())
            log_file_path = os.path.abspath(os.path.join(log_dir, DEFAULT_LOG_FILENAME))

        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

        # Get log file max size and backup count from environment variables
        log_max_bytes = get_env_value("LOG_MAX_BYTES", DEFAULT_LOG_MAX_BYTES, int)
        log_backup_count = get_env_value(
            "LOG_BACKUP_COUNT", DEFAULT_LOG_BACKUP_COUNT, int
        )

        try:
            # Add file handler
            file_handler = logging.handlers.RotatingFileHandler(
                filename=log_file_path,
                maxBytes=log_max_bytes,
                backupCount=log_backup_count,
                encoding="utf-8",
            )
            file_handler.setFormatter(detailed_formatter)
            file_handler.setLevel(level)
            logger_instance.addHandler(file_handler)
        except PermissionError as e:
            logger.warning(f"Could not create log file at {log_file_path}: {str(e)}")
            logger.warning("Continuing with console logging only")
