"""Gemini CLI client wrapper using subprocess"""
import json
import logging
import os
import shutil
import asyncio
import subprocess
from typing import Optional, Dict, Any

from .cache_manager import ContextCacheManager

# Set up logger for debugging
logger = logging.getLogger(__name__)


class GeminiClient:
    """Wrapper for Gemini CLI with caching and context management

    Uses the `gemini` CLI command instead of API calls.
    Requires Gemini CLI to be installed and authenticated.
    """

    def __init__(self, model: str = "auto"):
        # Validate CLI is installed
        cli_path = shutil.which("gemini")
        if not cli_path:
            raise RuntimeError(
                "Gemini CLI not found. Install with: npm install -g @google/gemini-cli\n"
                "Then authenticate with: gemini"
            )

        # Test CLI is working
        try:
            result = subprocess.run(
                ["gemini", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"Gemini CLI found but not working. Error: {result.stderr}"
                )
        except subprocess.TimeoutExpired:
            raise RuntimeError("Gemini CLI not responding (timeout)")
        except Exception as e:
            raise RuntimeError(f"Error testing Gemini CLI: {e}")

        self.model_name = model
        self.cli_path = cli_path

        # Initialize cache manager with configurable TTL
        ttl_minutes = int(os.getenv("CONTEXT_CACHE_TTL_MINUTES", "30"))
        self.cache_manager = ContextCacheManager(ttl_minutes=ttl_minutes)

    async def generate_content(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate content with Gemini CLI

        Note: temperature and max_tokens are not currently supported by CLI
        and are included for interface compatibility only.
        """
        # Build command
        cmd = [
            self.cli_path,
            "--output-format", "json",
            "-m", self.model_name,
            prompt
        ]

        try:
            # Execute CLI command asynchronously
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            # Wait for completion with timeout (5 minutes)
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=300.0  # 5 minutes
                )
            except asyncio.TimeoutError:
                await process.kill()
                await process.wait()
                raise RuntimeError("Gemini CLI request timed out after 5 minutes")

            # Add debug logging
            logger.debug(f"Gemini CLI returncode: {process.returncode}")
            logger.debug(f"Gemini CLI stdout length: {len(stdout)}")
            if stderr:
                logger.debug(f"Gemini CLI stderr: {stderr.decode('utf-8', errors='replace')}")

            # Check for errors
            if process.returncode != 0:
                error_msg = stderr.decode('utf-8', errors='replace').strip()
                raise RuntimeError(f"Gemini CLI error: {error_msg}")

            # Parse JSON response
            try:
                output = stdout.decode('utf-8', errors='replace')

                # Log raw output for debugging (truncated)
                logger.debug(f"Gemini CLI raw output (first 500 chars): {output[:500]}")

                # Validate output is not empty
                if not output or not output.strip():
                    raise RuntimeError(
                        "Gemini CLI returned empty response. "
                        "Check authentication with: gemini --version"
                    )

                result = json.loads(output)

                # Extract response text from CLI JSON format
                if isinstance(result, dict) and "response" in result:
                    response = result["response"]

                    # Validate response is not None
                    if response is None:
                        raise RuntimeError(
                            f"Gemini CLI returned None response. "
                            f"Raw output: {json.dumps(result)}"
                        )

                    # Validate response is not empty
                    if not response.strip():
                        raise RuntimeError(
                            f"Gemini CLI returned empty response string. "
                            f"Raw output: {json.dumps(result)}"
                        )

                    return response
                else:
                    # Non-dict response or missing "response" key
                    if not output.strip():
                        raise RuntimeError(
                            "Gemini CLI returned non-JSON empty output. "
                            "Check CLI status with: gemini --version"
                        )
                    # Fallback: return raw output if format unexpected
                    return output

            except json.JSONDecodeError as e:
                # If JSON parsing fails, return raw output
                output = stdout.decode('utf-8', errors='replace')

                # Validate decoded output
                if not output or not output.strip():
                    raise RuntimeError(
                        f"Gemini CLI returned invalid/empty output. "
                        f"JSON decode error: {str(e)}"
                    )

                return output

        except Exception as e:
            if isinstance(e, RuntimeError):
                raise
            raise RuntimeError(f"Error calling Gemini CLI: {e}")

    async def analyze_with_context(
        self,
        prompt: str,
        context: str,
        temperature: float = 0.7
    ) -> str:
        """Generate content with provided context

        Combines context and prompt into a single prompt for the CLI.
        """
        full_prompt = f"""Context:
{context}

Task:
{prompt}

Please provide a detailed, structured response."""

        return await self.generate_content(full_prompt, temperature)

    def cache_context(self, context_id: str, context: Dict[str, Any]) -> None:
        """Cache context for reuse (automatically sets as current context).

        Args:
            context_id: Unique identifier for context
            context: Context data to cache
        """
        self.cache_manager.cache_context(
            context_id,
            context,
            set_as_current=True  # Always set as current context
        )

    def get_cached_context(self, context_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached context (checks TTL expiration).

        Args:
            context_id: Context ID to retrieve

        Returns:
            Context data or None if not found/expired
        """
        return self.cache_manager.get_cached_context(context_id)

    def get_current_context(self) -> Optional[tuple[Dict[str, Any], str]]:
        """Get the current active context.

        Returns:
            Tuple of (context_data, context_id) or None if no current context/expired
        """
        return self.cache_manager.get_current_context()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics including hit rate
        """
        return self.cache_manager.get_stats()
