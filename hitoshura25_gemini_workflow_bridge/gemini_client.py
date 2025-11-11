"""Gemini CLI client wrapper using subprocess"""
import json
import shutil
import asyncio
import subprocess
from typing import Optional, Dict, Any


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
        self.context_cache: Dict[str, Any] = {}

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

            # Check for errors
            if process.returncode != 0:
                error_msg = stderr.decode('utf-8', errors='replace').strip()
                raise RuntimeError(f"Gemini CLI error: {error_msg}")

            # Parse JSON response
            try:
                output = stdout.decode('utf-8', errors='replace')
                result = json.loads(output)

                # Extract response text from CLI JSON format
                if isinstance(result, dict) and "response" in result:
                    return result["response"]
                else:
                    # Fallback: return raw output if format unexpected
                    return output

            except json.JSONDecodeError as e:
                # If JSON parsing fails, return raw output
                output = stdout.decode('utf-8', errors='replace')
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
        """Cache context for reuse"""
        self.context_cache[context_id] = context

    def get_cached_context(self, context_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached context"""
        return self.context_cache.get(context_id)
