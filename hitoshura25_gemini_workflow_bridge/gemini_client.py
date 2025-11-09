"""Gemini API client wrapper"""
import os
from typing import Optional, Dict, Any
import google.generativeai as genai
from pathlib import Path
import json

class GeminiClient:
    """Wrapper for Gemini API with caching and context management"""

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.0-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not set")

        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model)
        self.context_cache: Dict[str, Any] = {}

    async def generate_content(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate content with Gemini"""
        generation_config = genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens
        )

        response = await self.model.generate_content_async(
            prompt,
            generation_config=generation_config
        )

        return response.text

    async def analyze_with_context(
        self,
        prompt: str,
        context: str,
        temperature: float = 0.7
    ) -> str:
        """Generate content with provided context"""
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
