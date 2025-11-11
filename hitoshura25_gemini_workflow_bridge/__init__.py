"""
hitoshura25-gemini-workflow-bridge

MCP server that bridges Claude Code to Gemini CLI for workflow tasks like codebase analysis, specification creation, and code review
"""

__version__ = "0.1.0"
__author__ = "Vinayak Menon"
__license__ = "Apache-2.0"

from .generator import (
    
    analyze_codebase_with_gemini,
    
    create_specification_with_gemini,
    
    review_code_with_gemini,
    
    generate_documentation_with_gemini,
    
    ask_gemini,
    
)

__all__ = [
    
    'analyze_codebase_with_gemini',
    
    'create_specification_with_gemini',
    
    'review_code_with_gemini',
    
    'generate_documentation_with_gemini',
    
    'ask_gemini',
    
]