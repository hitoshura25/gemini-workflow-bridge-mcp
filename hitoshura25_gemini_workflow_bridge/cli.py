#!/usr/bin/env python3
"""
CLI for hitoshura25-gemini-workflow-bridge.

MCP server that bridges Claude Code to Gemini CLI for workflow tasks like codebase analysis, specification creation, and code review
"""

import sys
import argparse
from .generator import (
    
    analyze_codebase_with_gemini,
    
    create_specification_with_gemini,
    
    review_code_with_gemini,
    
    generate_documentation_with_gemini,
    
    ask_gemini,
    
)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description='MCP server that bridges Claude Code to Gemini CLI for workflow tasks like codebase analysis, specification creation, and code review')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    
    # analyze_codebase_with_gemini command
    analyze_codebase_with_gemini_parser = subparsers.add_parser(
        'analyze_codebase_with_gemini',
        help='Analyze codebase using Gemini's 2M token context window'
    )
    
    analyze_codebase_with_gemini_parser.add_argument(
        '--focus_description',
        type=str,
        required=True,
        help='What to focus on in the analysis'
    )
    
    analyze_codebase_with_gemini_parser.add_argument(
        '--directories',
        type=str,
        help='Directories to analyze'
    )
    
    analyze_codebase_with_gemini_parser.add_argument(
        '--file_patterns',
        type=str,
        help='File patterns to include'
    )
    
    analyze_codebase_with_gemini_parser.add_argument(
        '--exclude_patterns',
        type=str,
        help='Patterns to exclude'
    )
    

    
    # create_specification_with_gemini command
    create_specification_with_gemini_parser = subparsers.add_parser(
        'create_specification_with_gemini',
        help='Generate detailed technical specification using full codebase context'
    )
    
    create_specification_with_gemini_parser.add_argument(
        '--feature_description',
        type=str,
        required=True,
        help='What feature to specify'
    )
    
    create_specification_with_gemini_parser.add_argument(
        '--context_id',
        type=str,
        help='Optional context ID from previous analysis'
    )
    
    create_specification_with_gemini_parser.add_argument(
        '--spec_template',
        type=str,
        help='Specification template to use'
    )
    
    create_specification_with_gemini_parser.add_argument(
        '--output_path',
        type=str,
        help='Where to save the spec'
    )
    

    
    # review_code_with_gemini command
    review_code_with_gemini_parser = subparsers.add_parser(
        'review_code_with_gemini',
        help='Comprehensive code review using Gemini'
    )
    
    review_code_with_gemini_parser.add_argument(
        '--files',
        type=str,
        help='Files to review'
    )
    
    review_code_with_gemini_parser.add_argument(
        '--review_focus',
        type=str,
        help='Areas to focus on'
    )
    
    review_code_with_gemini_parser.add_argument(
        '--spec_path',
        type=str,
        help='Path to spec to review against'
    )
    
    review_code_with_gemini_parser.add_argument(
        '--output_path',
        type=str,
        help='Where to save review'
    )
    

    
    # generate_documentation_with_gemini command
    generate_documentation_with_gemini_parser = subparsers.add_parser(
        'generate_documentation_with_gemini',
        help='Generate comprehensive documentation with full codebase context'
    )
    
    generate_documentation_with_gemini_parser.add_argument(
        '--documentation_type',
        type=str,
        required=True,
        help='Type of documentation'
    )
    
    generate_documentation_with_gemini_parser.add_argument(
        '--scope',
        type=str,
        required=True,
        help='What to document'
    )
    
    generate_documentation_with_gemini_parser.add_argument(
        '--output_path',
        type=str,
        help='Where to save documentation'
    )
    
    generate_documentation_with_gemini_parser.add_argument(
        '--include_examples',
        action='store_true',
        help='Include code examples'
    )
    

    
    # ask_gemini command
    ask_gemini_parser = subparsers.add_parser(
        'ask_gemini',
        help='General-purpose Gemini query with optional codebase context'
    )
    
    ask_gemini_parser.add_argument(
        '--prompt',
        type=str,
        required=True,
        help='Question or task for Gemini'
    )
    
    ask_gemini_parser.add_argument(
        '--include_codebase_context',
        action='store_true',
        help='Load full codebase context'
    )
    
    ask_gemini_parser.add_argument(
        '--context_id',
        type=str,
        help='Reuse cached context'
    )
    
    ask_gemini_parser.add_argument(
        '--temperature',
        type=float,
        help='Temperature for generation'
    )
    

    

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    try:
        
        if args.command == 'analyze_codebase_with_gemini':
            
            result = analyze_codebase_with_gemini(
                
                focus_description=args.focus_description,
                
                directories=args.directories,
                
                file_patterns=args.file_patterns,
                
                exclude_patterns=args.exclude_patterns
                
            )
            
            print(result)

        elif args.command == 'create_specification_with_gemini':
            
            result = create_specification_with_gemini(
                
                feature_description=args.feature_description,
                
                context_id=args.context_id,
                
                spec_template=args.spec_template,
                
                output_path=args.output_path
                
            )
            
            print(result)

        elif args.command == 'review_code_with_gemini':
            
            result = review_code_with_gemini(
                
                files=args.files,
                
                review_focus=args.review_focus,
                
                spec_path=args.spec_path,
                
                output_path=args.output_path
                
            )
            
            print(result)

        elif args.command == 'generate_documentation_with_gemini':
            
            result = generate_documentation_with_gemini(
                
                documentation_type=args.documentation_type,
                
                scope=args.scope,
                
                output_path=args.output_path,
                
                include_examples=args.include_examples
                
            )
            
            print(result)

        elif args.command == 'ask_gemini':
            
            result = ask_gemini(
                
                prompt=args.prompt,
                
                include_codebase_context=args.include_codebase_context,
                
                context_id=args.context_id,
                
                temperature=args.temperature
                
            )
            
            print(result)
        

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())