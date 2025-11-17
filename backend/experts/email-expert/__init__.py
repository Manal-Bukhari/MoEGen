"""
Email Expert Module

This module provides a complete email generation pipeline:
- QueryEnhancer: Uses Gemini to analyze and enhance user queries
- EmailExpert: Generates emails using Flan-T5
- OutputParser: Cleans and formats the generated output
- EmailPipeline: Orchestrates the full pipeline
"""

from .email_pipeline import EmailPipeline
from .email_expert import EmailExpert
from .query_enhancer import QueryEnhancer
from .output_parser import OutputParser

__all__ = [
    'EmailPipeline',
    'EmailExpert',
    'QueryEnhancer',
    'OutputParser'
]
