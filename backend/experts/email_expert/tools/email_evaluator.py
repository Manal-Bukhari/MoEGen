"""
Email Evaluator - Enhanced with strict validation and programmatic checks
"""

import logging
import json
import re
from typing import Dict, Any, List
from .base_tool import init_gemini_model, parse_json_response
from ..config import (
    EVALUATOR_PENALTY_PER_ISSUE, EVALUATOR_MAX_PENALTY,
    EVALUATOR_CRITICAL_ISSUE_SCORE_CAP, EVALUATOR_DEFAULT_SCORE,
    EVALUATOR_MAX_CRITICAL_ERRORS_DISPLAY
)

logger = logging.getLogger(__name__)


class EmailEvaluator:
    """Evaluates emails using hybrid approach: programmatic checks + LLM evaluation."""
    
    def __init__(self, api_key: str, threshold: float, max_retries: int, evaluator_model: str = None):
        """
        Initialize EmailEvaluator.
        
        Args:
            api_key: Gemini API key (required)
            threshold: Evaluation threshold score (required)
            max_retries: Maximum retry attempts (required)
            evaluator_model: Preferred model name (optional)
        """
        self.api_key = api_key
        self.threshold = threshold
        self.max_retries = max_retries
        
        # Initialize model using shared utility
        self.model = init_gemini_model(self.api_key, evaluator_model)
        self.enabled = self.model is not None
        
        if not self.enabled:
            logger.warning("Evaluator disabled: model initialization failed")
    
    def extract_dates_from_text(self, text: str) -> List[str]:
        """Extract dates from text in various formats."""
        dates = []
        
        # Pattern 1: 17-18 November, Nov 17-18, etc.
        pattern1 = r'\b(\d{1,2}[-–]\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)(?:\s+\d{4})?)\b'
        dates.extend(re.findall(pattern1, text, re.IGNORECASE))
        
        # Pattern 2: November 17-18, 2025
        pattern2 = r'\b((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}[-–]\d{1,2}(?:,?\s+\d{4})?)\b'
        dates.extend(re.findall(pattern2, text, re.IGNORECASE))
        
        # Pattern 3: Individual dates like November 17, Dec 10, 2025-11-17
        pattern3 = r'\b(\d{4}-\d{2}-\d{2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?)\b'
        dates.extend(re.findall(pattern3, text, re.IGNORECASE))
        
        return dates
    
    def check_recipient_match(self, prompt: str, email: str) -> Dict[str, Any]:
        """Check if recipient in email matches the prompt using LLM-based analysis."""
        issues = []
        
        # Use LLM to check recipient match dynamically
        if self.enabled:
            try:
                check_prompt = f"""Analyze if the recipient in the email matches what was requested in the prompt.

PROMPT: {prompt}

GENERATED EMAIL: {email}

Check if:
1. The recipient mentioned in the prompt matches the recipient addressed in the email
2. If prompt mentions a group (like HR, team, department), email should address that group, not a specific person
3. If prompt mentions a specific person, email should address that person

Return ONLY valid JSON:
{{
    "recipient_match": true/false,
    "issues": ["list any recipient mismatches found"],
    "prompt_recipient": "recipient mentioned in prompt",
    "email_recipient": "recipient addressed in email"
}}"""
                
                response = self.model.generate_content(check_prompt)
                result_text = response.text.strip()
                check_result = parse_json_response(result_text, "recipient check")
                
                if check_result:
                    if not check_result.get("recipient_match", True):
                        issues.extend(check_result.get("issues", []))
            except Exception as e:
                logger.debug(f"LLM recipient check failed: {e}, using basic check")
                # Fallback to basic check
                if "hr" in prompt.lower() or "human resource" in prompt.lower():
                    if "hr" not in email.lower() and "human resource" not in email.lower():
                        issues.append("Prompt mentions HR but email doesn't address HR")
        
        return {
            "passed": len(issues) == 0,
            "issues": issues
        }
    
    def check_date_match(self, prompt: str, email: str) -> Dict[str, Any]:
        """Check if dates in email match the prompt."""
        prompt_dates = self.extract_dates_from_text(prompt)
        email_dates = self.extract_dates_from_text(email)
        
        issues = []
        
        if prompt_dates:
            logger.debug(f"Prompt dates: {prompt_dates}")
            logger.debug(f"Email dates: {email_dates}")
            
            # Check if email has dates
            if not email_dates:
                issues.append(f"Prompt specifies dates {prompt_dates} but email has no dates")
            else:
                # Check for month/day mismatch
                for prompt_date in prompt_dates:
                    date_found = False
                    for email_date in email_dates:
                        # Normalize and compare
                        if self._dates_match(prompt_date, email_date):
                            date_found = True
                            break
                    
                    if not date_found:
                        issues.append(f"Prompt date '{prompt_date}' not found in email. Email has: {email_dates}")
        
        return {
            "passed": len(issues) == 0,
            "issues": issues
        }
    
    def _dates_match(self, date1: str, date2: str) -> bool:
        """Check if two date strings refer to the same date(s)."""
        # Normalize both dates
        d1_norm = date1.lower().replace(',', '').replace('.', '')
        d2_norm = date2.lower().replace(',', '').replace('.', '')
        
        # Direct substring match
        if d1_norm in d2_norm or d2_norm in d1_norm:
            return True
        
        # Extract numbers and month names
        d1_nums = re.findall(r'\d+', d1_norm)
        d2_nums = re.findall(r'\d+', d2_norm)
        
        d1_month = re.search(r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)', d1_norm)
        d2_month = re.search(r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)', d2_norm)
        
        # Check if month matches and at least one number matches
        if d1_month and d2_month:
            if d1_month.group(1)[:3] == d2_month.group(1)[:3]:
                # Check if any numbers match
                if set(d1_nums) & set(d2_nums):
                    return True
        
        return False
    
    def check_context_match(self, prompt: str, email: str) -> Dict[str, Any]:
        """Check if email context matches the prompt intent using LLM-based analysis."""
        issues = []
        
        # Use LLM to check context match dynamically
        if self.enabled:
            try:
                check_prompt = f"""Analyze if the email context and intent match what was requested in the prompt.

PROMPT: {prompt}

GENERATED EMAIL: {email}

Check for:
1. Tense consistency - if prompt requests future action, email should use future/present tense, not past tense
2. Context match - if prompt is about requesting leave, email should request leave, not say it was already taken
3. Intent match - email should match the intent (request, inquiry, thank you, etc.) mentioned in prompt
4. Content relevance - email content should be relevant to what was requested

Return ONLY valid JSON:
{{
    "context_match": true/false,
    "issues": ["list any context/tense/intent mismatches found"],
    "prompt_intent": "intent detected in prompt",
    "email_intent": "intent detected in email"
}}"""
                
                response = self.model.generate_content(check_prompt)
                result_text = response.text.strip()
                check_result = parse_json_response(result_text, "context check")
                
                if check_result:
                    if not check_result.get("context_match", True):
                        issues.extend(check_result.get("issues", []))
            except Exception as e:
                logger.debug(f"LLM context check failed: {e}, skipping context check")
        
        return {
            "passed": len(issues) == 0,
            "issues": issues
        }
    
    def programmatic_checks(self, prompt: str, email: str) -> Dict[str, Any]:
        """Run all programmatic checks before LLM evaluation."""
        logger.debug(f"Running programmatic checks: prompt={len(prompt)} chars, email={len(email)} chars")
        all_issues = []
        
        # Recipient check
        recipient_check = self.check_recipient_match(prompt, email)
        if not recipient_check["passed"]:
            logger.warning(f"Recipient check failed: {recipient_check['issues']}")
            all_issues.extend(recipient_check["issues"])
        
        # Date check
        date_check = self.check_date_match(prompt, email)
        if not date_check["passed"]:
            logger.warning(f"Date check failed: {date_check['issues']}")
            all_issues.extend(date_check["issues"])
        
        # Context check
        context_check = self.check_context_match(prompt, email)
        if not context_check["passed"]:
            logger.warning(f"Context check failed: {context_check['issues']}")
            all_issues.extend(context_check["issues"])
        
        # Calculate penalty score
        penalty = min(len(all_issues) * EVALUATOR_PENALTY_PER_ISSUE, EVALUATOR_MAX_PENALTY)
        logger.info(f"Programmatic checks complete: {len(all_issues)} issues found, penalty: {penalty:.1f}")
        
        return {
            "passed": len(all_issues) == 0,
            "issues": all_issues,
            "penalty": penalty,
            "num_critical_issues": len(all_issues)
        }
    
    def evaluate(self, prompt: str, generated_email: str) -> Dict[str, Any]:
        """
        Evaluate email against prompt using hybrid approach.
        
        Returns dict with:
        - score (0-10)
        - passed (bool)
        - feedback (str)
        - criteria_scores (dict)
        """
        logger.info("EmailEvaluator.evaluate() called")
        logger.debug(f"Prompt: {prompt[:100]}..., Email length: {len(generated_email)} chars")
        
        if not self.enabled:
            logger.warning("Evaluator disabled, returning default pass")
            return {"score": EVALUATOR_DEFAULT_SCORE, "passed": True, "feedback": "Evaluator disabled"}
        
        # First run programmatic checks
        prog_checks = self.programmatic_checks(prompt, generated_email)
        
        if not prog_checks["passed"]:
            logger.warning("Programmatic checks failed:")
            for issue in prog_checks["issues"]:
                logger.warning(f"  - {issue}")
        
        # Enhanced LLM evaluation prompt
        eval_prompt = f"""You are a STRICT email evaluator. Evaluate this email against the user's request.

USER'S REQUEST: {prompt}

GENERATED EMAIL: {generated_email}

CRITICAL INSTRUCTIONS:
1. Compare dates EXACTLY - if request says "17-18 November" but email says "December 10", that's WRONG
2. Check recipient EXACTLY - if request says "HR" but email says "Mr. Smith", that's WRONG
3. Check context - if it's a sick leave request, it should REQUEST leave, not say "I have taken"
4. Check tense - requesting future leave should use future/present tense, not past tense
5. BE VERY STRICT - any mismatch in dates, recipients, or context = LOW SCORE

Score each criterion 0-10 (be harsh on errors):

1. COMPLETENESS: Are ALL specific details from request included correctly?
   - Dates must match EXACTLY (check day, month, year)
   - Recipient must match (HR vs manager vs specific person)
   - All mentioned items must be present (documentation, reason, etc.)
   
2. STRUCTURE: Proper email format?
   - Subject line
   - Greeting to correct recipient
   - Body with clear request
   - Professional closing
   
3. ACCURACY: Does content match request?
   - Dates are EXACTLY as requested (not different dates)
   - Recipient is EXACTLY as requested
   - Context matches (sick leave vs vacation vs job posting, etc.)
   
4. TONE: Professional and appropriate?
   - Not too casual or too formal
   - Appropriate for the situation
   
5. CLARITY: Clear and understandable?
   - No contradictions
   - Tense consistency
   - Logical flow

SCORING RULES:
- Wrong dates = automatic 0-2 for completeness and accuracy
- Wrong recipient = automatic 0-3 for completeness
- Wrong context (e.g., job posting instead of leave request) = automatic 0-2 for accuracy
- Past tense when should be requesting = -2 points from clarity

Return ONLY this JSON structure (no markdown, no backticks):
{{
    "completeness": <0-10>,
    "structure": <0-10>,
    "accuracy": <0-10>,
    "tone": <0-10>,
    "clarity": <0-10>,
    "overall_score": <average of above 5>,
    "feedback": "<detailed explanation of issues found>",
    "critical_errors": ["list any date mismatches, recipient errors, context errors"],
    "missing_elements": ["list missing details from request"]
}}

BE STRICT. If dates don't match, score must be low."""
        logger.debug(f"Evaluation prompt length: {len(eval_prompt)} chars")

        try:
            logger.info("Calling Gemini API for LLM evaluation...")
            response = self.model.generate_content(eval_prompt)
            result = response.text.strip()
            logger.debug(f"Raw response length: {len(result)} chars")
            
            # Use shared JSON parser utility
            evaluation = parse_json_response(result, "evaluation")
            if not evaluation:
                raise ValueError("Failed to parse evaluation JSON")
            
            llm_score = evaluation.get("overall_score", 0)
            logger.debug(f"LLM overall score: {llm_score:.1f}")
            
            # Apply programmatic penalty
            final_score = max(0, llm_score - prog_checks["penalty"])
            logger.debug(f"Score after penalty: {final_score:.1f}")
            
            # If programmatic checks found critical issues, cap score
            if prog_checks["num_critical_issues"] > 0:
                logger.warning(f"Critical issues found, capping score at {EVALUATOR_CRITICAL_ISSUE_SCORE_CAP}")
                final_score = min(final_score, EVALUATOR_CRITICAL_ISSUE_SCORE_CAP)
            
            passed = final_score >= self.threshold
            
            # Combine feedback
            combined_feedback = evaluation.get("feedback", "")
            if prog_checks["issues"]:
                combined_feedback = "CRITICAL ISSUES: " + "; ".join(prog_checks["issues"]) + ". " + combined_feedback
            
            logger.info(f"Evaluation complete: LLM Score={llm_score:.1f}, Penalty={prog_checks['penalty']:.1f}, Final Score={final_score:.1f}, Passed={passed}")
            if evaluation.get("critical_errors"):
                logger.warning(f"Critical errors: {evaluation['critical_errors']}")
            
            return {
                "score": final_score,
                "llm_score": llm_score,
                "penalty": prog_checks["penalty"],
                "passed": passed,
                "feedback": combined_feedback,
                "criteria_scores": {
                    "completeness": evaluation.get("completeness", 0),
                    "structure": evaluation.get("structure", 0),
                    "accuracy": evaluation.get("accuracy", 0),
                    "tone": evaluation.get("tone", 0),
                    "clarity": evaluation.get("clarity", 0)
                },
                "critical_errors": evaluation.get("critical_errors", []) + prog_checks["issues"],
                "missing_elements": evaluation.get("missing_elements", [])
            }
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}", exc_info=True)
            
            # If LLM fails, use programmatic checks only
            if not prog_checks["passed"]:
                logger.warning("Using programmatic checks only due to LLM failure")
                fallback_score = max(0, EVALUATOR_DEFAULT_SCORE - prog_checks["penalty"])
                return {
                    "score": fallback_score,
                    "passed": fallback_score >= self.threshold,
                    "feedback": "; ".join(prog_checks["issues"]),
                    "criteria_scores": {},
                    "critical_errors": prog_checks["issues"]
                }
            
            logger.warning("LLM evaluation failed, no programmatic issues found, returning default pass")
            return {"score": EVALUATOR_DEFAULT_SCORE, "passed": True, "feedback": f"Error: {e}"}