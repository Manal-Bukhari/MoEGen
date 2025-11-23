"""
Email Evaluator - Enhanced with strict validation and programmatic checks
"""

import os
import logging
import json
import re
from typing import Dict, Any, Tuple, List
from datetime import datetime
import google.generativeai as genai

logger = logging.getLogger(__name__)


class EmailEvaluator:
    """Evaluates emails using hybrid approach: programmatic checks + LLM evaluation."""
    
    def __init__(self, api_key: str = None, threshold: float = None, max_retries: int = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.threshold = threshold if threshold is not None else float(os.getenv("EVALUATOR_THRESHOLD", "7.0"))
        self.max_retries = max_retries if max_retries is not None else int(os.getenv("EVALUATOR_MAX_RETRIES", "2"))
        
        # Get model name from environment
        evaluator_model = os.getenv("EVALUATOR_MODEL", "gemini-2.5-flash")
        
        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                # Try environment-specified model first, then fallback
                for model_name in [evaluator_model, "gemini-2.5-flash", "gemini-1.5-flash", "gemini-1.5-pro"]:
                    try:
                        self.model = genai.GenerativeModel(model_name)
                        logger.info(f"✅ Evaluator: {model_name}")
                        self.enabled = True
                        break
                    except:
                        continue
            except Exception as e:
                logger.warning(f"⚠️ Evaluator disabled: {e}")
                self.enabled = False
        else:
            logger.warning("⚠️ No GEMINI_API_KEY. Evaluator disabled.")
            self.enabled = False
    
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
        """Check if recipient in email matches the prompt."""
        prompt_lower = prompt.lower()
        email_lower = email.lower()
        
        issues = []
        
        # Check for HR mention
        if 'hr' in prompt_lower or 'human resource' in prompt_lower:
            if 'mr.' in email_lower or 'mrs.' in email_lower or 'ms.' in email_lower:
                # Check if it's a specific name after Mr/Mrs
                if re.search(r'mr\.|mrs\.|ms\..*?smith|jones|brown|taylor', email_lower):
                    issues.append("Email addressed to specific person (Mr./Mrs. Smith) instead of HR")
            if 'hr' not in email_lower and 'human resource' not in email_lower:
                issues.append("Prompt mentions HR but email doesn't address HR team")
        
        # Check for manager mention
        if 'manager' in prompt_lower:
            if 'manager' not in email_lower and 'supervisor' not in email_lower:
                issues.append("Prompt mentions manager but email doesn't address manager")
        
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
            logger.info(f"📅 Prompt dates: {prompt_dates}")
            logger.info(f"📅 Email dates: {email_dates}")
            
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
        """Check if email context matches the prompt intent."""
        prompt_lower = prompt.lower()
        email_lower = email.lower()
        
        issues = []
        
        # Sick leave specific checks
        if 'sick' in prompt_lower or 'leave' in prompt_lower or 'unwell' in prompt_lower:
            if 'taken sick leave' in email_lower:
                issues.append("Email says 'taken sick leave' (past tense) instead of requesting future leave")
            
            if 'interested in this position' in email_lower or 'fill in for me' in email_lower:
                issues.append("Email sounds like job posting/availability notice, not a sick leave request")
            
            if 'employer is' in email_lower or 'looking for someone' in email_lower:
                issues.append("Email mentions employer looking for replacement - wrong context for leave request")
            
            # Should mention being unwell if prompt mentions it
            if 'unwell' in prompt_lower or 'sick' in prompt_lower:
                if 'unwell' not in email_lower and 'sick' not in email_lower and 'ill' not in email_lower:
                    issues.append("Prompt mentions being unwell but email doesn't mention illness")
        
        # Check for tense consistency
        if 'request' in prompt_lower or 'requesting' in prompt_lower:
            if re.search(r'\b(have taken|took|had taken)\b', email_lower):
                issues.append("Email uses past tense instead of requesting future leave")
        
        return {
            "passed": len(issues) == 0,
            "issues": issues
        }
    
    def programmatic_checks(self, prompt: str, email: str) -> Dict[str, Any]:
        """Run all programmatic checks before LLM evaluation."""
        logger.info("🔍 EmailEvaluator.programmatic_checks() called")
        logger.debug(f"   Prompt length: {len(prompt)} chars, Email length: {len(email)} chars")
        all_issues = []
        
        # Recipient check
        logger.debug("👤 Checking recipient match...")
        recipient_check = self.check_recipient_match(prompt, email)
        if not recipient_check["passed"]:
            logger.warning(f"   Recipient check failed: {recipient_check['issues']}")
            all_issues.extend(recipient_check["issues"])
        else:
            logger.debug("   ✅ Recipient check passed")
        
        # Date check
        logger.debug("📅 Checking date match...")
        date_check = self.check_date_match(prompt, email)
        if not date_check["passed"]:
            logger.warning(f"   Date check failed: {date_check['issues']}")
            all_issues.extend(date_check["issues"])
        else:
            logger.debug("   ✅ Date check passed")
        
        # Context check
        logger.debug("📝 Checking context match...")
        context_check = self.check_context_match(prompt, email)
        if not context_check["passed"]:
            logger.warning(f"   Context check failed: {context_check['issues']}")
            all_issues.extend(context_check["issues"])
        else:
            logger.debug("   ✅ Context check passed")
        
        # Calculate penalty score
        penalty = min(len(all_issues) * 2.5, 8.0)  # Each issue = -2.5 points, max -8
        logger.info(f"📊 Programmatic checks complete: {len(all_issues)} issues found, penalty: {penalty:.1f}")
        
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
        logger.info("📊 EmailEvaluator.evaluate() called")
        logger.debug(f"   Prompt: {prompt[:100]}...")
        logger.debug(f"   Email length: {len(generated_email)} chars")
        
        if not self.enabled:
            logger.warning("⚠️ Evaluator disabled, returning default pass")
            return {"score": 10.0, "passed": True, "feedback": "Evaluator disabled"}
        
        # First run programmatic checks
        logger.info("🔍 Running programmatic checks...")
        prog_checks = self.programmatic_checks(prompt, generated_email)
        
        if not prog_checks["passed"]:
            logger.warning(f"❌ Programmatic checks failed:")
            for issue in prog_checks["issues"]:
                logger.warning(f"   - {issue}")
        else:
            logger.info("✅ All programmatic checks passed")
        
        # Enhanced LLM evaluation prompt
        logger.info("🤖 Building LLM evaluation prompt...")
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
        logger.debug(f"   Evaluation prompt length: {len(eval_prompt)} chars")

        try:
            logger.info("🤖 Calling Gemini API for LLM evaluation...")
            response = self.model.generate_content(eval_prompt)
            result = response.text.strip()
            logger.debug(f"   Raw response length: {len(result)} chars")
            
            # Extract JSON
            logger.debug("🔧 Extracting JSON from evaluation response...")
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0].strip()
                logger.debug("   Found JSON in ```json code block")
            elif "```" in result:
                result = result.split("```")[1].split("```")[0].strip()
                logger.debug("   Found JSON in ``` code block")
            
            # Clean and parse
            logger.debug("🧹 Cleaning and parsing JSON...")
            result = result.replace('\n', ' ')
            result = re.sub(r',(\s*[}\]])', r'\1', result)
            evaluation = json.loads(result)
            logger.debug(f"   Successfully parsed evaluation JSON")
            
            llm_score = evaluation.get("overall_score", 0)
            logger.debug(f"   LLM overall score: {llm_score:.1f}")
            
            # Apply programmatic penalty
            logger.debug(f"📉 Applying penalty: {prog_checks['penalty']:.1f}")
            final_score = max(0, llm_score - prog_checks["penalty"])
            logger.debug(f"   Score after penalty: {final_score:.1f}")
            
            # If programmatic checks found critical issues, cap score at 5.0
            if prog_checks["num_critical_issues"] > 0:
                logger.warning(f"   ⚠️ Critical issues found, capping score at 5.0")
                final_score = min(final_score, 5.0)
            
            passed = final_score >= self.threshold
            logger.debug(f"   Threshold: {self.threshold}, Passed: {passed}")
            
            # Combine feedback
            combined_feedback = evaluation.get("feedback", "")
            if prog_checks["issues"]:
                combined_feedback = "CRITICAL ISSUES: " + "; ".join(prog_checks["issues"]) + ". " + combined_feedback
            
            logger.info(f"✅ Evaluation complete:")
            logger.info(f"   LLM Score: {llm_score:.1f}")
            logger.info(f"   Penalty: {prog_checks['penalty']:.1f}")
            logger.info(f"   Final Score: {final_score:.1f}")
            logger.info(f"   Threshold: {self.threshold}, Passed: {passed}")
            if evaluation.get("critical_errors"):
                logger.warning(f"   Critical errors: {evaluation['critical_errors']}")
            
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
            logger.error(f"❌ Evaluation failed: {e}", exc_info=True)
            
            # If LLM fails, use programmatic checks only
            if not prog_checks["passed"]:
                logger.warning("⚠️ Using programmatic checks only due to LLM failure")
                fallback_score = max(0, 10.0 - prog_checks["penalty"])
                logger.info(f"   Fallback score: {fallback_score:.1f}")
                return {
                    "score": fallback_score,
                    "passed": fallback_score >= self.threshold,  # FIX: Check threshold, not hardcode False
                    "feedback": "; ".join(prog_checks["issues"]),
                    "criteria_scores": {},
                    "critical_errors": prog_checks["issues"]
                }
            
            logger.warning("⚠️ LLM evaluation failed, no programmatic issues found, returning default pass")
            return {"score": 10.0, "passed": True, "feedback": f"Error: {e}"}
    
    def evaluate_with_regeneration(
        self,
        prompt: str,
        generated_email: str,
        email_expert,
        enhanced_query: Dict[str, Any] = None,
        max_length: int = 300,  # ✅ Add parameter with default
        temperature: float = 0.5  # ✅ Add parameter with default
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Evaluate and regenerate if score < threshold.
        
        Returns:
            (final_email, evaluation_dict)
        """
        logger.info(f"🔄 EmailEvaluator.evaluate_with_regeneration() called")
        logger.info(f"   Max retries: {self.max_retries}, Threshold: {self.threshold}")
        attempt = 0
        current_email = generated_email
        all_evaluations = []
        
        while attempt <= self.max_retries:
            # Evaluate current email
            logger.info(f"📊 Evaluation attempt {attempt + 1}/{self.max_retries + 1}")
            evaluation = self.evaluate(prompt, current_email)
            all_evaluations.append(evaluation)
            
            logger.info(f"🔍 Attempt {attempt + 1}: Score={evaluation['score']:.1f}, Passed={evaluation['passed']}")
            
            # Check if passed
            if evaluation["passed"]:
                logger.info(f"✅ Passed on attempt {attempt + 1} (score: {evaluation['score']:.1f})")
                evaluation["attempts"] = attempt + 1
                evaluation["all_scores"] = [e["score"] for e in all_evaluations]
                return current_email, evaluation
            
            # Check max retries
            if attempt >= self.max_retries:
                logger.warning(f"⚠️ Max retries reached. Using best attempt.")
                best_idx = max(range(len(all_evaluations)), key=lambda i: all_evaluations[i]["score"])
                best_eval = all_evaluations[best_idx]
                best_eval["attempts"] = attempt + 1
                best_eval["all_scores"] = [e["score"] for e in all_evaluations]
                
                if best_idx == 0:
                    return generated_email, best_eval
                else:
                    return current_email, best_eval
            
            # Regenerate with feedback
            logger.info(f"🔄 Regenerating email (score {evaluation['score']:.1f} < {self.threshold})")
            logger.info(f"   Feedback: {evaluation['feedback'][:200]}...")
            if evaluation.get("critical_errors"):
                logger.warning(f"   Critical errors: {', '.join(evaluation['critical_errors'][:3])}")
            
            try:
                logger.debug(f"   Calling email_expert.generate() with max_length={max_length}, temperature={min(temperature + (attempt * 0.1), 1.0):.2f}")
                # ✅ Regenerate using passed parameters (from .env)
                current_email = email_expert.generate(
                    enhanced_query=enhanced_query,
                    prompt=prompt,
                    max_length=max_length,  # Use passed parameter
                    temperature=min(temperature + (attempt * 0.1), 1.0)  # Slightly increase temp per attempt
                )
                
                # Parse it
                logger.debug("   Parsing regenerated email...")
                from .output_parser import OutputParser
                parser = OutputParser()
                current_email = parser.parse(current_email, enhanced_query)
                
                logger.info(f"   ✅ Regenerated email: {len(current_email)} chars")
                
            except Exception as e:
                logger.error(f"❌ Regeneration failed: {e}")
                evaluation["attempts"] = attempt + 1
                evaluation["all_scores"] = [e["score"] for e in all_evaluations]
                return current_email, evaluation
            
            attempt += 1
        
        # Fallback
        evaluation = all_evaluations[-1]
        evaluation["attempts"] = attempt
        evaluation["all_scores"] = [e["score"] for e in all_evaluations]
        return current_email, evaluation


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # ✅ Use environment variable or default
    evaluator = EmailEvaluator()  # Will read from .env
    
    if not evaluator.enabled:
        logger.error("❌ No API key")
    else:
        logger.info("\n" + "="*60)
        logger.info("TEST 1: WRONG EMAIL (from your screenshot)")
        logger.info("="*60)
        
        wrong_email = """Subject: Sick Leave Request

Dear Mr. & Mrs. Smith, I am writing to inform you that I have taken sick leave from work. My employer is currently looking for someone to fill in for me. I would like to be available until the end of the week. I will be leaving on Monday, December 10th at 12:00 p.m. Please let me know if you are interested in this position. Thank you.

Best regards,
[Your Name]"""
        
        result = evaluator.evaluate(
            "Write a professional sick-leave email to HR requesting leave for 17-18 November 2025. Mention that I am unwell, will be unavailable for work, and will provide any required documentation. Keep the tone polite and concise.",
            wrong_email
        )
        
        logger.info(f"\n📊 LLM Score: {result.get('llm_score', 0):.1f}")
        logger.info(f"   Penalty: {result.get('penalty', 0):.1f}")
        logger.info(f"   Final Score: {result['score']:.1f}")
        logger.info(f"   Passed: {result['passed']}")
        logger.info(f"   Feedback: {result['feedback']}")
        if result.get('critical_errors'):
            logger.info(f"   Critical Errors:")
            for error in result['critical_errors']:
                logger.info(f"      - {error}")
        
        logger.info("\n" + "="*60)
        logger.info("TEST 2: CORRECT EMAIL")
        logger.info("="*60)
        
        good_email = """Subject: Sick Leave Request - November 17-18, 2025

Dear HR Team,

I am writing to request sick leave for November 17-18, 2025, as I am currently unwell and will be unavailable for work during this period.

I will provide any required medical documentation upon my return to the office.

Thank you for your understanding.

Best regards,
John Doe"""
        
        result2 = evaluator.evaluate(
            "Write a professional sick-leave email to HR requesting leave for 17-18 November 2025. Mention that I am unwell, will be unavailable for work, and will provide any required documentation. Keep the tone polite and concise.",
            good_email
        )
        
        logger.info(f"\n📊 LLM Score: {result2.get('llm_score', 0):.1f}")
        logger.info(f"   Penalty: {result2.get('penalty', 0):.1f}")
        logger.info(f"   Final Score: {result2['score']:.1f}")
        logger.info(f"   Passed: {result2['passed']}")
        logger.info(f"   Feedback: {result2['feedback']}")