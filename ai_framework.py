"""
AI Framework for Patient Communication Scenarios
Supports two modes:
  - simulation: Returns realistic mock responses (no API key needed)
  - live: Calls Azure OpenAI with rate-limit retry logic

"""

import json
import random
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Enums & Data Classes
# ---------------------------------------------------------------------------


class AIUseCaseType(Enum):
    AUTHORSHIP_DETECTION = "authorship_detection"
    RESPONSE_GENERATION = "response_generation"
    CRITICALITY_ANALYSIS = "criticality_analysis"
    CATEGORIZATION = "categorization"


class MessageTarget(Enum):
    PATIENT = "patient"
    PROVIDER = "provider"
    BOTH = "both"
    FIRST_PATIENT_PER_SESSION = "first_patient_per_session"


@dataclass
class AIResult:
    use_case_type: AIUseCaseType
    message_id: str
    result: Any
    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float
    timestamp: datetime = field(default_factory=datetime.now)
    error: Optional[str] = None


@dataclass
class TokenCostCalculator:
    model: str = "gpt-4"
    prompt_price_per_1k: float = 0.03
    completion_price_per_1k: float = 0.06

    def count_tokens(self, text: str) -> int:
        """Approximate token count (4 chars ~ 1 token).
        If tiktoken is available, use the real encoder."""
        try:
            import tiktoken

            enc = tiktoken.encoding_for_model(self.model)
            return len(enc.encode(text))
        except Exception:
            return max(1, len(text) // 4)

    def calculate_cost(
        self, input_tokens: int, output_tokens: int
    ) -> Tuple[float, float, float]:
        inp = (input_tokens / 1000) * self.prompt_price_per_1k
        out = (output_tokens / 1000) * self.completion_price_per_1k
        return inp, out, inp + out


@dataclass
class AIConfiguration:
    enabled_use_cases: List[AIUseCaseType] = field(default_factory=list)
    use_case_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def enable_use_case(
        self, uc: AIUseCaseType, config: Optional[Dict[str, Any]] = None
    ):
        if uc not in self.enabled_use_cases:
            self.enabled_use_cases.append(uc)
        if config:
            self.use_case_configs[uc.value] = config

    def disable_use_case(self, uc: AIUseCaseType):
        if uc in self.enabled_use_cases:
            self.enabled_use_cases.remove(uc)
        self.use_case_configs.pop(uc.value, None)

    def is_enabled(self, uc: AIUseCaseType) -> bool:
        return uc in self.enabled_use_cases

    def get_config(self, uc: AIUseCaseType) -> Dict[str, Any]:
        return self.use_case_configs.get(uc.value, {})


# ---------------------------------------------------------------------------
# Base Use Case
# ---------------------------------------------------------------------------


class BaseAIUseCase(ABC):
    def __init__(
        self,
        mode: str = "simulation",
        api_key: str = "",
        azure_endpoint: str = "",
        api_version: str = "",
        model: str = "gpt-4",
        cost_calculator: Optional[TokenCostCalculator] = None,
    ):
        self.mode = mode
        self.model = model
        self.cost_calculator = cost_calculator or TokenCostCalculator(model=model)
        self.results_cache: Dict[str, AIResult] = {}
        self._client = None

        self._custom_prompt: Optional[str] = None

        if mode == "live" and api_key:
            try:
                from openai import AzureOpenAI

                self._client = AzureOpenAI(
                    api_key=api_key,
                    api_version=api_version,
                    azure_endpoint=azure_endpoint,
                )
            except ImportError:
                raise RuntimeError(
                    "openai package required for live mode: pip install openai"
                )

    @property
    @abstractmethod
    def use_case_type(self) -> AIUseCaseType: ...

    @property
    @abstractmethod
    def applies_to(self) -> MessageTarget: ...

    @property
    @abstractmethod
    def prompt_template(self) -> str:
        """Return the default prompt template for this use case."""
        ...

    def get_active_prompt(self) -> str:
        """Return custom prompt if set, otherwise the default template."""
        return self._custom_prompt if self._custom_prompt else self.prompt_template

    def set_custom_prompt(self, prompt: Optional[str]):
        """Override the default prompt template. Pass None to reset."""
        self._custom_prompt = prompt
        self.clear_cache()  # Clear cached results since prompt changed

    def format_prompt(self, message_data: Dict[str, Any]) -> str:
        """Format the active prompt template with actual message data."""
        email_text = message_data.get("email_text", "")
        return self.get_active_prompt().format(message=email_text)

    @abstractmethod
    def parse_response(self, response: str) -> Any: ...

    @abstractmethod
    def simulate_response(self, message_data: Dict[str, Any]) -> str:
        """Return a mock AI response string for simulation mode."""
        ...

    def get_max_tokens(self) -> int:
        return 150

    def should_process_message(
        self, message: Dict[str, Any], session_info=None
    ) -> bool:
        sender = message.get("sender", "").upper()
        if self.applies_to == MessageTarget.PATIENT:
            return sender in ("PATIENT", "CAREGIVER")
        if self.applies_to == MessageTarget.PROVIDER and sender != "PROVIDER":
            return False
        if self.applies_to == MessageTarget.FIRST_PATIENT_PER_SESSION:
            if sender not in ("PATIENT", "CAREGIVER"):
                return False
            if session_info:
                return message.get("message_order") == 1
        return True

    # --- API call with rate-limit retry ------------------------------------

    def call_ai(
        self, prompt: str, retry_count: int = 0, max_retries: int = 3
    ) -> Tuple[str, int, int]:
        """Call the AI API with rate-limit retry logic."""
        input_tokens = self.cost_calculator.count_tokens(prompt)

        if self.mode == "simulation" or self._client is None:
            raise RuntimeError("Should not call_ai in simulation mode")

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=self.get_max_tokens(),
            )
            output = response.choices[0].message.content.strip()
            output_tokens = response.usage.completion_tokens
            return output, input_tokens, output_tokens

        except Exception as e:
            error_str = str(e)
            # Rate limit handling (HTTP 429)
            if "429" in error_str or "rate limit" in error_str.lower():
                if retry_count < max_retries:
                    wait_time = 1
                    match = re.search(r"retry after (\d+) second", error_str)
                    if match:
                        wait_time = int(match.group(1))
                    print(
                        f"Rate limit hit. Waiting {wait_time}s before retry "
                        f"{retry_count + 1}/{max_retries}"
                    )
                    time.sleep(wait_time)
                    return self.call_ai(prompt, retry_count + 1, max_retries)
                else:
                    print(f"Max retries ({max_retries}) exceeded for rate limit")
                    raise
            raise

    # --- Message processing ------------------------------------------------

    def process_message(self, message: Dict[str, Any], session_info=None) -> AIResult:
        message_id = message.get("message_id", "")
        cache_key = f"{self.use_case_type.value}_{message_id}"
        if cache_key in self.results_cache:
            return self.results_cache[cache_key]

        try:
            if self.mode == "simulation":
                raw = self.simulate_response(message)
                prompt_text = self.format_prompt(message)
                input_tokens = self.cost_calculator.count_tokens(prompt_text)
                output_tokens = self.cost_calculator.count_tokens(raw)
            else:
                prompt_text = self.format_prompt(message)
                raw, input_tokens, output_tokens = self.call_ai(prompt_text)

            parsed = self.parse_response(raw)
            ic, oc, tc = self.cost_calculator.calculate_cost(
                input_tokens, output_tokens
            )
            result = AIResult(
                use_case_type=self.use_case_type,
                message_id=message_id,
                result=parsed,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                input_cost=ic,
                output_cost=oc,
                total_cost=tc,
            )
            self.results_cache[cache_key] = result
            return result

        except Exception as e:
            return AIResult(
                use_case_type=self.use_case_type,
                message_id=message_id,
                result=None,
                input_tokens=0,
                output_tokens=0,
                input_cost=0,
                output_cost=0,
                total_cost=0,
                error=str(e),
            )

    def process_messages(
        self,
        messages: List[Dict[str, Any]],
        session_mapping: Optional[Dict[int, List[str]]] = None,
    ) -> List[AIResult]:
        """Process multiple messages, respecting session-level targeting."""
        results = []
        if (
            self.applies_to == MessageTarget.FIRST_PATIENT_PER_SESSION
            and session_mapping
        ):
            for session_num, message_ids in session_mapping.items():
                session_msgs = [
                    m for m in messages if m.get("message_id") in message_ids
                ]
                for msg in sorted(
                    session_msgs, key=lambda x: x.get("message_order", 0)
                ):
                    if msg.get("sender", "").upper() in ("PATIENT", "CAREGIVER"):
                        info = {"session_number": session_num, "message_order": 1}
                        if self.should_process_message(msg, info):
                            results.append(self.process_message(msg, info))
                        break
        else:
            for msg in messages:
                if self.should_process_message(msg):
                    results.append(self.process_message(msg))
        return results

    def get_total_cost(self) -> float:
        return sum(r.total_cost for r in self.results_cache.values())

    def clear_cache(self):
        self.results_cache.clear()


# ---------------------------------------------------------------------------
# Authorship Detection
# ---------------------------------------------------------------------------


class AuthorshipDetection(BaseAIUseCase):
    @property
    def use_case_type(self):
        return AIUseCaseType.AUTHORSHIP_DETECTION

    @property
    def applies_to(self):
        return MessageTarget.FIRST_PATIENT_PER_SESSION

    @property
    def prompt_template(self) -> str:
        return (
            "You are a clinical assistant. Classify whether the following message "
            "is sent by a patient or a care partner.\n\n"
            "Examples:\n"
            "message: 'my father is not doing well and i'd like an update on his meds.'\n"
            "output: 1\n\n"
            "message: 'i forgot to take my medication yesterday.'\n"
            "output: 0\n\n"
            "message: 'update on condition'\n"
            "output: 2\n\n"
            "Instructions:\n"
            "- Output 0 if the message is from the patient themselves\n"
            "- Output 1 if the message is from a caretaker/care partner "
            "(family member, friend, etc.)\n"
            "- Output 2 if it's ambiguous and you cannot determine with confidence\n\n"
            "message: '{message}'\n"
            "output:"
        )

    def simulate_response(self, msg: Dict[str, Any]) -> str:
        text = msg.get("email_text", "").lower()
        caregiver_signals = [
            "my father",
            "my mother",
            "my husband",
            "my wife",
            "my dad",
            "mom ",
            "dad ",
            "my grandfather",
            "my grandmother",
            "as primary caregiver",
            "my parent",
            "my spouse",
            "his medication",
            "her medication",
            "doesn't recognize me",
        ]
        patient_signals = [
            "i've been",
            "i forgot",
            "my medication",
            "i'm experiencing",
            "i noticed",
            "i'm worried",
            "i have",
            "i need to",
        ]
        for s in caregiver_signals:
            if s in text:
                return "1"
        for s in patient_signals:
            if s in text:
                return "0"
        return "2"

    def parse_response(self, response: str) -> Dict[str, Any]:
        try:
            val = int(response.strip())
            if val not in (0, 1, 2):
                val = 2
        except ValueError:
            val = 2
        labels = {0: "patient", 1: "caretaker", 2: "ambiguous"}
        return {"classification": val, "label": labels[val]}

    def get_max_tokens(self):
        return 10


# ---------------------------------------------------------------------------
# Criticality Analysis
# ---------------------------------------------------------------------------


class CriticalityAnalysis(BaseAIUseCase):
    @property
    def use_case_type(self):
        return AIUseCaseType.CRITICALITY_ANALYSIS

    @property
    def applies_to(self):
        return MessageTarget.PATIENT

    @property
    def prompt_template(self) -> str:
        return (
            "You are a healthcare triage assistant. Analyze the following patient "
            "message and rate its urgency/criticality on a scale of 1-5.\n\n"
            "Scale:\n"
            "1 - Routine (general questions, appointment scheduling)\n"
            "2 - Low priority (medication refills, test results inquiry)\n"
            "3 - Moderate (mild symptoms, follow-up needed)\n"
            "4 - High priority (concerning symptoms, medication issues)\n"
            "5 - Urgent (severe symptoms, immediate attention needed)\n\n"
            "Consider factors like:\n"
            "- Severity of symptoms mentioned\n"
            "- Emotional distress indicators\n"
            "- Time-sensitive medication issues\n"
            "- Safety concerns\n\n"
            'Patient message: "{message}"\n\n'
            "Output only a number from 1 to 5:"
        )

    def simulate_response(self, msg: Dict[str, Any]) -> str:
        text = msg.get("email_text", "").lower()
        urgent = [
            "emergency",
            "severe",
            "aggressive",
            "immediate",
            "chest pain",
            "can't breathe",
            "unconscious",
            "seizure",
            "wandered out",
            "fall",
        ]
        high = [
            "agitated",
            "confused",
            "doesn't recognize",
            "overdose",
            "swallowing",
            "lost weight",
            "dizzy",
            "nauseous",
            "side effect",
            "sundowning",
        ]
        moderate = [
            "worse",
            "not eating",
            "nightmares",
            "sleeping",
            "progression",
            "worried",
            "concerns",
            "pill organizer",
        ]
        low = ["refill", "follow-up", "schedule", "results", "check-up"]
        for kw in urgent:
            if kw in text:
                return "5"
        for kw in high:
            if kw in text:
                return "4"
        for kw in moderate:
            if kw in text:
                return "3"
        for kw in low:
            if kw in text:
                return "2"
        return "3"

    def parse_response(self, response: str) -> Dict[str, Any]:
        try:
            score = int(response.strip())
            score = max(1, min(5, score))
        except ValueError:
            score = 3
        labels = {
            1: "Routine",
            2: "Low Priority",
            3: "Moderate",
            4: "High Priority",
            5: "Urgent",
        }
        return {"criticality_score": score, "criticality_label": labels[score]}

    def get_max_tokens(self):
        return 10


# ---------------------------------------------------------------------------
# Message Categorization
# ---------------------------------------------------------------------------


class MessageCategorization(BaseAIUseCase):
    @property
    def use_case_type(self):
        return AIUseCaseType.CATEGORIZATION

    @property
    def applies_to(self):
        return MessageTarget.PATIENT

    @property
    def prompt_template(self) -> str:
        return (
            "You are a healthcare triage assistant. Categorize the following patient "
            "message as either clinical or non-clinical.\n\n"
            "Non-clinical (0) - Administrative matters that can be handled by "
            "administrative staff:\n"
            "- Appointment scheduling, rescheduling, or cancellation\n"
            "- Insurance questions or billing inquiries\n"
            "- General facility information (hours, location, parking)\n"
            "- Requesting forms or paperwork\n"
            "- Technical issues with patient portal\n"
            "- Non-medical administrative requests\n\n"
            "Clinical (1) - Medical matters that require healthcare provider attention:\n"
            "- Symptoms, pain, or health concerns\n"
            "- Medication questions, side effects, or refill requests\n"
            "- Test results questions or interpretation\n"
            "- Medical advice requests\n"
            "- Treatment plan discussions\n"
            "- Follow-up on medical procedures\n"
            "- Emergency or urgent medical situations\n\n"
            "Examples:\n"
            "message: 'I need to schedule an appointment for next week'\n"
            "output: 0\n\n"
            "message: 'I'm experiencing chest pain and shortness of breath'\n"
            "output: 1\n\n"
            "message: 'Can you refill my blood pressure medication?'\n"
            "output: 1\n\n"
            "message: 'What are your office hours on weekends?'\n"
            "output: 0\n\n"
            "message: 'I have questions about my insurance coverage'\n"
            "output: 0\n\n"
            "message: 'I'm having side effects from my new medication'\n"
            "output: 1\n\n"
            "Instructions:\n"
            "- Output 0 for non-clinical (administrative) messages\n"
            "- Output 1 for clinical (medical) messages\n\n"
            "message: '{message}'\n"
            "output:"
        )

    def simulate_response(self, msg: Dict[str, Any]) -> str:
        text = msg.get("email_text", "").lower()
        nonclinical = [
            "reschedule",
            "appointment",
            "insurance",
            "billing",
            "fax",
            "password",
            "mychart",
            "visiting hours",
            "records",
            "support group",
            "respite care",
            "portal",
            "office hours",
            "parking",
        ]
        for kw in nonclinical:
            if kw in text:
                return "0"
        return "1"

    def parse_response(self, response: str) -> Dict[str, Any]:
        try:
            val = int(response.strip())
            if val not in (0, 1):
                val = 1  # Default to clinical for safety
        except ValueError:
            val = 1
        cat = {0: "Non-Clinical", 1: "Clinical"}
        handler = {0: "Administrative Staff", 1: "Healthcare Provider"}
        return {
            "category_code": val,
            "category_label": cat[val],
            "recommended_handler": handler[val],
        }

    def get_max_tokens(self):
        return 10


# ---------------------------------------------------------------------------
# Response Generation
# ---------------------------------------------------------------------------


class ResponseGeneration(BaseAIUseCase):
    @property
    def use_case_type(self):
        return AIUseCaseType.RESPONSE_GENERATION

    @property
    def applies_to(self):
        return MessageTarget.PATIENT

    @property
    def prompt_template(self) -> str:
        return (
            "You are a healthcare provider assistant. Generate a professional, "
            "empathetic response to the following patient message.\n\n"
            "Guidelines:\n"
            "- Be professional and compassionate\n"
            "- Acknowledge their concerns\n"
            "- Provide helpful information when appropriate\n"
            "- If medical advice is needed, suggest scheduling an appointment\n"
            "- Keep the response very short, concise, clear and short\n\n"
            'Patient message: "{message}"\n\n'
            "Response:"
        )

    def simulate_response(self, msg: Dict[str, Any]) -> str:
        text = msg.get("email_text", "").lower()
        templates = [
            "Thank you for reaching out. I understand your concern. Based on what "
            "you've described, I'd recommend scheduling a follow-up appointment so "
            "we can assess the situation. In the meantime, please continue the "
            "current medication regimen and contact our urgent line if symptoms "
            "worsen significantly.",
            "I appreciate you keeping us informed. The changes you're describing are "
            "important for us to evaluate. Please bring the patient in this week for "
            "an assessment. Our nurse will call to schedule a convenient time. If you "
            "notice any sudden worsening, please call our office immediately.",
            "Thank you for your message. I've noted your concerns and would like to "
            "review them in more detail. Let's schedule a telehealth visit to discuss "
            "the current situation and adjust the care plan as needed. Our office will "
            "reach out to coordinate scheduling.",
        ]
        idx = hash(text) % len(templates)
        return templates[idx]

    def parse_response(self, response: str) -> Dict[str, Any]:
        return {
            "generated_response": response.strip(),
            "timestamp": datetime.now().isoformat(),
        }

    def should_process_message(self, message, session_info=None):
        if not super().should_process_message(message, session_info):
            return False
        return message.get("message_type", "") == "Patient Medical Advice Request"

    def get_max_tokens(self):
        return 300


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class AIUseCaseFactory:
    _map = {
        AIUseCaseType.AUTHORSHIP_DETECTION: AuthorshipDetection,
        AIUseCaseType.RESPONSE_GENERATION: ResponseGeneration,
        AIUseCaseType.CRITICALITY_ANALYSIS: CriticalityAnalysis,
        AIUseCaseType.CATEGORIZATION: MessageCategorization,
    }

    @staticmethod
    def create_use_case(
        use_case_type,
        mode: str = "simulation",
        api_key: str = "",
        azure_endpoint: str = "",
        api_version: str = "",
        model: str = "gpt-4",
        cost_calculator: Optional[TokenCostCalculator] = None,
    ) -> "BaseAIUseCase":
        if isinstance(use_case_type, str):
            try:
                use_case_type = AIUseCaseType(use_case_type)
            except ValueError:
                raise ValueError(f"Unknown use case type string: {use_case_type}")
        elif not isinstance(use_case_type, AIUseCaseType):
            raise ValueError(
                f"Expected AIUseCaseType enum or string, got {type(use_case_type)}"
            )

        cls = AIUseCaseFactory._map.get(use_case_type)
        if not cls:
            raise ValueError(f"Unknown use case: {use_case_type}")
        return cls(
            mode=mode,
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            model=model,
            cost_calculator=cost_calculator,
        )


# ---------------------------------------------------------------------------
# Results Manager
# ---------------------------------------------------------------------------


class AIResultsManager:
    def __init__(self):
        self.results: Dict[str, List[AIResult]] = {}

    def add_results(self, scenario_id: str, results: List[AIResult]):
        self.results.setdefault(scenario_id, []).extend(results)

    def get_results(
        self, scenario_id: str, uc: Optional[AIUseCaseType] = None
    ) -> List[AIResult]:
        all_r = self.results.get(scenario_id, [])
        return [r for r in all_r if r.use_case_type == uc] if uc else all_r

    def get_cost_summary(self, scenario_id: str) -> Dict[str, Any]:
        results = self.get_results(scenario_id)
        if not results:
            return {
                "total_cost": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "messages_processed": 0,
                "by_use_case": {},
            }
        summary = {
            "total_cost": sum(r.total_cost for r in results),
            "total_input_tokens": sum(r.input_tokens for r in results),
            "total_output_tokens": sum(r.output_tokens for r in results),
            "messages_processed": len(results),
            "by_use_case": {},
        }
        for uct in AIUseCaseType:
            uc_results = [r for r in results if r.use_case_type == uct]
            if uc_results:
                summary["by_use_case"][uct.value] = {
                    "count": len(uc_results),
                    "total_cost": sum(r.total_cost for r in uc_results),
                    "avg_cost": sum(r.total_cost for r in uc_results) / len(uc_results),
                    "total_input_tokens": sum(r.input_tokens for r in uc_results),
                    "total_output_tokens": sum(r.output_tokens for r in uc_results),
                }
        return summary

    def get_all_scenarios_summary(self) -> Dict[str, Any]:
        all_results = []
        for rs in self.results.values():
            all_results.extend(rs)
        if not all_results:
            return {
                "total_scenarios": 0,
                "total_cost": 0,
                "avg_cost_per_scenario": 0,
                "most_expensive_use_case": None,
            }
        uc_costs: Dict[str, float] = {}
        for r in all_results:
            uc = r.use_case_type.value
            uc_costs[uc] = uc_costs.get(uc, 0) + r.total_cost
        most = max(uc_costs.items(), key=lambda x: x[1]) if uc_costs else (None, 0)
        return {
            "total_scenarios": len(self.results),
            "total_cost": sum(r.total_cost for r in all_results),
            "avg_cost_per_scenario": sum(r.total_cost for r in all_results)
            / len(self.results),
            "avg_cost_per_message": sum(r.total_cost for r in all_results)
            / len(all_results),
            "most_expensive_use_case": most[0],
            "use_case_breakdown": uc_costs,
        }

    def export_results(self, scenario_id: str) -> Dict[str, Any]:
        results = self.get_results(scenario_id)
        return {
            "scenario_id": scenario_id,
            "results": [
                {
                    "use_case_type": r.use_case_type.value,
                    "message_id": r.message_id,
                    "result": r.result,
                    "input_tokens": r.input_tokens,
                    "output_tokens": r.output_tokens,
                    "total_cost": r.total_cost,
                    "timestamp": r.timestamp.isoformat(),
                    "error": r.error,
                }
                for r in results
            ],
            "summary": self.get_cost_summary(scenario_id),
        }
