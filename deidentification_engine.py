"""
De-identification Engine for Patient Communication Data
Removes and replaces PHI (Protected Health Information) with realistic fake data.

Two modes:
  - Lightweight (regex-only): No external NLP models required.
  - Full (spaCy + optional scispaCy): Better entity detection.

The mode is controlled via DeIdentificationConfig.use_spacy / use_scispacy.

Changelog:
  [2025-07-21] Integrated scispaCy (en_ner_bc5cdr_md) for auto drug detection; fallback to exclusion list.
               Fixed name structure issues: preserves original format (e.g., Nancy -> Belle).
               Prevents false matches with doctors' names and family links.
  [2025-07-17] Fixed improper de-ID of meds (e.g., Tramadol).
               Added medication exclusion list and config (exclude_medications).
               Improved case-insensitive filtering and expandability.
  [2025-07-14] Fixed incomplete name de-ID and real name collisions.
               Added synthetic name generation with config options:
               "synthetic" (recommended), "coded", "pattern", "faker".
"""

import re
import json
import random
import string
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple, Optional, Any, Set


# ---------------------------------------------------------------------------
# Optional NLP imports (graceful fallback)
# ---------------------------------------------------------------------------
_spacy_nlp = None
_sci_nlp = None


def _load_spacy():
    global _spacy_nlp
    if _spacy_nlp is None:
        try:
            import spacy
            _spacy_nlp = spacy.load("en_core_web_sm")
        except Exception:
            pass
    return _spacy_nlp


def _load_scispacy():
    global _sci_nlp
    if _sci_nlp is None:
        try:
            import scispacy  # noqa: F401
            import spacy
            _sci_nlp = spacy.load("en_ner_bc5cdr_md")
        except Exception:
            pass
    return _sci_nlp


# Optional Faker import
_faker = None
try:
    from faker import Faker
    _faker = Faker()
    Faker.seed(42)
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class EntityType(Enum):
    PERSON_NAME = "PERSON_NAME"
    EMAIL = "EMAIL"
    PHONE = "PHONE"
    SSN = "SSN"
    MRN = "MRN"
    ADDRESS = "ADDRESS"
    DATE = "DATE"
    ORGANIZATION = "ORGANIZATION"
    CREDIT_CARD = "CREDIT_CARD"
    URL = "URL"
    IP_ADDRESS = "IP_ADDRESS"
    FAX = "FAX"
    LICENSE = "LICENSE"
    INSURANCE_ID = "INSURANCE_ID"


@dataclass
class IdentifiedEntity:
    entity_type: EntityType
    original_text: str
    start_pos: int
    end_pos: int
    context: str = ""
    confidence: float = 1.0


@dataclass
class DeIdentificationConfig:
    # Entity detection flags
    detect_names: bool = True
    detect_emails: bool = True
    detect_phones: bool = True
    detect_ssn: bool = True
    detect_mrn: bool = True
    detect_addresses: bool = True
    detect_dates: bool = True
    detect_organizations: bool = True
    detect_credit_cards: bool = True
    detect_urls: bool = True

    # Replacement options
    preserve_length: bool = False
    preserve_format: bool = True
    consistent_replacement: bool = True
    shift_dates_by_days: int = -30

    # Gender preservation
    preserve_gender: bool = True

    # Name generation method: "synthetic", "coded", "pattern", "faker"
    name_generation_method: str = "synthetic"

    # NLP model toggles
    use_spacy: bool = False
    use_scispacy: bool = False

    # Organization handling
    replace_healthcare_orgs: bool = True
    healthcare_org_keywords: List[str] = field(default_factory=lambda: [
        "hospital", "clinic", "medical", "health", "center", "institute",
        "laboratory", "lab", "pharmacy", "surgical", "emergency", "urgent care"
    ])

    # Medication exclusion
    exclude_medications: bool = True
    use_scispacy_medication_detection: bool = True
    medication_exclusion_list: List[str] = field(default_factory=lambda: [
        # Pain medications
        "tramadol", "acetaminophen", "ibuprofen", "aspirin", "naproxen", "morphine",
        "oxycodone", "hydrocodone", "codeine", "fentanyl", "gabapentin", "pregabalin",

        # Antibiotics
        "amoxicillin", "azithromycin", "ciprofloxacin", "doxycycline", "cephalexin",
        "metronidazole", "clindamycin", "levofloxacin", "sulfamethoxazole", "trimethoprim",

        # Antidepressants / Mental Health
        "sertraline", "fluoxetine", "citalopram", "escitalopram", "paroxetine",
        "venlafaxine", "duloxetine", "bupropion", "trazodone", "mirtazapine",
        "alprazolam", "lorazepam", "clonazepam", "diazepam", "buspirone",

        # Cardiovascular
        "lisinopril", "amlodipine", "metoprolol", "atenolol", "carvedilol",
        "losartan", "valsartan", "hydrochlorothiazide", "furosemide", "warfarin",
        "clopidogrel", "atorvastatin", "simvastatin", "rosuvastatin",

        # Diabetes
        "metformin", "glipizide", "glyburide", "pioglitazone", "sitagliptin",
        "insulin", "liraglutide", "empagliflozin", "dapagliflozin",

        # Respiratory
        "albuterol", "fluticasone", "budesonide", "montelukast", "ipratropium",
        "tiotropium", "salmeterol", "formoterol", "beclomethasone",

        # GI medications
        "omeprazole", "esomeprazole", "pantoprazole", "ranitidine", "famotidine",
        "ondansetron", "promethazine", "metoclopramide", "docusate", "polyethylene glycol",

        # Topical / Dermatological
        "fluocinonide", "hydrocortisone", "triamcinolone", "clobetasol", "mupirocin",
        "ketoconazole", "clotrimazole", "terbinafine", "tretinoin", "adapalene",

        # Eye medications
        "avenova", "latanoprost", "timolol", "brimonidine", "dorzolamide",
        "prednisolone", "ofloxacin", "erythromycin", "tobramycin",

        # Dementia-specific medications
        "donepezil", "memantine", "rivastigmine", "galantamine",

        # Other common medications
        "levothyroxine", "prednisone", "methylprednisolone", "cyclobenzaprine",
        "hydroxyzine", "cetirizine", "loratadine", "diphenhydramine",
        "tamsulosin", "oxybutynin", "tolterodine", "finasteride", "dutasteride",
        "melatonin", "seroquel",

        # Common brand names
        "tylenol", "advil", "motrin", "aleve", "prozac", "zoloft", "lexapro",
        "cymbalta", "wellbutrin", "xanax", "ativan", "valium", "ambien", "lunesta",
        "lipitor", "crestor", "plavix", "coumadin", "glucophage", "januvia",
        "lantus", "humalog", "novolog", "ventolin", "proair", "symbicort",
        "advair", "prilosec", "nexium", "prevacid", "zantac", "pepcid",
        "synthroid", "levoxyl", "deltasone", "medrol", "flexeril", "robaxin",
        "zyrtec", "claritin", "allegra", "benadryl", "flomax", "proscar",
        "casodex", "tamiflu", "macrobid",
        "aricept", "namenda", "exelon", "razadyne",

        # Non-medication terms frequently mis-detected as names
        "mychart", "foley", "catheter", "ct", "scan", "doppler", "test",
        "dementia", "diagnostics", "lung", "pet", "medicare", "md", "medicaid",
        "mom", "johns hopkins", "greenspring", "gss", "nurse", "practitioner",
        "cna", "lumosity", "rite aid", "memory clinic", "clinic", "asap",
        "request", "psychiatrist", "psychologist", "physician", "provider",
        "nurse", "doctor", "specialist", "therapist", "counselor", "surgeon",
        "appointment", "consultation", "referral", "discharge", "admission",
    ])


# ---------------------------------------------------------------------------
# Synthetic name pools
# ---------------------------------------------------------------------------

SYNTH_MALE = [
    "Alpha", "Beta", "Charlie", "Delta", "Echo", "Felix", "Gabriel",
    "Hugo", "Ivan", "Juliet", "Kilo", "Lima", "Mike", "Nova", "Oscar",
]
SYNTH_FEMALE = [
    "Aria", "Belle", "Cora", "Diana", "Eva", "Fiona", "Grace",
    "Hope", "Iris", "Joy", "Kate", "Luna", "Maya", "Nora", "Opal",
]
SYNTH_NEUTRAL = [
    "River", "Sky", "Sage", "Phoenix", "Quinn", "Rowan", "Taylor",
    "Casey", "Jordan", "Avery", "Blake", "Cameron", "Drew", "Emery",
]
SYNTH_LAST = [
    "Prime", "Nova", "Vertex", "Matrix", "Cipher",
    "Delta", "Omega", "Zenith", "Apex", "Nexus",
]

KNOWN_MALE = {
    "john", "james", "robert", "michael", "william", "david", "richard",
    "joseph", "thomas", "mark", "donald", "george", "steven", "paul",
}
KNOWN_FEMALE = {
    "mary", "patricia", "linda", "barbara", "elizabeth", "jennifer",
    "maria", "susan", "margaret", "sarah", "karen", "nancy", "lisa",
}

MEDICAL_TITLES = ["Dr.", "MD", "RN", "NP", "PA", "PhD", "DO", "Dr "]


# ---------------------------------------------------------------------------
# Entity Replacer
# ---------------------------------------------------------------------------

class EntityReplacer:
    """Handles consistent replacement of entities."""

    def __init__(self, config: DeIdentificationConfig):
        self.config = config
        self.replacement_map: Dict[str, Dict[str, str]] = {
            et.value: {} for et in EntityType
        }

    @staticmethod
    def _detect_gender(name: str) -> str:
        first = name.lower().split()[0] if name else ""
        if first in KNOWN_MALE:
            return "male"
        if first in KNOWN_FEMALE:
            return "female"
        return "unknown"

    # --- Name generation (4 methods) ----------------------------------------

    def _generate_person_name(self, original: str) -> str:
        """Generate a safe replacement person name that is clearly synthetic."""
        title = ""
        for med_title in MEDICAL_TITLES:
            if med_title in original:
                title = med_title + " "
                break

        original_clean = original.replace(title.strip(), "").strip()
        original_parts = original_clean.split() if original_clean else []
        original_first = original_parts[0].lower() if original_parts else ""
        has_last_name = len(original_parts) > 1

        def _coded(with_last: bool = True) -> str:
            prefix = "PERSON" if not title else "PROVIDER"
            return f"{prefix}_{random.randint(100, 999)}"

        def _pattern(with_last: bool = True) -> str:
            C, V = "BCDFGHJKLMNPQRSTVWXYZ", "AEIOU"
            first_n = (random.choice(C) + random.choice(V) +
                       random.choice(C) + random.choice(V)).title()
            if not with_last:
                return first_n
            last_n = (random.choice(C) + random.choice(V) +
                      random.choice(C) + random.choice(V) +
                      random.choice(C)).title()
            return f"{first_n} {last_n}"

        method = self.config.name_generation_method
        max_attempts = 20

        for _ in range(max_attempts):
            if method == "coded":
                fake_name = _coded(has_last_name)
            elif method == "pattern":
                fake_name = _pattern(has_last_name)
            elif method == "synthetic":
                if self.config.preserve_gender:
                    gender = self._detect_gender(original)
                    pool = (SYNTH_MALE if gender == "male"
                            else SYNTH_FEMALE if gender == "female"
                            else SYNTH_NEUTRAL)
                else:
                    pool = SYNTH_MALE + SYNTH_FEMALE + SYNTH_NEUTRAL
                available = [n for n in pool if n.lower() != original_first]
                if available:
                    chosen = random.choice(available)
                    fake_name = (f"{chosen} {random.choice(SYNTH_LAST)}"
                                 if has_last_name else chosen)
                else:
                    fake_name = _pattern(has_last_name)
            else:  # "faker"
                if _faker is None:
                    fake_name = _pattern(has_last_name)
                else:
                    if self.config.preserve_gender:
                        gender = self._detect_gender(original)
                        full = (_faker.name_male() if gender == "male"
                                else _faker.name_female() if gender == "female"
                                else _faker.name())
                    else:
                        full = _faker.name()
                    fake_name = full.split()[0] if not has_last_name else full

            fake_first = fake_name.lower().split()[0] if fake_name else ""
            if fake_first != original_first and fake_name.lower() != original_clean.lower():
                return title + fake_name

        return title + _coded(has_last_name)

    # --- Other generators ---------------------------------------------------

    def _generate_email(self, original: str) -> str:
        if _faker:
            domain = original.split("@")[1] if "@" in original else "example.com"
            email = _faker.email()
            if any(k in domain for k in ["hospital", "clinic", "medical", "health"]):
                email = email.split("@")[0] + "@medical-center.org"
            return email
        return "".join(random.choices(string.ascii_lowercase, k=6)) + "@example.org"

    def _generate_phone(self, original: str) -> str:
        d = lambda n: "".join(str(random.randint(0, 9)) for _ in range(n))
        if _faker:
            if re.match(r'\(\d{3}\) \d{3}-\d{4}', original):
                return _faker.phone_number()
            elif re.match(r'\d{3}-\d{3}-\d{4}', original):
                return f"{d(3)}-{d(3)}-{d(4)}"
            elif re.match(r'\d{10}', original):
                return d(10)
            return _faker.phone_number()
        if "(" in original:
            return f"({d(3)}) {d(3)}-{d(4)}"
        return f"{d(3)}-{d(3)}-{d(4)}"

    def _generate_ssn(self, _original: str) -> str:
        if _faker:
            return _faker.ssn()
        return f"{random.randint(100,999):03d}-{random.randint(10,99):02d}-{random.randint(1000,9999):04d}"

    def _generate_mrn(self, original: str) -> str:
        clean = original.replace("MRN", "").replace("mrn", "").strip(": ")
        if clean.isdigit():
            return str(random.randint(10 ** (len(clean) - 1), 10 ** len(clean) - 1))
        return "".join(random.choices(string.ascii_uppercase + string.digits, k=len(clean)))

    def _generate_address(self, _original: str) -> str:
        if _faker:
            return _faker.address().replace("\n", ", ")
        num = random.randint(100, 9999)
        streets = ["Main St", "Oak Ave", "Elm Blvd", "Cedar Ln", "Park Dr"]
        cities = ["Anytown", "Fakeville", "Sampleburg", "Testcity"]
        return f"{num} {random.choice(streets)}, {random.choice(cities)}"

    def _shift_date(self, original: str) -> str:
        formats = [
            "%m/%d/%Y", "%Y-%m-%d", "%B %d, %Y", "%b %d, %Y",
            "%m-%d-%Y", "%d/%m/%Y", "%Y/%m/%d",
        ]
        for fmt in formats:
            try:
                d = datetime.strptime(original, fmt)
                shifted = d + timedelta(days=self.config.shift_dates_by_days)
                return shifted.strftime(fmt)
            except ValueError:
                continue
        return original

    def _generate_organization(self, original: str) -> str:
        is_healthcare = any(k in original.lower() for k in self.config.healthcare_org_keywords)
        if is_healthcare and self.config.replace_healthcare_orgs:
            templates = [
                "Regional Medical Center", "Community Hospital",
                "Healthcare System", "Medical Group",
                "Health Center", "Clinical Associates",
            ]
            city = _faker.city() if _faker else "Sampleville"
            return f"{city} {random.choice(templates)}"
        return _faker.company() if _faker else "Acme Corp"

    # --- Main replacement dispatcher ----------------------------------------

    def get_replacement(self, entity: IdentifiedEntity) -> str:
        key = entity.original_text.lower().strip()
        etype = entity.entity_type.value

        if self.config.consistent_replacement and key in self.replacement_map[etype]:
            return self.replacement_map[etype][key]

        generators = {
            EntityType.PERSON_NAME: lambda: self._generate_person_name(entity.original_text),
            EntityType.EMAIL: lambda: self._generate_email(entity.original_text),
            EntityType.PHONE: lambda: self._generate_phone(entity.original_text),
            EntityType.SSN: lambda: self._generate_ssn(entity.original_text),
            EntityType.MRN: lambda: self._generate_mrn(entity.original_text),
            EntityType.ADDRESS: lambda: self._generate_address(entity.original_text),
            EntityType.DATE: lambda: self._shift_date(entity.original_text),
            EntityType.ORGANIZATION: lambda: self._generate_organization(entity.original_text),
            EntityType.CREDIT_CARD: lambda: (_faker.credit_card_number() if _faker
                                             else "XXXX-XXXX-XXXX-" + str(random.randint(1000, 9999))),
            EntityType.URL: lambda: (_faker.url() if _faker else "https://example.org"),
        }

        replacement = generators.get(entity.entity_type,
                                     lambda: f"[REDACTED-{entity.entity_type.value}]")()
        if self.config.consistent_replacement:
            self.replacement_map[etype][key] = replacement
        return replacement


# ---------------------------------------------------------------------------
# Entity Detector
# ---------------------------------------------------------------------------

class EntityDetector:
    """Detects various types of entities in text."""

    def __init__(self, config: DeIdentificationConfig):
        self.config = config
        self._med_set: Set[str] = (
            {m.lower() for m in config.medication_exclusion_list}
            if config.exclude_medications else set()
        )

        self._patterns = {
            EntityType.EMAIL: re.compile(
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'),
            EntityType.PHONE: re.compile(
                r'(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}'),
            EntityType.SSN: re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
            EntityType.CREDIT_CARD: re.compile(
                r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'),
            EntityType.URL: re.compile(r'https?://[^\s]+'),
            EntityType.IP_ADDRESS: re.compile(
                r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'),
            EntityType.MRN: re.compile(r'\b(?:MRN|mrn)[:\s]?\d{6,10}\b'),
            EntityType.FAX: re.compile(
                r'\b(?:fax|Fax|FAX)[:\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b'),
            EntityType.PERSON_NAME: re.compile(r'\bDr\.\s+[A-Z][a-z]+\b'),
        }
        self._address_pat = re.compile(
            r'\b\d{1,5}\s+[\w\s]+(?:street|st|avenue|ave|road|rd|lane|ln|'
            r'drive|dr|court|ct|boulevard|blvd)\b',
            re.IGNORECASE,
        )
        self._name_pat = re.compile(
            r'\b(?:Dr\.?\s+)?[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b')

        self._scispacy_cache: Dict[str, bool] = {}

    # --- Medication detection (dual-tier) -----------------------------------

    def _is_medication_scispacy(self, text: str) -> bool:
        if not self.config.use_scispacy_medication_detection:
            return False
        sci = _load_scispacy()
        if sci is None:
            return False
        text_lower = text.lower().strip()
        if text_lower in self._scispacy_cache:
            return self._scispacy_cache[text_lower]
        doc = sci(text)
        is_med = any(ent.label_ == "CHEMICAL" for ent in doc.ents)
        self._scispacy_cache[text_lower] = is_med
        return is_med

    def _is_medication(self, text: str) -> bool:
        if not self.config.exclude_medications:
            return False
        text_lower = text.lower().strip()
        if text_lower in self._med_set:
            return True
        for word in text_lower.split():
            if word.strip('.,;:!?"\'') in self._med_set:
                return True
        if self._is_medication_scispacy(text):
            return True
        return False

    def _should_detect(self, entity_type: EntityType) -> bool:
        flags = {
            EntityType.EMAIL: self.config.detect_emails,
            EntityType.PHONE: self.config.detect_phones,
            EntityType.SSN: self.config.detect_ssn,
            EntityType.MRN: self.config.detect_mrn,
            EntityType.CREDIT_CARD: self.config.detect_credit_cards,
            EntityType.URL: self.config.detect_urls,
        }
        return flags.get(entity_type, True)

    def detect_entities(self, text: str) -> List[IdentifiedEntity]:
        entities: List[IdentifiedEntity] = []

        # spaCy-based detection
        if self.config.use_spacy:
            nlp = _load_spacy()
            if nlp:
                doc = nlp(text)
                for ent in doc.ents:
                    if self._is_medication(ent.text):
                        continue
                    if ent.label_ == "PERSON" and self.config.detect_names:
                        entities.append(IdentifiedEntity(
                            EntityType.PERSON_NAME, ent.text,
                            ent.start_char, ent.end_char))
                    elif ent.label_ == "ORG" and self.config.detect_organizations:
                        entities.append(IdentifiedEntity(
                            EntityType.ORGANIZATION, ent.text,
                            ent.start_char, ent.end_char))

        # Regex-based name detection (when spaCy is off)
        if self.config.detect_names and not self.config.use_spacy:
            skip_phrases = {
                "patient medical", "medical advice", "advice request",
                "enterprise id",
            }
            for m in self._name_pat.finditer(text):
                name = m.group()
                if self._is_medication(name):
                    continue
                if name.lower() in skip_phrases:
                    continue
                entities.append(IdentifiedEntity(
                    EntityType.PERSON_NAME, name, m.start(), m.end()))

        # Structured regex patterns
        for etype, pattern in self._patterns.items():
            if self._should_detect(etype):
                for m in pattern.finditer(text):
                    entities.append(IdentifiedEntity(
                        etype, m.group(), m.start(), m.end()))

        # Addresses
        if self.config.detect_addresses:
            for m in self._address_pat.finditer(text):
                entities.append(IdentifiedEntity(
                    EntityType.ADDRESS, m.group(), m.start(), m.end()))

        # De-duplicate overlapping entities (keep longer)
        entities.sort(key=lambda e: (e.start_pos, -e.end_pos))
        filtered: List[IdentifiedEntity] = []
        last_end = -1
        for e in entities:
            if e.start_pos >= last_end:
                filtered.append(e)
                last_end = e.end_pos
        return filtered


# ---------------------------------------------------------------------------
# Main Engine
# ---------------------------------------------------------------------------

class DeIdentificationEngine:
    """Main de-identification engine."""

    def __init__(self, config: Optional[DeIdentificationConfig] = None):
        self.config = config or DeIdentificationConfig()
        self.detector = EntityDetector(self.config)
        self.replacer = EntityReplacer(self.config)
        self.stats: Dict[str, Any] = {
            "texts_processed": 0,
            "entities_found": 0,
            "entities_replaced": 0,
            "entity_types": {et.value: 0 for et in EntityType},
            "medications_preserved": 0,
            "medications_preserved_scispacy": 0,
            "medications_preserved_list": 0,
        }

    def deidentify_text(self, text: str) -> Tuple[str, List[IdentifiedEntity]]:
        """De-identify a single text string."""
        if not text:
            return text, []
        entities = self.detector.detect_entities(text)
        entities_rev = sorted(entities, key=lambda e: e.start_pos, reverse=True)
        result = text
        for entity in entities_rev:
            replacement = self.replacer.get_replacement(entity)
            result = result[:entity.start_pos] + replacement + result[entity.end_pos:]
            self.stats["entities_replaced"] += 1
            self.stats["entity_types"][entity.entity_type.value] += 1
        self.stats["texts_processed"] += 1
        self.stats["entities_found"] += len(entities)
        return result, entities

    def deidentify_dict(self, data: Dict[str, Any],
                        fields_to_process: List[str]) -> Dict[str, Any]:
        """De-identify specific fields in a dictionary."""
        out = data.copy()
        for fld in fields_to_process:
            if fld in data and isinstance(data[fld], str):
                out[fld], _ = self.deidentify_text(data[fld])
        return out

    def add_medications_to_exclude(self, medications: List[str]):
        """Add additional medications to the exclusion list at runtime."""
        self.config.medication_exclusion_list.extend(medications)
        self.detector._med_set = {
            m.lower() for m in self.config.medication_exclusion_list
        }

    def save_mapping(self, filepath: str):
        """Save the replacement mapping to a JSON file."""
        data = {
            "config": {
                "shift_dates_by_days": self.config.shift_dates_by_days,
                "preserve_gender": self.config.preserve_gender,
                "consistent_replacement": self.config.consistent_replacement,
                "exclude_medications": self.config.exclude_medications,
                "use_scispacy_medication_detection": self.config.use_scispacy_medication_detection,
                "medication_count": len(self.config.medication_exclusion_list),
            },
            "mappings": self.replacer.replacement_map,
            "stats": self.stats,
            "created_at": datetime.now().isoformat(),
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def load_mapping(self, filepath: str):
        """Load a replacement mapping from a JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        self.replacer.replacement_map = data["mappings"]
        if "config" in data:
            for key, value in data["config"].items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)

    def get_stats(self) -> Dict[str, Any]:
        return self.stats.copy()

    def reset_stats(self):
        self.stats = {
            "texts_processed": 0,
            "entities_found": 0,
            "entities_replaced": 0,
            "entity_types": {et.value: 0 for et in EntityType},
            "medications_preserved": 0,
            "medications_preserved_scispacy": 0,
            "medications_preserved_list": 0,
        }
