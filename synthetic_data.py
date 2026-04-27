"""
Synthetic Data Generator for SmartRep AI Sandbox
Generates realistic patient portal messages focused on dementia care scenarios.
Each record simulates a real EHR message with Enterprise_ID, sender info, etc.
"""

import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

random.seed(42)

# ---------------------------------------------------------------------------
# Building blocks for synthetic messages
# ---------------------------------------------------------------------------

PATIENT_NAMES = [
    ("John", "Smith", "M"), ("Mary", "Chen", "F"), ("Robert", "Johnson", "M"),
    ("Patricia", "Adams", "F"), ("William", "Thompson", "M"), ("Linda", "Garcia", "F"),
    ("James", "Robinson", "M"), ("Sarah", "Martinez", "F"), ("David", "Wilson", "M"),
    ("Elizabeth", "Taylor", "F"), ("Richard", "Brown", "M"), ("Nancy", "Davis", "F"),
]

CAREGIVER_NAMES = [
    ("Michael", "Smith", "M"), ("Jane", "Martinez", "F"), ("Karen", "Thompson", "F"),
    ("Daniel", "Chen", "M"), ("Susan", "Adams", "F"), ("Thomas", "Garcia", "M"),
    ("Angela", "Robinson", "F"), ("Steven", "Wilson", "M"), ("Laura", "Taylor", "F"),
    ("Brian", "Brown", "M"), ("Melissa", "Davis", "F"), ("Kevin", "Johnson", "M"),
]

PROVIDER_NAMES = [
    "Dr. Smith", "Dr. Williams", "Dr. Johnson", "Dr. Rodriguez",
    "Dr. Thompson", "Dr. Anderson", "Dr. Davis", "Dr. Martinez",
]

ADDRESSES = [
    "123 Oak Street, Springfield", "456 Pine Avenue, Riverside",
    "789 Maple Drive, Brookfield", "321 Cedar Lane, Westfield",
    "654 Elm Court, Lakewood", "987 Birch Road, Fairview",
]

PHONES = [
    "(312) 123-4567", "(410) 987-6543", "(312) 456-7890",
    "(410) 234-5678", "(412) 567-8901", "(410) 678-9012",
    "(412) 789-0123", "(312) 345-6789", "(410) 456-7890",
]

EMAILS = [
    "john.smith@email.com", "jane.martinez@email.com", "michael.wilson@gmail.com",
    "karen.t@outlook.com", "susan.adams@yahoo.com", "angela.r@email.com",
    "laura.t@gmail.com", "brian.b@hotmail.com", "melissa.d@email.com",
]

MEDICATIONS = [
    "Aricept", "Namenda", "Exelon", "Razadyne", "Tramadol",
    "sertraline", "fluocinonide", "Avenova", "donepezil", "memantine",
    "Lexapro", "melatonin", "trazodone", "lorazepam", "Seroquel",
]

# Template messages — {placeholders} filled at generation time
CLINICAL_PATIENT_MESSAGES = [
    "I've been having more trouble remembering things lately and I forgot to take my {med} yesterday. "
    "I'm worried about my memory getting worse. {provider} said to contact you if things changed. "
    "Please call me at {phone}.",

    "My memory has been getting worse this past week. I can't remember where I put my keys and "
    "I got lost driving home from the grocery store. {provider} prescribed {med} last month. "
    "Should I increase the dose? My address is {address}. Email me at {email}.",

    "I'm experiencing more confusion than usual and I had a fall yesterday. I didn't hit my head "
    "but my hip hurts. Currently taking {med} and {med2}. {provider} told me to report any falls. "
    "You can reach me at {phone}.",

    "The new medication {med} that {provider} prescribed is making me very dizzy and nauseous. "
    "I've been taking it for two weeks. Should I stop taking it? Contact me at {email} or {phone}.",

    "I've noticed I'm having trouble swallowing my food and I've lost about 5 pounds this month. "
    "{provider} mentioned this could be part of the progression. What should we do? "
    "I live at {address}. My number is {phone}.",
]

CLINICAL_CAREGIVER_MESSAGES = [
    "My {relation} {patient_name} has been very confused and agitated today. "
    "{pronoun_subj} doesn't recognize me and keeps asking for {pronoun_poss} mother. "
    "{provider} said this might happen. Is this normal progression? You can reach me at {phone}.",

    "My {relation} {patient_name} keeps forgetting to take {pronoun_poss} {med} and sometimes takes it twice. "
    "I'm worried about overdose. {provider} prescribed it last month. Should I get a pill organizer? "
    "My contact is {email} or call {phone}.",

    "My {relation} {patient_name} at {address} is having severe sundowning episodes. "
    "{pronoun_subj}'s trying to leave the house at night and becoming aggressive when we stop {pronoun_obj}. "
    "{provider} recommended we contact you. We need immediate help. Call me at {phone}.",

    "As my {relation} {patient_name}'s primary caregiver, I'm noticing {pronoun_subj} is becoming "
    "more withdrawn and not eating well. {provider} started {pronoun_obj} on {med} three weeks ago. "
    "Is this a side effect? Contact me at {email}.",

    "I need to report that my {relation} {patient_name} wandered out of the house last night at 2 AM. "
    "We found {pronoun_obj} three blocks away at {address}. {provider} said to let you know immediately. "
    "Please call {phone}.",

    "My {relation} {patient_name} has been on {med} for 6 months and I think it's helping "
    "with the agitation, but {pronoun_subj} is now having vivid nightmares. {provider} said we could "
    "try adjusting the dose. What do you recommend? Email me at {email}.",
]

NONCLINICAL_MESSAGES = [
    "I need to reschedule my {relation}'s neurology appointment with {provider} next week. "
    "{pronoun_subj} had a fall and we're dealing with that right now. Please call me at {phone} "
    "or email {email}.",

    "Can you provide information about local dementia support groups and respite care services "
    "near our address at {address}? I'm caring for my {relation} {patient_name}. "
    "{provider} suggested we look into these services. Contact me at {phone}.",

    "I'm trying to access my {relation}'s MyChart portal but the password isn't working. "
    "The account is under {patient_name}, date of birth {dob}. "
    "Can you help me reset it? Call me at {phone}.",

    "Could you fax {patient_name}'s medical records to the new specialist? "
    "The fax number is (410) 555-0199. {provider} made the referral last week. "
    "If you need anything else, email me at {email}.",

    "What are the visiting hours for the memory care unit? My {relation} {patient_name} "
    "was recently admitted and I'd like to bring some personal items. "
    "I can be reached at {phone}.",

    "I need to update the insurance information for {patient_name}. "
    "The new policy number is BX-7823491. {provider}'s office asked me to contact you. "
    "Please call {phone} or email {email}.",
]

PROVIDER_RESPONSES = [
    "Thank you for reaching out. I understand your concern about {patient_name}'s condition. "
    "Based on what you've described, I'd recommend scheduling a follow-up appointment so we can "
    "assess the situation in person. In the meantime, continue the current medication regimen. "
    "If symptoms worsen significantly, please call our urgent line.",

    "I appreciate you keeping us informed. The symptoms you describe are consistent with "
    "disease progression, but we want to make sure nothing else is going on. "
    "Please bring {patient_name} in this week for an evaluation. "
    "I'll have the nurse call you to schedule.",

    "Thank you for your message. I've reviewed {patient_name}'s chart and adjusted the "
    "medication dosage. Please pick up the new prescription from the pharmacy. "
    "Monitor for any changes and let us know how things go over the next two weeks.",
]


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def _pick(lst):
    return random.choice(lst)


def _fill_message(template: str, ctx: dict) -> str:
    """Best-effort placeholder replacement."""
    meds = random.sample(MEDICATIONS, min(2, len(MEDICATIONS)))
    defaults = {
        "med": meds[0],
        "med2": meds[1] if len(meds) > 1 else meds[0],
        "provider": _pick(PROVIDER_NAMES),
        "phone": _pick(PHONES),
        "email": _pick(EMAILS),
        "address": _pick(ADDRESSES),
        "dob": f"{random.randint(1,12):02d}/{random.randint(1,28):02d}/{random.randint(1935,1955)}",
    }
    defaults.update(ctx)
    # Replace known placeholders; leave unknown ones as-is
    result = template
    for key, val in defaults.items():
        result = result.replace("{" + key + "}", str(val))
    return result


def generate_synthetic_dataset(n_enterprises: int = 12, seed: int = 42) -> list[dict]:
    """Generate a full synthetic dataset of patient portal messages.

    Returns a list of dicts suitable for pd.DataFrame().
    """
    random.seed(seed)
    records = []
    msg_id_counter = 1

    for ent_idx in range(n_enterprises):
        enterprise_id = f"ENT{1000 + ent_idx}"
        patient = PATIENT_NAMES[ent_idx % len(PATIENT_NAMES)]
        caregiver = CAREGIVER_NAMES[ent_idx % len(CAREGIVER_NAMES)]
        patient_name = f"{patient[0]} {patient[1]}"
        caregiver_name = f"{caregiver[0]} {caregiver[1]}"
        gender = patient[2]
        pronoun_subj = "He" if gender == "M" else "She"
        pronoun_obj = "him" if gender == "M" else "her"
        pronoun_poss = "his" if gender == "M" else "her"
        relation = _pick(["father", "husband", "grandfather"]) if gender == "M" else _pick(["mother", "wife", "grandmother"])

        ctx = {
            "patient_name": patient_name,
            "relation": relation,
            "pronoun_subj": pronoun_subj,
            "pronoun_obj": pronoun_obj,
            "pronoun_poss": pronoun_poss,
        }

        base_date = datetime(2024, 1, 15) + timedelta(days=ent_idx * 7)
        n_sessions = random.randint(2, 4)

        for sess in range(1, n_sessions + 1):
            session_date = base_date + timedelta(days=sess * random.randint(3, 14))
            msg_order = 0

            # Decide type of first message
            if random.random() < 0.25:
                # Non-clinical message
                template = _pick(NONCLINICAL_MESSAGES)
                sender = "PATIENT" if random.random() < 0.4 else "CAREGIVER"
                msg_type = "Patient Medical Advice Request"
                is_clinical = False
            elif random.random() < 0.5:
                # Caregiver clinical message
                template = _pick(CLINICAL_CAREGIVER_MESSAGES)
                sender = "CAREGIVER"
                msg_type = "Patient Medical Advice Request"
                is_clinical = True
            else:
                # Patient clinical message
                template = _pick(CLINICAL_PATIENT_MESSAGES)
                sender = "PATIENT"
                msg_type = "Patient Medical Advice Request"
                is_clinical = True

            msg_order += 1
            email_text = _fill_message(template, ctx)
            records.append({
                "Message_ID": f"MSG{msg_id_counter:05d}",
                "Enterprise_ID": enterprise_id,
                "Patient_Name": patient_name,
                "Sender": sender,
                "Sender_Name": patient_name if sender == "PATIENT" else caregiver_name,
                "Message_Type": msg_type,
                "Email_Text": email_text,
                "Msg_Delivered": (session_date + timedelta(hours=random.randint(8, 17), minutes=random.randint(0, 59))).isoformat(),
                "Session_Number": sess,
                "Message_Order": msg_order,
                "Is_Clinical_Ground_Truth": is_clinical,
                "Author_Ground_Truth": "patient" if sender == "PATIENT" else "caretaker",
            })
            msg_id_counter += 1

            # Provider response
            msg_order += 1
            resp_template = _pick(PROVIDER_RESPONSES)
            resp_text = _fill_message(resp_template, ctx)
            provider = _pick(PROVIDER_NAMES)
            records.append({
                "Message_ID": f"MSG{msg_id_counter:05d}",
                "Enterprise_ID": enterprise_id,
                "Patient_Name": patient_name,
                "Sender": "PROVIDER",
                "Sender_Name": provider,
                "Message_Type": "Provider Response",
                "Email_Text": resp_text,
                "Msg_Delivered": (session_date + timedelta(hours=random.randint(1, 4), minutes=random.randint(0, 59))).isoformat(),
                "Session_Number": sess,
                "Message_Order": msg_order,
                "Is_Clinical_Ground_Truth": None,
                "Author_Ground_Truth": "provider",
            })
            msg_id_counter += 1

            # Possible follow-up from patient/caregiver
            if random.random() < 0.5:
                msg_order += 1
                followup_templates = [
                    "Thank you for the response. I'll bring {patient_name} in this week. "
                    "Should I keep giving {pronoun_obj} the {med} in the meantime? Call me at {phone}.",
                    "I appreciate the quick reply. One more question — {patient_name} has been "
                    "sleeping a lot more than usual. Is that related to the {med}? "
                    "My email is {email}.",
                    "Thanks, {provider}. I'll schedule the appointment. In the meantime, "
                    "{patient_name}'s caregiver can be reached at {phone} if anything urgent comes up.",
                ]
                followup = _fill_message(_pick(followup_templates), ctx)
                sender_followup = "CAREGIVER" if sender == "CAREGIVER" else _pick(["PATIENT", "CAREGIVER"])
                records.append({
                    "Message_ID": f"MSG{msg_id_counter:05d}",
                    "Enterprise_ID": enterprise_id,
                    "Patient_Name": patient_name,
                    "Sender": sender_followup,
                    "Sender_Name": patient_name if sender_followup == "PATIENT" else caregiver_name,
                    "Message_Type": "Patient Medical Advice Request",
                    "Email_Text": followup,
                    "Msg_Delivered": (session_date + timedelta(hours=random.randint(5, 8), minutes=random.randint(0, 59))).isoformat(),
                    "Session_Number": sess,
                    "Message_Order": msg_order,
                    "Is_Clinical_Ground_Truth": True,
                    "Author_Ground_Truth": "patient" if sender_followup == "PATIENT" else "caretaker",
                })
                msg_id_counter += 1

    return records


def save_synthetic_csv(path: str = "data/synthetic_messages.csv", n_enterprises: int = 12):
    """Generate and save synthetic data to CSV."""
    records = generate_synthetic_dataset(n_enterprises)
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    with open(path_obj, "w", newline="", encoding="utf-8") as f:
        if records:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)

    return len(records), path


if __name__ == "__main__":
    count, fpath = save_synthetic_csv()
    print(f"Generated {count} synthetic messages → {fpath}")
