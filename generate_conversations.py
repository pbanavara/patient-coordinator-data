"""
Generate 1000 post-transplant patient-coordinator conversations for Gemma 4 fine-tuning.
Covers 4 scenarios: elevated creatinine, follow-up scheduling, emotional distress, lifestyle changes.
"""

import json
import random
from pathlib import Path

# ─── Data pools ───────────────────────────────────────────────────────────────

PATIENT_NAMES = [
    "James Carter", "Maria Gonzalez", "Robert Kim", "Linda Patel", "David Johnson",
    "Sarah Williams", "Michael Brown", "Emily Davis", "Thomas Wilson", "Jessica Martinez",
    "Christopher Anderson", "Amanda Taylor", "Daniel Thomas", "Ashley Jackson", "Matthew White",
    "Stephanie Harris", "Andrew Martin", "Nicole Thompson", "Joshua Garcia", "Megan Robinson",
    "Kevin Lewis", "Lauren Walker", "Brian Hall", "Rachel Allen", "Tyler Young",
    "Christina Hernandez", "Nathan King", "Samantha Wright", "Aaron Scott", "Brittany Green",
    "Patrick Adams", "Heather Baker", "Sean Nelson", "Amber Mitchell", "Justin Carter",
    "Tiffany Perez", "Brandon Roberts", "Courtney Turner", "Gregory Phillips", "Crystal Campbell",
    "Raymond Parker", "Vanessa Evans", "Frank Edwards", "Monica Collins", "Wayne Stewart",
    "Kimberly Sanchez", "Dennis Morris", "Theresa Rogers", "Jerry Reed", "Denise Cook",
]

COORDINATOR_NAMES = [
    "Nurse Jennifer Walsh", "Coordinator Mark Stevens", "Nurse Practitioner Susan Chen",
    "Coordinator Lisa Nguyen", "Nurse David Park", "Coordinator Amanda Foster",
    "Nurse Michael Torres", "Coordinator Rebecca Hughes", "Nurse Patricia Young",
    "Coordinator Steven Ramirez",
]

DOCTOR_NAMES = [
    "Dr. Richardson", "Dr. Patel", "Dr. Kim", "Dr. Okonkwo", "Dr. Lieberman",
    "Dr. Torres", "Dr. Hassan", "Dr. Nakamura", "Dr. Williams", "Dr. Fernandez",
]

TRANSPLANT_TYPES = [
    "kidney", "liver", "heart", "lung", "kidney-pancreas",
]

IMMUNOSUPPRESSANTS = [
    ("tacrolimus", "mycophenolate", "prednisone"),
    ("cyclosporine", "azathioprine", "prednisone"),
    ("tacrolimus", "sirolimus", "prednisone"),
    ("tacrolimus", "mycophenolate mofetil", "methylprednisolone"),
    ("everolimus", "mycophenolate", "prednisone"),
]

SIDE_EFFECTS = [
    "nausea and stomach upset",
    "tremors in my hands",
    "headaches and dizziness",
    "difficulty sleeping",
    "increased sensitivity to sunlight",
    "swollen gums",
    "hair thinning",
    "mood swings and irritability",
    "muscle cramps",
    "fatigue throughout the day",
]

MONTHS_POST_TX = [1, 2, 3, 4, 6, 8, 10, 12, 18, 24, 36]

DIETARY_CHALLENGES = [
    "I find it hard to avoid high-potassium foods like bananas and oranges",
    "cutting out salt has been really difficult — I miss my regular cooking",
    "I struggle to avoid grapefruit since I used to love it",
    "limiting fluids is harder than I expected, especially during hot weather",
    "I keep forgetting to avoid raw fish and undercooked meats",
    "the low-sodium diet has made eating out almost impossible",
]

EMOTIONAL_STRESSORS = [
    "I feel like a burden on my family and I don't know how to cope",
    "I'm constantly anxious about rejection and I can't sleep",
    "I feel isolated because my friends don't understand what I'm going through",
    "I'm struggling with survivor's guilt knowing someone had to die for me to live",
    "returning to work feels overwhelming and I'm not sure I can handle it",
    "my relationship with my spouse has become strained since the transplant",
    "I feel depressed because my recovery is slower than I expected",
    "I'm terrified every time I get a cold, thinking it means rejection",
]

APPOINTMENT_CONCERNS = [
    "whether I need to bring all my medications with me",
    "if I should fast before the blood draw",
    "how long the appointment will take",
    "whether my caregiver can stay with me during the visit",
    "if there are parking options near the clinic",
    "what to do if I miss a dose the morning of the appointment",
    "whether I'll get my results the same day",
]

# ─── Ambiguous / longitudinal scenario data pools ────────────────────────────

BORDERLINE_ELEVATION_CAUSES = [
    "mild dehydration over the past few days",
    "a significant increase in dietary protein intake",
    "notably more strenuous physical activity than usual",
    "a recent upper respiratory illness that has now resolved",
    "elevated stress and disrupted sleep over the past week",
    "a change in timing or absorption of tacrolimus due to dietary shifts",
]

WATCHFUL_WAITING_CRITERIA = [
    "fever above 38°C, severe pain, or a significant decrease in urination",
    "fever, chills, significant swelling in your legs or face, or reduced urine output",
    "fever above 38.5°C, sudden weight gain over two pounds, or severe fatigue",
    "any fever, marked decrease in urine output, or new pain near the transplant site",
]

INFECTION_PRESENTATIONS = [
    {
        "symptom": "a low-grade fever of 37.9°C for the past two days with chills and loss of appetite",
        "likely_source": "a respiratory or urinary tract infection",
    },
    {
        "symptom": "intermittent fevers reaching 38.2°C, body aches, and significant fatigue",
        "likely_source": "a possible viral or bacterial infection",
    },
    {
        "symptom": "a persistent cough with mild fever and night sweats for the past week",
        "likely_source": "a respiratory infection, which can be serious in immunosuppressed patients",
    },
    {
        "symptom": "burning with urination, lower back discomfort, and a temperature of 38.1°C",
        "likely_source": "a urinary tract infection, which can directly affect the transplanted organ",
    },
    {
        "symptom": "mild fever, sore throat, and swollen lymph nodes persisting for four days",
        "likely_source": "a possible viral infection — EBV or CMV are important to check in your context",
    },
]

TOXICITY_PRESENTATIONS = [
    {
        "symptoms": "persistent hand tremors and headaches",
        "toxicity_note": "tacrolimus toxicity",
        "creatinine_delta_range": (0.4, 0.9),
    },
    {
        "symptoms": "worsening headaches and elevated blood pressure readings at home",
        "toxicity_note": "calcineurin inhibitor toxicity",
        "creatinine_delta_range": (0.3, 0.8),
    },
    {
        "symptoms": "significant tremors and persistent nausea despite consistent dosing",
        "toxicity_note": "possible supratherapeutic tacrolimus levels",
        "creatinine_delta_range": (0.5, 1.0),
    },
    {
        "symptoms": "fine hand tremors, light sensitivity, and marked fatigue",
        "toxicity_note": "tacrolimus toxicity or early rejection",
        "creatinine_delta_range": (0.3, 0.7),
    },
]

DIETARY_TACROLIMUS_FACTORS = [
    "started eating grapefruit or grapefruit juice again recently",
    "significantly changed the timing of when I take my morning medications",
    "been eating a much higher protein diet than usual",
    "started a new antifungal medication my dentist recently prescribed",
    "been drinking an herbal tea supplement I found online",
    "switched to a high-fat diet that may be affecting medication absorption",
]

MONITORING_INTERVALS = ["48 hours", "72 hours", "three to five days", "one week"]

LONGITUDINAL_LIFE_EVENTS = [
    "returned to part-time work",
    "started driving again",
    "attended a family gathering for the first time since the transplant",
    "celebrated the transplant anniversary with family",
    "enrolled in a cardiac rehabilitation program",
    "resumed a hobby I had given up before the transplant",
    "reduced caregiver support as my independence improved",
    "had a minor cold that required extra monitoring",
    "had a scheduled biopsy that came back clear",
    "adjusted my immunosuppression dose after a drug level review",
]

# ─── Scenario builders ────────────────────────────────────────────────────────

def scenario_elevated_creatinine(idx: int) -> dict:
    patient = random.choice(PATIENT_NAMES)
    first = patient.split()[0]
    coordinator = random.choice(COORDINATOR_NAMES)
    doctor = random.choice(DOCTOR_NAMES)
    tx_type = random.choice(TRANSPLANT_TYPES)
    months = random.choice(MONTHS_POST_TX)
    creatinine = round(random.uniform(1.8, 4.5), 1)
    baseline = round(creatinine - random.uniform(0.4, 1.2), 1)
    meds = random.choice(IMMUNOSUPPRESSANTS)
    side_effect = random.choice(SIDE_EFFECTS)
    missed = random.choice([True, False])
    symptom = random.choice([
        "some swelling in my ankles", "decreased urine output", "fatigue",
        "mild shortness of breath", "slight back pain near the transplant site",
        "nausea after meals",
    ])

    messages = [
        {
            "role": "system",
            "content": (
                "You are an experienced transplant coordinator conducting a post-transplant "
                "follow-up call. You are empathetic, professional, and knowledgeable about "
                "immunosuppression, rejection signs, and post-transplant care protocols. "
                "Your goal is to assess the patient's current status, address concerns, "
                "and escalate appropriately."
            ),
        },
        {
            "role": "coordinator",
            "content": (
                f"Hello, may I speak with {patient}? This is {coordinator} calling from "
                f"the transplant clinic. Is now a good time to talk?"
            ),
        },
        {
            "role": "patient",
            "content": "Yes, this is {first}. Go ahead.".replace("{first}", first),
        },
        {
            "role": "coordinator",
            "content": (
                f"Hi {first}, I'm calling because we received your recent lab results and "
                f"I wanted to go over them with you. Your creatinine came back at {creatinine} mg/dL, "
                f"which is higher than your baseline of around {baseline} mg/dL. This is something "
                f"we need to look into together. First, can you walk me through your medication "
                f"routine? Are you taking your {meds[0]}, {meds[1]}, and {meds[2]} as prescribed?"
            ),
        },
        {
            "role": "patient",
            "content": (
                (
                    f"I have to be honest — I missed a couple of doses this week because "
                    f"I've been dealing with {side_effect}. It made me feel so bad that I "
                    f"skipped the evening dose twice."
                ) if missed else (
                    f"Yes, I've been taking everything on schedule. But I have been dealing "
                    f"with {side_effect}, which has been uncomfortable. Could that be affecting things?"
                )
            ),
        },
        {
            "role": "coordinator",
            "content": (
                (
                    f"Thank you for being honest with me, {first}. Missing doses of your "
                    f"immunosuppressants can sometimes lead to the body reacting to the transplanted "
                    f"{tx_type}, which may explain the elevated creatinine. It's very important that "
                    f"we keep consistent levels in your bloodstream. I hear you about the {side_effect} — "
                    f"let's address that so it doesn't interfere with your medications. "
                    f"Have you noticed any other symptoms recently?"
                ) if missed else (
                    f"Good to hear you've been consistent with your medications — that really helps us "
                    f"narrow this down. The {side_effect} is worth discussing with {doctor}. "
                    f"Have you noticed any other changes, like {symptom}?"
                )
            ),
        },
        {
            "role": "patient",
            "content": f"Actually yes, I've been noticing {symptom} for the past few days.",
        },
        {
            "role": "coordinator",
            "content": (
                f"I appreciate you sharing that, {first}. {symptom.capitalize()} combined with an "
                f"elevated creatinine at {months} months post-transplant is something {doctor} needs "
                f"to evaluate promptly. I'm going to schedule you for an urgent visit — ideally within "
                f"the next 24 to 48 hours. We may also need to repeat your labs and possibly do an "
                f"ultrasound of the transplanted {tx_type}. In the meantime, please make sure you take "
                f"all your medications on time, stay well hydrated with water, and avoid NSAIDs like "
                f"ibuprofen. If your symptoms worsen — particularly if you develop a fever above 38°C, "
                f"severe pain, or difficulty breathing — please go to the emergency room immediately."
            ),
        },
        {
            "role": "patient",
            "content": "That's worrying. Is this rejection? Am I going to lose the transplant?",
        },
        {
            "role": "coordinator",
            "content": (
                f"I completely understand why you're concerned, and it's natural to feel anxious. "
                f"An elevated creatinine doesn't automatically mean rejection — there are several "
                f"possible causes including dehydration, medication levels being off, or an infection. "
                f"We need the additional tests to know for certain. The fact that we caught this now "
                f"and you're coming in quickly gives us the best chance to address whatever is going on. "
                f"Try to stay calm — our team is here for you every step of the way. Do you have "
                f"any other questions for me right now, {first}?"
            ),
        },
        {
            "role": "patient",
            "content": (
                "No, I think I understand. I'll make sure to take my medications and come in tomorrow. "
                "Thank you for calling."
            ),
        },
        {
            "role": "coordinator",
            "content": (
                f"You're very welcome, {first}. Our scheduling team will call you within the hour to "
                f"confirm your appointment time. Please don't hesitate to call the clinic line if "
                f"anything changes before then. Take care and we'll see you very soon."
            ),
        },
    ]

    return {
        "id": f"conv_{idx:04d}",
        "scenario": "elevated_creatinine",
        "transplant_type": tx_type,
        "months_post_transplant": months,
        "messages": messages,
    }


def scenario_followup_appointment(idx: int) -> dict:
    patient = random.choice(PATIENT_NAMES)
    first = patient.split()[0]
    coordinator = random.choice(COORDINATOR_NAMES)
    doctor = random.choice(DOCTOR_NAMES)
    tx_type = random.choice(TRANSPLANT_TYPES)
    months = random.choice(MONTHS_POST_TX)
    concern = random.choice(APPOINTMENT_CONCERNS)
    new_symptom = random.choice([
        "occasional mild itching",
        "a slight increase in blood pressure readings",
        "some difficulty concentrating",
        "occasional low-grade fever",
        "minor weight gain of about 3 pounds",
        "a new mild skin rash on my forearm",
    ])
    days_until = random.randint(3, 14)

    messages = [
        {
            "role": "system",
            "content": (
                "You are an experienced transplant coordinator conducting a post-transplant "
                "follow-up call. You are empathetic, professional, and knowledgeable about "
                "immunosuppression, rejection signs, and post-transplant care protocols. "
                "Your goal is to assess the patient's current status, address concerns, "
                "and escalate appropriately."
            ),
        },
        {
            "role": "coordinator",
            "content": (
                f"Hi {patient}, this is {coordinator} from the transplant team. I'm calling "
                f"to remind you about your upcoming {months}-month post-transplant check-up "
                f"with {doctor}, scheduled in {days_until} days. Do you have a few minutes?"
            ),
        },
        {
            "role": "patient",
            "content": "Of course, thank you for the reminder. I had it written down but it's good to confirm.",
        },
        {
            "role": "coordinator",
            "content": (
                f"Great. Before the appointment, I'd like to do a quick check-in. Overall, "
                f"how have you been feeling since your last visit? Any new symptoms or concerns "
                f"you'd like {doctor} to know about?"
            ),
        },
        {
            "role": "patient",
            "content": (
                f"Overall I feel okay, but I've been noticing {new_symptom}. I wasn't sure "
                f"if it was worth mentioning or if it's normal."
            ),
        },
        {
            "role": "coordinator",
            "content": (
                f"Thank you for flagging that, {first} — it's absolutely worth mentioning. "
                f"{new_symptom.capitalize()} can sometimes be related to your medications or the "
                f"transplanted {tx_type} adapting over time, but {doctor} will want to assess it. "
                f"I'll make a note in your chart so it's on the agenda for your visit. "
                f"Is there anything specific you're wondering about regarding the appointment itself?"
            ),
        },
        {
            "role": "patient",
            "content": f"Yes, actually — I have a question about {concern}.",
        },
        {
            "role": "coordinator",
            "content": (
                (
                    f"Yes, please bring all your current medications — ideally in their original "
                    f"bottles — so {doctor} can review them. If that's not possible, a written list "
                    f"with doses and timing works well too."
                ) if "medications" in concern else
                (
                    f"Yes, we will be drawing blood that day, so please fast for at least 8 hours "
                    f"beforehand. Water is fine to drink, and please take your morning medications "
                    f"with just a small sip of water unless told otherwise."
                ) if "fast" in concern else
                (
                    f"Your appointment should take approximately 2 to 3 hours — that includes "
                    f"registration, lab draws, imaging if needed, and the visit with {doctor}. "
                    f"Try to plan for the full block to avoid feeling rushed."
                ) if "long" in concern else
                (
                    f"Absolutely — your caregiver is welcome and encouraged to be with you during "
                    f"the visit. Having a support person present can be really helpful for "
                    f"remembering information and asking questions."
                ) if "caregiver" in concern else
                (
                    f"There is a parking structure adjacent to the clinic building. "
                    f"Transplant patients are eligible for a discounted parking pass — just mention "
                    f"it at the parking booth and show your appointment confirmation."
                ) if "parking" in concern else
                (
                    f"If you miss a morning dose, take it as soon as you remember — unless it's "
                    f"within 2 hours of your next scheduled dose. Please don't double up. "
                    f"Let {doctor} know at the appointment and we'll check your drug levels."
                ) if "miss" in concern else
                (
                    f"We aim to have results back within 24 to 48 hours, but some tests may take "
                    f"a few days. We'll call you with any results that need immediate attention, "
                    f"and the rest will be available through your patient portal."
                )
            ),
        },
        {
            "role": "patient",
            "content": "That's really helpful, thank you. I feel more prepared now.",
        },
        {
            "role": "coordinator",
            "content": (
                f"Wonderful. Is there anything else on your mind before the visit, {first}? "
                f"Sometimes patients think of questions between calls that they want to make "
                f"sure get addressed."
            ),
        },
        {
            "role": "patient",
            "content": (
                "I think I'm good for now. I'll write down my questions before the appointment. "
                "Thanks for checking in."
            ),
        },
        {
            "role": "coordinator",
            "content": (
                f"That's a great idea — keeping a running list is one of the best ways to make "
                f"the most of your time with {doctor}. We'll see you in {days_until} days, {first}. "
                f"Don't hesitate to call us before then if anything comes up. Take care!"
            ),
        },
    ]

    return {
        "id": f"conv_{idx:04d}",
        "scenario": "followup_appointment",
        "transplant_type": tx_type,
        "months_post_transplant": months,
        "messages": messages,
    }


def scenario_emotional_distress(idx: int) -> dict:
    patient = random.choice(PATIENT_NAMES)
    first = patient.split()[0]
    coordinator = random.choice(COORDINATOR_NAMES)
    doctor = random.choice(DOCTOR_NAMES)
    tx_type = random.choice(TRANSPLANT_TYPES)
    months = random.choice(MONTHS_POST_TX)
    stressor = random.choice(EMOTIONAL_STRESSORS)
    resource = random.choice([
        "our transplant social worker, who specializes in exactly these kinds of challenges",
        "a transplant peer support group where you can connect with others who share similar experiences",
        "our licensed counselor who works specifically with transplant patients",
        "the Transplant Living Foundation's online community and helpline",
        "a mindfulness-based stress reduction program tailored for chronic illness",
    ])

    messages = [
        {
            "role": "system",
            "content": (
                "You are an experienced transplant coordinator conducting a post-transplant "
                "follow-up call. You are empathetic, professional, and knowledgeable about "
                "immunosuppression, rejection signs, and post-transplant care protocols. "
                "Your goal is to assess the patient's current status, address concerns, "
                "and escalate appropriately."
            ),
        },
        {
            "role": "coordinator",
            "content": (
                f"Hi {patient}, this is {coordinator} from the transplant clinic. "
                f"I'm calling to check in on how things are going for you at "
                f"{months} months post-transplant. How are you doing today?"
            ),
        },
        {
            "role": "patient",
            "content": (
                f"Honestly, not that great emotionally. Physically I'm okay, but "
                f"{stressor}. I didn't know if I should bring this up or not."
            ),
        },
        {
            "role": "coordinator",
            "content": (
                f"I'm really glad you shared that with me, {first}, and please know "
                f"you absolutely should bring it up — this is just as important as your "
                f"physical health. What you're feeling is something many transplant recipients "
                f"experience, and it doesn't make you weak or ungrateful. Can you tell me "
                f"a little more about what's been going on?"
            ),
        },
        {
            "role": "patient",
            "content": (
                "It's been building up for a while. Some days I feel completely overwhelmed "
                "and I don't really see who I can talk to about this. My family is supportive "
                "but they don't really understand the full picture."
            ),
        },
        {
            "role": "coordinator",
            "content": (
                f"That sense of isolation is very real and very common after a transplant, {first}. "
                f"Even with loving family around you, it can feel like no one truly understands "
                f"unless they've been through it themselves. Your feelings are completely valid. "
                f"I want to make sure you have the right support. Have you been sleeping okay, "
                f"and are you finding any enjoyment in daily activities, or has that been difficult?"
            ),
        },
        {
            "role": "patient",
            "content": (
                "Sleep has been disrupted — I wake up anxious in the middle of the night. "
                "And activities I used to enjoy feel less appealing. I still do them but "
                "I don't get the same satisfaction."
            ),
        },
        {
            "role": "coordinator",
            "content": (
                f"What you're describing — disrupted sleep, reduced enjoyment in activities, "
                f"persistent anxiety — these are signs that you could really benefit from "
                f"professional support, and there's no shame in that at all. I'd like to "
                f"connect you with {resource}. They work with transplant patients regularly "
                f"and understand the unique emotional landscape you're navigating. "
                f"Would you be open to that?"
            ),
        },
        {
            "role": "patient",
            "content": "I think I'd be willing to try. I just didn't know that was something available to me.",
        },
        {
            "role": "coordinator",
            "content": (
                f"It absolutely is, and I'm so glad you're open to it. I'm going to send a "
                f"referral today and someone will reach out to you within 2 to 3 business days "
                f"to schedule a first appointment. In the meantime, I also want to mention this "
                f"to {doctor} at your next visit so your whole team is aware and aligned in "
                f"supporting you. Is that okay with you?"
            ),
        },
        {
            "role": "patient",
            "content": "Yes, that's fine. I appreciate that you're taking this seriously.",
        },
        {
            "role": "coordinator",
            "content": (
                f"Of course — you deserve comprehensive care, and that includes your mental "
                f"well-being, {first}. You've been through something enormous, and it takes "
                f"real strength to acknowledge when you need support. Please reach out to us "
                f"at any time — day or night there's a nurse line available. You are not alone "
                f"in this. Take good care, and I'll follow up with you next week to see how "
                f"you're feeling."
            ),
        },
    ]

    return {
        "id": f"conv_{idx:04d}",
        "scenario": "emotional_distress",
        "transplant_type": tx_type,
        "months_post_transplant": months,
        "messages": messages,
    }


def scenario_lifestyle_changes(idx: int) -> dict:
    patient = random.choice(PATIENT_NAMES)
    first = patient.split()[0]
    coordinator = random.choice(COORDINATOR_NAMES)
    doctor = random.choice(DOCTOR_NAMES)
    tx_type = random.choice(TRANSPLANT_TYPES)
    months = random.choice(MONTHS_POST_TX)
    dietary_challenge = random.choice(DIETARY_CHALLENGES)
    exercise_level = random.choice([
        "I've been doing light walking for about 20 minutes most days",
        "I've been mostly sedentary — I'm nervous about overdoing it",
        "I've started a gentle yoga class twice a week",
        "I've been trying to do some stretching in the morning",
        "I've been pretty active — maybe too much, I'm not sure",
        "I haven't started any structured exercise yet",
    ])
    win = random.choice([
        "I did manage to stop drinking alcohol completely",
        "I've been taking my medications exactly on time every day",
        "I've been tracking my fluid intake and blood pressure daily",
        "I cut out processed foods almost entirely",
        "I've been consistent with my sun protection routine",
        "I joined an online support forum for transplant recipients",
    ])

    messages = [
        {
            "role": "system",
            "content": (
                "You are an experienced transplant coordinator conducting a post-transplant "
                "follow-up call. You are empathetic, professional, and knowledgeable about "
                "immunosuppression, rejection signs, and post-transplant care protocols. "
                "Your goal is to assess the patient's current status, address concerns, "
                "and escalate appropriately."
            ),
        },
        {
            "role": "coordinator",
            "content": (
                f"Hello {patient}, this is {coordinator} from the transplant team. "
                f"I'm doing a lifestyle check-in call — just {months} months in, "
                f"this is a really important time to make sure you're adjusting well. "
                f"Do you have a few minutes to chat?"
            ),
        },
        {
            "role": "patient",
            "content": "Sure, I've actually been meaning to ask some questions.",
        },
        {
            "role": "coordinator",
            "content": (
                f"Perfect. Let's start with diet — one of the most impactful areas after "
                f"a {tx_type} transplant. How are you managing the dietary guidelines we "
                f"discussed at discharge?"
            ),
        },
        {
            "role": "patient",
            "content": (
                f"Most of it is going okay, but I have to admit: {dietary_challenge}. "
                f"I try but sometimes I slip up."
            ),
        },
        {
            "role": "coordinator",
            "content": (
                f"I really appreciate your honesty, {first} — that's very helpful. "
                f"The challenges you're describing are very common, and occasional slips "
                f"don't mean you've failed. The goal is consistency over time, not perfection. "
                f"For the specific issue you mentioned, I'd like to connect you with our "
                f"transplant dietitian who can help you find practical substitutes and meal "
                f"strategies that fit your real life. Would that be useful?"
            ),
        },
        {
            "role": "patient",
            "content": "Yes, actually that would help a lot. I've been guessing a lot of the time.",
        },
        {
            "role": "coordinator",
            "content": (
                f"Great — I'll put in a referral for a dietitian consult. Now let's talk "
                f"about physical activity. What has your exercise routine looked like lately?"
            ),
        },
        {
            "role": "patient",
            "content": exercise_level + ".",
        },
        {
            "role": "coordinator",
            "content": (
                (
                    f"That's a really good start, {first}. Twenty minutes of walking most days "
                    f"is exactly the kind of consistent, low-impact activity that supports "
                    f"{tx_type} health. As you feel stronger, we can gradually increase duration "
                    f"and add light strength work — we'll revisit goals with {doctor} at your next visit."
                ) if "walking" in exercise_level else
                (
                    f"It's completely understandable to be cautious, {first}. The good news is "
                    f"that gentle activity — like a 15-minute daily walk — is actually beneficial "
                    f"and safe at your stage. We have a cardiac rehab-style program adapted for "
                    f"transplant patients if you'd like a guided, supervised start. Would that interest you?"
                ) if "sedentary" in exercise_level or "haven't" in exercise_level else
                (
                    f"That sounds lovely — yoga is excellent for flexibility, breathing, and stress "
                    f"reduction. Just be mindful of hot yoga or any heated classes, as immunosuppressed "
                    f"patients can be more sensitive to heat and infection risks in shared spaces."
                ) if "yoga" in exercise_level else
                (
                    f"Morning stretching is a wonderful habit to build on, {first}. "
                    f"As you feel more comfortable, adding short walks and gradually increasing "
                    f"duration is the next step. The key is listening to your body and not "
                    f"pushing through pain."
                ) if "stretching" in exercise_level else
                (
                    f"That enthusiasm is great, {first}, but it's important we don't overdo it "
                    f"in the early months. I'd recommend keeping sessions under 45 minutes and "
                    f"avoiding high-impact activities for now. Let's check in with {doctor} "
                    f"about safe intensity levels for where you are in recovery."
                ) if "too much" in exercise_level else
                (
                    f"Starting from scratch is perfectly fine, {first}. The recommendation is "
                    f"to begin with 10 to 15 minutes of easy walking daily and build from there. "
                    f"Consistency matters more than intensity at this stage."
                )
            ),
        },
        {
            "role": "patient",
            "content": (
                f"That makes sense. I also wanted you to know that {win}. "
                f"I'm proud of that one."
            ),
        },
        {
            "role": "coordinator",
            "content": (
                f"You should absolutely be proud of that, {first} — that is a significant "
                f"achievement and it makes a real difference in your long-term outcomes. "
                f"Please keep that up. I'll make sure to note this progress in your chart "
                f"and share it with {doctor} — positive momentum matters clinically too. "
                f"Is there anything else on the lifestyle front you'd like to discuss?"
            ),
        },
        {
            "role": "patient",
            "content": "No, I think that covers it. This was really helpful.",
        },
        {
            "role": "coordinator",
            "content": (
                f"I'm glad, {first}. You're doing a lot of things right, and the places "
                f"where you're struggling are completely normal — we'll address them together. "
                f"I'll follow up after the dietitian referral goes through. Keep taking your "
                f"medications on schedule, stay active, and don't hesitate to call if anything "
                f"comes up. Talk soon!"
            ),
        },
    ]

    return {
        "id": f"conv_{idx:04d}",
        "scenario": "lifestyle_changes",
        "transplant_type": tx_type,
        "months_post_transplant": months,
        "messages": messages,
    }


# ─── Ambiguous scenario: borderline creatinine ───────────────────────────────

def scenario_borderline_creatinine(idx: int) -> dict:
    patient = random.choice(PATIENT_NAMES)
    first = patient.split()[0]
    coordinator = random.choice(COORDINATOR_NAMES)
    doctor = random.choice(DOCTOR_NAMES)
    tx_type = random.choice(TRANSPLANT_TYPES)
    months = random.choice(MONTHS_POST_TX)
    baseline = round(random.uniform(0.9, 1.4), 1)
    creatinine = round(baseline + random.uniform(0.2, 0.4), 1)
    meds = random.choice(IMMUNOSUPPRESSANTS)
    cause = random.choice(BORDERLINE_ELEVATION_CAUSES)
    er_criteria = random.choice(WATCHFUL_WAITING_CRITERIA)
    repeat_interval = random.choice(MONITORING_INTERVALS)
    minor_symptom = random.choice([
        "slightly more tired than usual",
        "mildly reduced urine output — though I've also been drinking less water",
        "a bit of ankle swelling that comes and goes",
        "some lower back discomfort I attributed to my sleeping position",
        "mild bloating, but I also changed my diet recently",
    ])

    messages = [
        {
            "role": "system",
            "content": (
                "You are an experienced transplant coordinator conducting a post-transplant "
                "follow-up call. You are empathetic, professional, and knowledgeable about "
                "immunosuppression, rejection signs, and post-transplant care protocols. "
                "Your goal is to assess the patient's current status, address concerns, "
                "and escalate appropriately."
            ),
        },
        {
            "role": "coordinator",
            "content": (
                f"Hello, may I speak with {patient}? This is {coordinator} calling from "
                f"the transplant clinic regarding your most recent lab work. Is this a good time?"
            ),
        },
        {
            "role": "patient",
            "content": f"Yes, this is {first}. Is something wrong with my results?",
        },
        {
            "role": "coordinator",
            "content": (
                f"Nothing that requires immediate alarm, {first}, but I do want to discuss "
                f"what we saw. Your creatinine came back at {creatinine} mg/dL — a modest rise "
                f"from your recent baseline of {baseline} mg/dL. It's a small difference and "
                f"there are several possible explanations, not all of them concerning. Can I ask "
                f"a few questions to help us figure out what's going on?"
            ),
        },
        {
            "role": "patient",
            "content": (
                f"Of course. I'll mention — I've had {cause} recently. Could that explain it?"
            ),
        },
        {
            "role": "coordinator",
            "content": (
                f"Absolutely, that's a very plausible explanation and it's helpful that you "
                f"made that connection. {cause.capitalize()} is a known cause of transient "
                f"creatinine elevation and doesn't necessarily mean anything is wrong with "
                f"the transplanted {tx_type}. That said, at {months} months post-transplant "
                f"we always keep rejection and medication-related causes in our differential — "
                f"not because that's what we think this is, but because ruling it out is our "
                f"responsibility. Have you noticed any physical symptoms at all?"
            ),
        },
        {
            "role": "patient",
            "content": (
                f"Actually, I have noticed I've been {minor_symptom}. But I wasn't sure "
                f"if it was related or just coincidence."
            ),
        },
        {
            "role": "coordinator",
            "content": (
                f"I'm glad you mentioned that. {minor_symptom.capitalize()} on its own isn't "
                f"diagnostic of anything, but in combination with even a mild creatinine rise "
                f"it's worth tracking carefully. Rather than bringing you in immediately, I'd "
                f"like you to repeat your labs in {repeat_interval}. That gives us a trend. "
                f"If the creatinine is holding steady or falling, we can likely attribute this "
                f"to what you described and continue monitoring. If it keeps rising, I'll need "
                f"you in the clinic that same day. Does that approach make sense to you?"
            ),
        },
        {
            "role": "patient",
            "content": (
                "Yes, that makes sense. But what should I watch for in the meantime — "
                "how will I know if something more serious is developing?"
            ),
        },
        {
            "role": "coordinator",
            "content": (
                f"Great question. Your red flags before the repeat labs are: {er_criteria}. "
                f"If any of those occur, don't wait — go directly to the emergency room and "
                f"call us on the way. Otherwise, stay well hydrated with water, take your "
                f"{meds[0]} and {meds[1]} on schedule without missing doses, and avoid NSAIDs "
                f"like ibuprofen or naproxen, which can stress the transplanted {tx_type}. "
                f"I want to be transparent: this may be completely benign, but we treat every "
                f"lab value seriously because early detection is how we protect you."
            ),
        },
        {
            "role": "patient",
            "content": (
                "I really appreciate that you're taking it seriously without panicking me. "
                "I'll increase my water intake and watch for those warning signs."
            ),
        },
        {
            "role": "coordinator",
            "content": (
                f"That's exactly the right approach, {first}. I'll have the lab order sent to "
                f"your usual draw site today. After we get those results I'll call you personally "
                f"to review the trend together. If anything changes before then, please don't "
                f"hesitate to call the clinic line. {doctor} is also aware and will be looped "
                f"in based on what the repeat shows. Take care — we'll be in touch soon."
            ),
        },
    ]

    return {
        "id": f"conv_{idx:04d}",
        "scenario": "borderline_creatinine",
        "transplant_type": tx_type,
        "months_post_transplant": months,
        "messages": messages,
    }


# ─── Ambiguous scenario: tacrolimus toxicity vs rejection ─────────────────────

def scenario_tacrolimus_toxicity_vs_rejection(idx: int) -> dict:
    patient = random.choice(PATIENT_NAMES)
    first = patient.split()[0]
    coordinator = random.choice(COORDINATOR_NAMES)
    doctor = random.choice(DOCTOR_NAMES)
    tx_type = random.choice(TRANSPLANT_TYPES)
    months = random.choice(MONTHS_POST_TX)
    baseline = round(random.uniform(0.9, 1.4), 1)
    presentation = random.choice(TOXICITY_PRESENTATIONS)
    lo, hi = presentation["creatinine_delta_range"]
    creatinine = round(baseline + random.uniform(lo, hi), 1)
    meds = random.choice(IMMUNOSUPPRESSANTS)
    dietary_factor = random.choice(DIETARY_TACROLIMUS_FACTORS)

    messages = [
        {
            "role": "system",
            "content": (
                "You are an experienced transplant coordinator conducting a post-transplant "
                "follow-up call. You are empathetic, professional, and knowledgeable about "
                "immunosuppression, rejection signs, and post-transplant care protocols. "
                "Your goal is to assess the patient's current status, address concerns, "
                "and escalate appropriately."
            ),
        },
        {
            "role": "coordinator",
            "content": (
                f"Hello {patient}, this is {coordinator} from the transplant team. "
                f"I'm calling because your recent labs flagged some changes I want to "
                f"review with you. Are you free to talk for a few minutes?"
            ),
        },
        {
            "role": "patient",
            "content": (
                f"Yes — actually I'm glad you called. I've been experiencing "
                f"{presentation['symptoms']} and wasn't sure if I should come in."
            ),
        },
        {
            "role": "coordinator",
            "content": (
                f"I'm very glad you mentioned that right away, {first} — it's directly "
                f"relevant to what I'm calling about. Your creatinine has come in at "
                f"{creatinine} mg/dL, up from your baseline of {baseline} mg/dL. Combined "
                f"with the {presentation['symptoms']} you're describing, I need to be "
                f"straightforward with you: we have two main possibilities to consider and "
                f"at this stage we can't rule either one out without more testing. One is "
                f"{presentation['toxicity_note']} — meaning your tacrolimus levels may be "
                f"too high. The other is an early rejection episode. Both can produce a "
                f"very similar picture. Does that make sense so far?"
            ),
        },
        {
            "role": "patient",
            "content": (
                f"That's unsettling to hear. Is there any way to tell which it is without "
                f"guessing? I should also mention — I've {dietary_factor}. Could that matter?"
            ),
        },
        {
            "role": "coordinator",
            "content": (
                f"That detail is actually very important — thank you for sharing it. "
                f"The fact that you've {dietary_factor} can significantly affect your "
                f"tacrolimus blood levels, sometimes pushing them into a toxic range. "
                f"That shifts our thinking somewhat toward a toxicity picture, but we "
                f"still need objective data to know for certain. Here is the plan: I'm "
                f"ordering an urgent tacrolimus trough level today alongside a full "
                f"metabolic panel. If your tacrolimus level comes back supratherapeutic "
                f"— meaning too high — we'll adjust the dose and track the creatinine "
                f"closely. If the level is in range or low, rejection moves up our "
                f"differential and {doctor} will discuss whether a biopsy of the "
                f"transplanted {tx_type} is the next step."
            ),
        },
        {
            "role": "patient",
            "content": (
                "A biopsy — that sounds serious. Should I go to the emergency room right now?"
            ),
        },
        {
            "role": "coordinator",
            "content": (
                f"I don't believe you need the ER right now, {first}, but I do need you "
                f"to come to the clinic today or first thing tomorrow for the blood draw — "
                f"do not wait for your next scheduled appointment. A biopsy, if it becomes "
                f"necessary, is a same-day procedure done under imaging guidance. I know "
                f"the word sounds alarming, but it is the definitive way to distinguish "
                f"between toxicity and rejection at the tissue level. Treating the wrong "
                f"condition — suppressing an immune response that's actually being driven "
                f"by toxic drug levels — can harm the {tx_type}. Getting the diagnosis "
                f"right is worth the procedure."
            ),
        },
        {
            "role": "patient",
            "content": (
                "I understand. I'll come in tomorrow morning. Should I stop my tacrolimus "
                "in the meantime since it might be too high?"
            ),
        },
        {
            "role": "coordinator",
            "content": (
                f"Please do not stop your tacrolimus — this is very important. Stopping "
                f"abruptly creates far greater risk than a temporarily elevated level, "
                f"because it removes all protection against rejection. Take your usual "
                f"dose of {meds[0]} tonight, but tomorrow morning hold it until after "
                f"the blood draw so we capture an accurate trough level. Continue your "
                f"{meds[1]} and {meds[2]} on schedule as normal. Also, please stop "
                f"{'the ' + dietary_factor.split(' recently')[0] if 'recently' in dietary_factor else dietary_factor} "
                f"until we've spoken again. I'm flagging your case for {doctor} to review "
                f"the moment results arrive. Any other questions?"
            ),
        },
        {
            "role": "patient",
            "content": (
                "No — I think I understand. It's scary not knowing which it is, but "
                "I'm glad you explained both possibilities clearly. I'll be there tomorrow."
            ),
        },
        {
            "role": "coordinator",
            "content": (
                f"That's exactly the right mindset, {first}. Uncertainty is hard, but "
                f"we're moving quickly and you'll have answers very soon. Our clinic line "
                f"is available around the clock — if your symptoms worsen overnight, "
                f"particularly if you develop fever, severe pain, or markedly reduced "
                f"urination, call us or go to the ER. See you tomorrow and take care."
            ),
        },
    ]

    return {
        "id": f"conv_{idx:04d}",
        "scenario": "tacrolimus_toxicity_vs_rejection",
        "transplant_type": tx_type,
        "months_post_transplant": months,
        "messages": messages,
    }


# ─── Ambiguous scenario: stable labs, concerning symptoms ─────────────────────

def scenario_stable_labs_concerning_symptoms(idx: int) -> dict:
    patient = random.choice(PATIENT_NAMES)
    first = patient.split()[0]
    coordinator = random.choice(COORDINATOR_NAMES)
    doctor = random.choice(DOCTOR_NAMES)
    tx_type = random.choice(TRANSPLANT_TYPES)
    months = random.choice(MONTHS_POST_TX)
    baseline_creatinine = round(random.uniform(0.9, 1.3), 1)
    meds = random.choice(IMMUNOSUPPRESSANTS)
    infection_pres = random.choice(INFECTION_PRESENTATIONS)
    days_of_symptoms = random.randint(2, 7)

    messages = [
        {
            "role": "system",
            "content": (
                "You are an experienced transplant coordinator conducting a post-transplant "
                "follow-up call. You are empathetic, professional, and knowledgeable about "
                "immunosuppression, rejection signs, and post-transplant care protocols. "
                "Your goal is to assess the patient's current status, address concerns, "
                "and escalate appropriately."
            ),
        },
        {
            "role": "coordinator",
            "content": (
                f"Hi {patient}, this is {coordinator} calling from the transplant clinic "
                f"for your {months}-month routine check-in. Your labs from last week "
                f"looked reassuring — creatinine stable at {baseline_creatinine} mg/dL. "
                f"How have you been feeling overall?"
            ),
        },
        {
            "role": "patient",
            "content": (
                f"Now that you ask — I've been dealing with {infection_pres['symptom']} "
                f"for the past {days_of_symptoms} days. I assumed it was just a regular "
                f"bug and didn't want to bother the clinic over something minor."
            ),
        },
        {
            "role": "coordinator",
            "content": (
                f"I'm really glad you mentioned that, {first} — and please, never hesitate "
                f"to call us with symptoms like these. I know your labs look stable, but I "
                f"need to explain something important: because your immunosuppressants lower "
                f"your immune response, your body's ability to mount a visible inflammatory "
                f"reaction is muted. That means infections that a healthy person's immune "
                f"system would handle easily can escalate quickly for you — and they won't "
                f"always show up in creatinine until the illness has already progressed. "
                f"Can I ask — have you been checking your temperature?"
            ),
        },
        {
            "role": "patient",
            "content": (
                "Yes, it's been in the range I described. I've just been resting and "
                "drinking fluids — that's what I'd normally do for something like this."
            ),
        },
        {
            "role": "coordinator",
            "content": (
                f"Resting and hydrating are good instincts, but with your level of "
                f"immunosuppression, a wait-and-see approach isn't safe here. Based on "
                f"what you're describing — {infection_pres['symptom']} for {days_of_symptoms} "
                f"days — my concern is {infection_pres['likely_source']}. The risk "
                f"profile for you is simply not the same as for someone with a healthy "
                f"immune system. I'd like you to come in today, or go to an urgent care "
                f"facility that can reach us. Please don't take any over-the-counter fever "
                f"reducers before being evaluated, and absolutely no NSAIDs."
            ),
        },
        {
            "role": "patient",
            "content": (
                "Really? I didn't think it was that serious — my creatinine is fine so I "
                "assumed the transplant was okay and I just had a regular cold."
            ),
        },
        {
            "role": "coordinator",
            "content": (
                f"That reasoning makes intuitive sense, but transplant care is a bit "
                f"counterintuitive here, {first}. Creatinine reflects organ function and "
                f"can look completely normal even while an infection is developing rapidly. "
                f"By the time creatinine rises due to infection, the illness has already "
                f"progressed further than we'd want. We're not in a panic — but we are "
                f"in 'act today' mode. I'm going to call ahead to our clinic or coordinate "
                f"with urgent care so they know your transplant history and your "
                f"immunosuppression regimen: {meds[0]}, {meds[1]}, and {meds[2]}. "
                f"That way they can evaluate you appropriately from the moment you arrive."
            ),
        },
        {
            "role": "patient",
            "content": "Okay — I'll go in. Is there anything specific I should tell them?",
        },
        {
            "role": "coordinator",
            "content": (
                f"Yes — lead with this: 'I am a post-{tx_type}-transplant patient on "
                f"immunosuppressants.' Show them your medication list if you have it with "
                f"you. Ask them to run a full blood count, comprehensive metabolic panel, "
                f"urinalysis, and depending on your symptoms a chest X-ray may be needed. "
                f"Please also ask them to contact our transplant team directly — the number "
                f"is on your clinic card. If at any point your fever climbs above 38.5°C "
                f"or you feel significantly worse, bypass urgent care and go directly to "
                f"the emergency room. Can you head in within the next hour or two?"
            ),
        },
        {
            "role": "patient",
            "content": (
                "Yes, I can go now. Thank you for explaining — I genuinely didn't realize "
                "how different this is from a normal cold."
            ),
        },
        {
            "role": "coordinator",
            "content": (
                f"That's exactly why we do these check-ins, {first}. You did the right "
                f"thing by mentioning it when I asked — many patients don't, thinking "
                f"they'd be wasting our time. Please call us once you've been seen and "
                f"have results. We'll be in touch with {doctor} as well so the whole "
                f"team is aware. Take care, and please go soon."
            ),
        },
    ]

    return {
        "id": f"conv_{idx:04d}",
        "scenario": "stable_labs_concerning_symptoms",
        "transplant_type": tx_type,
        "months_post_transplant": months,
        "messages": messages,
    }


# ─── Longitudinal helpers ─────────────────────────────────────────────────────

def _creatinine_trend_text(prev_val: float, curr_val: float) -> str:
    delta = curr_val - prev_val
    if delta > 0.5:
        return "significantly higher than"
    elif delta > 0.2:
        return "notably higher than"
    elif delta > 0.05:
        return "slightly elevated compared to"
    elif abs(delta) <= 0.05:
        return "essentially stable compared to"
    elif delta < -0.2:
        return "notably improved from"
    else:
        return "slightly improved from"


def _ordinal(n: int) -> str:
    mapping = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth"}
    return mapping.get(n, f"{n}th")


def _gap_text(prev_month: int, curr_month: int) -> str:
    diff = curr_month - prev_month
    if diff == 1:
        return "last month"
    elif diff == 2:
        return "two months ago"
    elif diff == 3:
        return "three months ago"
    elif diff == 6:
        return "six months ago"
    elif diff == 12:
        return "a year ago"
    else:
        return f"{diff} months ago"


def _month_label(month: int) -> str:
    if month == 12:
        return "one-year"
    elif month == 24:
        return "two-year"
    else:
        return f"{month}-month"


def _generate_visit_history(rng, trajectory: str, baseline: float, months: list) -> list:
    """Generate creatinine values and clinical status for each visit month."""
    visits = []
    prev = baseline
    n = len(months)

    for i, month in enumerate(months):
        progress = i / max(n - 1, 1)

        if trajectory == "stable_improving":
            creatinine = max(0.7, round(prev + rng.uniform(-0.1, 0.1), 1))
            status = "stable" if abs(creatinine - baseline) < 0.15 else "borderline"

        elif trajectory == "rocky_stabilizing":
            if progress < 0.5:
                delta = rng.uniform(-0.15, 0.45)
            else:
                target = baseline + 0.05
                delta = (target - prev) * 0.4 + rng.uniform(-0.1, 0.1)
            creatinine = max(0.7, round(prev + delta, 1))
            if creatinine > baseline + 0.35:
                status = "elevated"
            elif creatinine > baseline + 0.15:
                status = "borderline"
            else:
                status = "stable"

        elif trajectory == "concerning_progression":
            creatinine = round(prev + rng.uniform(0.05, 0.2), 1)
            if creatinine > baseline + 0.6:
                status = "elevated"
            elif creatinine > baseline + 0.3:
                status = "borderline"
            else:
                status = "mildly_elevated"

        else:  # relapse_recovery
            if i == 1:
                creatinine = round(baseline + rng.uniform(1.0, 1.8), 1)
                status = "acutely_elevated"
            elif i == 2:
                creatinine = max(baseline - 0.1, round(prev - rng.uniform(0.4, 0.8), 1))
                status = "improving"
            elif i >= 3:
                creatinine = max(0.7, round(baseline + rng.uniform(-0.05, 0.15), 1))
                status = "recovered"
            else:
                creatinine = max(0.7, round(baseline + rng.uniform(-0.1, 0.1), 1))
                status = "stable"

        visits.append({"month": month, "creatinine": creatinine, "status": status})
        prev = creatinine

    return visits


def make_longitudinal_profiles(seed: int = 999, n: int = 30) -> list:
    """
    Generate n longitudinal patient profiles with full visit histories.
    Trajectories: stable_improving (10), rocky_stabilizing (10),
                  concerning_progression (5), relapse_recovery (5).
    """
    rng = random.Random(seed)
    trajectory_plan = (
        ["stable_improving"] * 10
        + ["rocky_stabilizing"] * 10
        + ["concerning_progression"] * 5
        + ["relapse_recovery"] * 5
    )

    patient_pool = PATIENT_NAMES.copy()
    rng.shuffle(patient_pool)

    profiles = []
    for i in range(n):
        name = patient_pool[i % len(patient_pool)]
        tx_type = rng.choice(TRANSPLANT_TYPES)
        meds = rng.choice(IMMUNOSUPPRESSANTS)
        coordinator = rng.choice(COORDINATOR_NAMES)
        doctor = rng.choice(DOCTOR_NAMES)
        baseline = round(rng.uniform(0.9, 1.3), 1)
        traj = trajectory_plan[i]

        possible_months = [1, 2, 3, 4, 6, 8, 10, 12, 18, 24]
        n_visits = rng.randint(4, 6)
        visit_months = sorted(rng.sample(possible_months, n_visits))

        visits = _generate_visit_history(rng, traj, baseline, visit_months)

        profiles.append({
            "patient_id": f"long_{i + 1:03d}",
            "name": name,
            "transplant_type": tx_type,
            "meds": meds,
            "coordinator": coordinator,
            "doctor": doctor,
            "baseline_creatinine": baseline,
            "trajectory": traj,
            "visits": visits,
        })

    return profiles


# ─── Longitudinal visit scenario ──────────────────────────────────────────────

def scenario_longitudinal_visit(profile: dict, visit_idx: int, conv_idx: int) -> dict:
    name = profile["name"]
    first = name.split()[0]
    coordinator = profile["coordinator"]
    doctor = profile["doctor"]
    tx_type = profile["transplant_type"]
    meds = profile["meds"]
    baseline = profile["baseline_creatinine"]
    traj = profile["trajectory"]

    current = profile["visits"][visit_idx]
    curr_month = current["month"]
    curr_creatinine = current["creatinine"]
    curr_status = current["status"]

    prior_visits = profile["visits"][:visit_idx]
    is_first_visit = visit_idx == 0
    visit_num = visit_idx + 1

    # ── Build time-anchored history references ──
    if prior_visits:
        last = prior_visits[-1]
        last_month = last["month"]
        last_creatinine = last["creatinine"]
        gap = _gap_text(last_month, curr_month)
        trend = _creatinine_trend_text(last_creatinine, curr_creatinine)
        history_prefix = (
            f"When we last spoke {gap} at your {_month_label(last_month)} check-in, "
            f"your creatinine was {last_creatinine} mg/dL. "
        )
        if len(prior_visits) >= 2:
            first_v = prior_visits[0]
            history_prefix += (
                f"Looking back at your full trend since month {first_v['month']}, "
                f"your creatinine started at {first_v['creatinine']} mg/dL and is now "
                f"{trend} that early reading. "
            )
    else:
        last_creatinine = baseline
        gap = None
        trend = None
        history_prefix = ""

    life_event = random.choice(LONGITUDINAL_LIFE_EVENTS)

    # ── Coordinator opening ──
    if is_first_visit:
        opening = (
            f"Hello, may I speak with {name}? This is {coordinator} from the "
            f"transplant clinic. I'm calling for your {_month_label(curr_month)} "
            f"post-transplant follow-up. Is this a good time to talk?"
        )
    else:
        opening = (
            f"Hello {first}, this is {coordinator} from the transplant team. "
            f"This is our {_ordinal(visit_num)} check-in call — you're now "
            f"{curr_month} months post-transplant. {history_prefix}"
            f"Do you have a few minutes?"
        )

    # ── Patient response ──
    if is_first_visit:
        patient_open = (
            f"Yes, this is {first}. I've been keeping a list of questions "
            f"— I'm glad you called."
        )
    elif curr_status == "acutely_elevated":
        patient_open = (
            f"Yes — I'm relieved you called. I've been feeling quite unwell "
            f"this past week and I was about to ring the clinic myself."
        )
    elif curr_status in ["improving", "recovered"]:
        patient_open = (
            f"Yes — and I have to say, I'm feeling so much better than last time "
            f"we spoke. Whatever we did seems to be working."
        )
    elif visit_idx >= 3 and curr_status == "stable":
        patient_open = (
            f"Of course. You know, after {curr_month} months I finally feel "
            f"like things are settling into a real routine — I even {life_event}."
        )
    else:
        patient_open = f"Yes, of course. I've been keeping notes since our last call."

    # ── Lab discussion ──
    if curr_status == "stable":
        lab_msg = (
            f"That's wonderful to hear, {first} — and your labs are reflecting it. "
            f"Your creatinine today is {curr_creatinine} mg/dL, "
            + (
                f"which is {trend} your {last_creatinine} mg/dL from {gap}. "
                if not is_first_visit
                else f"right in line with your baseline of {baseline} mg/dL. "
            )
            + f"This is exactly the kind of stability we aim for at {curr_month} months "
            f"post-transplant. How have you been feeling day to day?"
        )
    elif curr_status == "borderline":
        lab_msg = (
            f"Let me share your latest results, {first}. Your creatinine is "
            f"{curr_creatinine} mg/dL — "
            + (
                f"{trend} your previous {last_creatinine} mg/dL from {gap}. "
                if not is_first_visit
                else f"slightly above your baseline of {baseline} mg/dL. "
            )
            + f"It's a modest change, and there are several possible explanations. "
            f"I'd like to ask a few targeted questions before we decide on next steps. "
            f"How have you been feeling physically this past week?"
        )
    elif curr_status in ["elevated", "mildly_elevated"]:
        lab_msg = (
            f"I want to go over your labs carefully, {first}. Your creatinine has "
            f"come in at {curr_creatinine} mg/dL — "
            + (
                f"{trend} your reading of {last_creatinine} mg/dL from {gap}. "
                f"Given the trend we've been tracking over these {curr_month} months, "
                if not is_first_visit
                else f"above your baseline of {baseline} mg/dL. "
            )
            + f"this is something {doctor} and I want to address promptly. "
            f"Can you walk me through how you've been feeling since we last spoke?"
        )
    elif curr_status == "acutely_elevated":
        lab_msg = (
            f"I need to discuss something urgent with you right away, {first}. "
            f"Your creatinine has risen sharply to {curr_creatinine} mg/dL — "
            + (
                f"up significantly from {last_creatinine} mg/dL {gap}. "
                if not is_first_visit
                else f"well above your baseline of {baseline} mg/dL. "
            )
            + f"This is an acute change and we need to act on it today. "
            f"Are you experiencing any symptoms right now?"
        )
    elif curr_status == "improving":
        lab_msg = (
            f"I have genuinely encouraging news for you today, {first}. "
            f"Your creatinine has come down to {curr_creatinine} mg/dL — "
            + (
                f"a meaningful improvement from {last_creatinine} mg/dL when we spoke {gap}. "
                if not is_first_visit
                else f"moving closer to your baseline of {baseline} mg/dL. "
            )
            + f"Your body is responding. How are you feeling compared to the last time we spoke?"
        )
    else:  # recovered
        lab_msg = (
            f"This is a call I'm genuinely happy to make, {first}. "
            f"Your creatinine is back down to {curr_creatinine} mg/dL — "
            f"essentially back to your baseline of {baseline} mg/dL. "
            f"Given everything you navigated earlier in your recovery, this is a "
            f"significant milestone. How are you feeling?"
        )

    # ── Patient to lab discussion ──
    if curr_status == "acutely_elevated":
        patient_to_labs = (
            f"I've had {random.choice(SIDE_EFFECTS)} and some decreased urine output "
            f"over the past few days. I wasn't sure if it was serious."
        )
    elif curr_status in ["improving", "recovered"]:
        patient_to_labs = (
            f"Much better, honestly. The symptoms from before have mostly resolved "
            f"and my energy is coming back. I even {life_event} this past month."
        )
    elif curr_status == "stable" and visit_idx >= 2:
        patient_to_labs = (
            f"Really well, all things considered. I {life_event} recently which felt "
            f"like a big step. My energy levels have been good."
        )
    else:
        patient_to_labs = (
            f"Mostly okay. I've had some {random.choice(SIDE_EFFECTS)} that comes and "
            f"goes, but overall I feel functional. I have been keeping up with my medications."
        )

    # ── Coordinator clinical response ──
    if curr_status == "acutely_elevated":
        clinical_response = (
            f"Decreased urine output combined with an acute creatinine rise at "
            f"{curr_month} months post-transplant is something we need to evaluate "
            f"today — not tomorrow, today. I'm going to arrange for you to come in "
            f"this afternoon or, if that's not possible, I want you at the emergency "
            f"room tonight. We'll need repeat labs, a drug level check, and likely an "
            f"ultrasound of the transplanted {tx_type}. We may also be looking at "
            f"a biopsy depending on what those show. Do you have a way to get to the "
            f"clinic today?"
        )
    elif curr_status in ["elevated", "mildly_elevated"]:
        clinical_response = (
            f"Given the trend we've been tracking, I want to make sure we stay ahead "
            f"of this. I'm going to ask {doctor} to review your case today and we'll "
            f"likely want you in for an urgent visit within the next 24 to 48 hours "
            f"for repeat labs and potentially a drug level check. In the meantime, "
            f"stay well hydrated, don't miss any doses of your {meds[0]} or {meds[1]}, "
            f"and avoid any NSAIDs. If you develop fever, significant swelling, or "
            f"your urine output drops noticeably, go directly to the ER."
        )
    elif curr_status == "borderline":
        clinical_response = (
            f"Given this mild change, I'd like to take a watchful waiting approach "
            f"rather than bring you in immediately. I'll order repeat labs in "
            f"{random.choice(MONITORING_INTERVALS)} and we'll look at the trend. "
            f"If the creatinine is stable or falling, we can likely attribute this "
            f"to a benign cause. If it continues to rise, I'll need you in the same "
            f"day. Red flags to watch for before then: "
            f"{random.choice(WATCHFUL_WAITING_CRITERIA)}. Call us immediately if any "
            f"of those occur."
        )
    elif curr_status in ["stable", "improving", "recovered"]:
        clinical_response = (
            f"This is exactly the kind of update we like to share. Your consistency "
            f"with medications and the lifestyle work you've been doing is showing up "
            f"in these results. For the {random.choice(SIDE_EFFECTS)} you mentioned — "
            f"let's flag that for {doctor} at your next visit. It's likely medication-related "
            f"and manageable, but worth documenting in your chart. Keep doing what you're doing."
        )
    else:
        clinical_response = (
            f"Based on what you're describing, I'll flag this for {doctor} and we'll "
            f"determine next steps together. Please continue your medications as prescribed "
            f"and call us if anything changes before your next scheduled visit."
        )

    # ── Patient reacts ──
    if curr_status == "acutely_elevated":
        patient_react = (
            f"That's alarming. I can get there this afternoon — should I bring anything?"
        )
    elif curr_status in ["elevated", "mildly_elevated"]:
        patient_react = (
            f"I'll come in. Is this heading toward rejection? I thought after "
            f"{curr_month} months I was past the high-risk window."
        )
    elif curr_status == "borderline":
        patient_react = (
            f"Okay — I'll watch for those signs and get the repeat labs done. "
            f"Should I be worried?"
        )
    else:
        patient_react = (
            f"That's a relief to hear. After everything at the beginning, stable "
            f"results feel like a real gift. Is there anything I should change going forward?"
        )

    # ── Coordinator closes ──
    if curr_status == "acutely_elevated":
        coordinator_close = (
            f"Bring your medication list and your insurance card. Don't eat or drink "
            f"anything other than water until you've been assessed, in case imaging "
            f"is needed. I'll call ahead to the clinic so they're ready for you. "
            f"Drive safely, {first} — or have someone bring you if possible. "
            f"We'll talk after your evaluation."
        )
    elif curr_status in ["elevated", "mildly_elevated"]:
        coordinator_close = (
            f"The rejection risk window does narrow over time, but it never fully "
            f"closes, and a trend like this still warrants investigation. The good "
            f"news is we caught this early because we've been tracking you consistently. "
            f"That's exactly why these check-ins matter. I'll have the scheduling team "
            f"call you within the hour, {first}. Don't miss your medications tonight."
        )
    elif curr_status == "borderline":
        coordinator_close = (
            f"You don't need to be alarmed, but you should stay alert. Borderline "
            f"values can resolve on their own or they can be the first signal of "
            f"something that needs intervention. That's why the repeat labs are so "
            f"important — they give us the trend rather than a single data point. "
            f"I'll send the order today, {first}. Talk soon."
        )
    else:
        coordinator_close = (
            f"At this stage, your focus should be maintaining what's working: "
            f"medication consistency, hydration, regular monitoring, and not "
            f"hesitating to call us with any changes. You've built a strong "
            f"foundation, {first}. Your {_month_label(curr_month)} results are "
            f"something {doctor} will be pleased to see. We'll talk again soon — "
            f"keep up the great work."
        )

    messages = [
        {"role": "system", "content": (
            "You are an experienced transplant coordinator conducting a post-transplant "
            "follow-up call. You are empathetic, professional, and knowledgeable about "
            "immunosuppression, rejection signs, and post-transplant care protocols. "
            "Your goal is to assess the patient's current status, address concerns, "
            "and escalate appropriately."
        )},
        {"role": "coordinator", "content": opening},
        {"role": "patient", "content": patient_open},
        {"role": "coordinator", "content": lab_msg},
        {"role": "patient", "content": patient_to_labs},
        {"role": "coordinator", "content": clinical_response},
        {"role": "patient", "content": patient_react},
        {"role": "coordinator", "content": coordinator_close},
    ]

    return {
        "id": f"conv_{conv_idx:04d}",
        "scenario": f"longitudinal_{traj}",
        "transplant_type": tx_type,
        "months_post_transplant": curr_month,
        "longitudinal_patient_id": profile["patient_id"],
        "visit_number": visit_num,
        "messages": messages,
    }


# ─── Format for Gemma 4 fine-tuning ──────────────────────────────────────────

def to_gemma_format(conv: dict) -> dict:
    """
    Convert to Gemma 4 chat format. The system message becomes part of the
    'user' turn in Gemma's expected structure. Non-system messages are
    interleaved as user (patient) / model (coordinator) turns.
    """
    system_content = ""
    chat_messages = []

    for msg in conv["messages"]:
        role = msg["role"]
        content = msg["content"]

        if role == "system":
            system_content = content
        elif role == "coordinator":
            chat_messages.append({"role": "model", "content": content})
        elif role == "patient":
            chat_messages.append({"role": "user", "content": content})

    # Gemma expects conversation to start with a user turn.
    # Prepend the system context to the first user turn if it exists.
    if system_content and chat_messages:
        first_user_idx = next(
            (i for i, m in enumerate(chat_messages) if m["role"] == "user"), None
        )
        if first_user_idx is not None:
            # Insert system as a special prefix
            chat_messages.insert(
                0, {"role": "system", "content": system_content}
            )
        # If coordinator speaks first, wrap with user system turn
        elif chat_messages[0]["role"] == "model":
            chat_messages.insert(
                0,
                {
                    "role": "user",
                    "content": f"[System: {system_content}]\n\n[Patient is ready to receive the coordinator's call.]",
                },
            )

    return {
        "id": conv["id"],
        "scenario": conv["scenario"],
        "transplant_type": conv["transplant_type"],
        "months_post_transplant": conv["months_post_transplant"],
        "messages": chat_messages,
    }


# ─── Main generation ──────────────────────────────────────────────────────────

def main():
    from collections import Counter

    random.seed(42)

    # ── Original 1 000 conversations (4 template scenarios) ──────────────────
    base_generators = [
        scenario_elevated_creatinine,
        scenario_followup_appointment,
        scenario_emotional_distress,
        scenario_lifestyle_changes,
    ]
    base_total = 1000
    per_scenario = base_total // 4
    base_counts = [per_scenario] * 4
    for i in range(base_total % 4):
        base_counts[i] += 1

    raw_conversations = []
    idx = 1
    for gen_fn, count in zip(base_generators, base_counts):
        for _ in range(count):
            raw_conversations.append(gen_fn(idx))
            idx += 1

    # ── Ambiguous single-visit scenarios (125 conversations) ─────────────────
    ambiguous_plan = [
        (scenario_borderline_creatinine, 42),
        (scenario_tacrolimus_toxicity_vs_rejection, 42),
        (scenario_stable_labs_concerning_symptoms, 41),
    ]
    for gen_fn, count in ambiguous_plan:
        for _ in range(count):
            raw_conversations.append(gen_fn(idx))
            idx += 1

    # ── Longitudinal multi-visit conversations (~150 conversations) ───────────
    # 30 patients × 4-6 visits each; visits linked by longitudinal_patient_id.
    # Conversations are ordered by patient then by visit_number so that a
    # training pipeline can easily group and sort them.
    profiles = make_longitudinal_profiles(seed=999, n=30)
    longitudinal_convs = []
    for profile in profiles:
        for visit_idx in range(len(profile["visits"])):
            longitudinal_convs.append(
                scenario_longitudinal_visit(profile, visit_idx, idx)
            )
            idx += 1

    # Keep longitudinal runs together (sorted by patient then visit) so the
    # JSONL preserves the temporal ordering within each patient's arc.
    longitudinal_convs.sort(
        key=lambda c: (c["longitudinal_patient_id"], c["visit_number"])
    )
    raw_conversations.extend(longitudinal_convs)

    # ── Shuffle non-longitudinal records, preserve longitudinal ordering ──────
    non_long = [c for c in raw_conversations if "longitudinal_patient_id" not in c]
    long_records = [c for c in raw_conversations if "longitudinal_patient_id" in c]
    random.shuffle(non_long)

    # Interleave: non-longitudinal first, then longitudinal block at the end
    # (keeps patient arcs intact for curriculum-style training).
    all_conversations = non_long + long_records

    # Re-index sequentially after shuffle
    for i, conv in enumerate(all_conversations):
        conv["id"] = f"conv_{i + 1:04d}"

    # ── Convert to Gemma format ───────────────────────────────────────────────
    gemma_conversations = [to_gemma_format(c) for c in all_conversations]

    output_dir = Path(__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path = output_dir / "conversations_raw.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(all_conversations, f, indent=2, ensure_ascii=False)

    gemma_path = output_dir / "conversations_gemma_finetune.json"
    with open(gemma_path, "w", encoding="utf-8") as f:
        json.dump(gemma_conversations, f, indent=2, ensure_ascii=False)

    jsonl_path = output_dir / "conversations_gemma_finetune.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for conv in gemma_conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")

    total = len(all_conversations)
    print(f"Generated {total} conversations total.")
    print(f"\nScenario breakdown:")
    scenario_counts = Counter(c["scenario"] for c in all_conversations)
    for scenario, count in sorted(scenario_counts.items()):
        print(f"  {scenario}: {count}")
    print(f"\nLongitudinal patients: {len(profiles)}")
    long_visit_counts = Counter(
        c["longitudinal_patient_id"]
        for c in all_conversations
        if "longitudinal_patient_id" in c
    )
    visit_dist = Counter(long_visit_counts.values())
    for visits, patients in sorted(visit_dist.items()):
        print(f"  {patients} patient(s) with {visits} visits")
    print(f"\nFiles written:")
    print(f"  {raw_path}")
    print(f"  {gemma_path}")
    print(f"  {jsonl_path}")


if __name__ == "__main__":
    main()
