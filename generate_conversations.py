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
    random.seed(42)
    total = 1000
    # Distribute evenly across 4 scenarios
    per_scenario = total // 4
    remainder = total % 4

    generators = [
        scenario_elevated_creatinine,
        scenario_followup_appointment,
        scenario_emotional_distress,
        scenario_lifestyle_changes,
    ]

    counts = [per_scenario] * 4
    for i in range(remainder):
        counts[i] += 1

    raw_conversations = []
    idx = 1
    for gen_fn, count in zip(generators, counts):
        for _ in range(count):
            conv = gen_fn(idx)
            raw_conversations.append(conv)
            idx += 1

    random.shuffle(raw_conversations)

    # Re-index after shuffle
    for i, conv in enumerate(raw_conversations):
        conv["id"] = f"conv_{i + 1:04d}"

    # Convert to Gemma format
    gemma_conversations = [to_gemma_format(c) for c in raw_conversations]

    output_dir = Path(__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save raw (human-readable) version
    raw_path = output_dir / "conversations_raw.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(raw_conversations, f, indent=2, ensure_ascii=False)

    # Save Gemma fine-tuning version
    gemma_path = output_dir / "conversations_gemma_finetune.json"
    with open(gemma_path, "w", encoding="utf-8") as f:
        json.dump(gemma_conversations, f, indent=2, ensure_ascii=False)

    # Also save as JSONL (one record per line) — common for fine-tuning pipelines
    jsonl_path = output_dir / "conversations_gemma_finetune.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for conv in gemma_conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")

    print(f"Generated {total} conversations.")
    print(f"Scenario breakdown:")
    from collections import Counter
    scenario_counts = Counter(c["scenario"] for c in raw_conversations)
    for scenario, count in sorted(scenario_counts.items()):
        print(f"  {scenario}: {count}")
    print(f"\nFiles written:")
    print(f"  {raw_path}")
    print(f"  {gemma_path}")
    print(f"  {jsonl_path}")


if __name__ == "__main__":
    main()
