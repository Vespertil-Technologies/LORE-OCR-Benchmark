"""
dataset/gt_generator.py

Generates diverse, realistic ground-truth dicts (gt_structs) for all 3 domains.

Each call to generate_batch(domain, n, seed) returns n distinct gt_structs
with varied names, dates, amounts, policy types, departments, etc.
drawn from realistic Indian-context pools.

No two samples in a batch are identical. Randomness is fully seeded
so the same seed always produces the same batch.

Usage:
    from dataset.gt_generator import generate_batch

    receipts  = generate_batch("receipts",  n=300, seed=42)
    insurance = generate_batch("insurance", n=300, seed=42)
    hospital  = generate_batch("hospital",  n=300, seed=42)
"""

from __future__ import annotations
import random
from datetime import date, timedelta
from typing import Any


# ══════════════════════════════════════════════════════════════════════════════
# SHARED POOLS
# ══════════════════════════════════════════════════════════════════════════════

_FIRST_NAMES = [
    "Aarav", "Aditi", "Ajay", "Akash", "Alok", "Amit", "Ananya", "Anil",
    "Anjali", "Ankit", "Anshul", "Arjun", "Aruna", "Arun", "Asha", "Ashwin",
    "Ayesha", "Deepa", "Deepak", "Divya", "Farhan", "Gautam", "Geeta",
    "Harish", "Harpreet", "Ishaan", "Jaya", "Karan", "Kavita", "Kiran",
    "Komal", "Krishna", "Lakshmi", "Madhur", "Mahesh", "Manish", "Maya",
    "Meera", "Mohammad", "Mohan", "Nandini", "Neha", "Nikhil", "Nisha",
    "Pankaj", "Pooja", "Pradeep", "Prakash", "Pranav", "Priya", "Rahul",
    "Rajesh", "Rajan", "Ranjit", "Rashmi", "Ravi", "Rekha", "Rohit",
    "Rohan", "Sandeep", "Sanjay", "Sara", "Sarika", "Seema", "Shilpa",
    "Shreya", "Shweta", "Smita", "Sneha", "Sonam", "Suresh", "Sushma",
    "Tanvi", "Usha", "Varun", "Vidya", "Vijay", "Vikram", "Vinay",
    "Vinita", "Vivek", "Yamini", "Zara",
]

_LAST_NAMES = [
    "Agarwal", "Ahuja", "Bhat", "Bose", "Chandra", "Chauhan", "Chopra",
    "Das", "Desai", "Deshpande", "Dubey", "Garg", "Ghosh", "Goswami",
    "Gupta", "Iyer", "Jain", "Joshi", "Kapoor", "Kaur", "Khan", "Khanna",
    "Kulkarni", "Kumar", "Malhotra", "Mathur", "Mehta", "Mishra", "Mittal",
    "Mukherjee", "Nair", "Naik", "Nanda", "Narang", "Pande", "Pandey",
    "Patel", "Pillai", "Prasad", "Rao", "Rastogi", "Reddy", "Roy",
    "Saxena", "Shah", "Sharma", "Shetty", "Singh", "Sinha", "Srivastava",
    "Tiwari", "Tripathi", "Varma", "Verma", "Yadav",
]

_CITIES = [
    "Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Chennai", "Kolkata",
    "Pune", "Ahmedabad", "Jaipur", "Lucknow", "Surat", "Kanpur",
    "Nagpur", "Indore", "Bhopal", "Visakhapatnam", "Patna", "Vadodara",
]

_PHONE_PREFIXES = ["98", "97", "96", "95", "94", "93", "91", "89", "88", "87", "86"]


def _random_name(rng: random.Random) -> str:
    return f"{rng.choice(_FIRST_NAMES)} {rng.choice(_LAST_NAMES)}"


def _random_phone(rng: random.Random) -> str | None:
    if rng.random() < 0.35:   # 35% of records have no phone
        return None
    prefix = rng.choice(_PHONE_PREFIXES)
    suffix = "".join(str(rng.randint(0, 9)) for _ in range(8))
    return prefix + suffix


def _random_date(rng: random.Random, start: date, end: date) -> str:
    delta = (end - start).days
    return (start + timedelta(days=rng.randint(0, delta))).isoformat()


def _random_address(rng: random.Random) -> str | None:
    if rng.random() < 0.45:
        return None
    num    = rng.randint(1, 999)
    street = rng.choice([
        "MG Road", "Station Road", "Park Street", "Gandhi Nagar",
        "Nehru Street", "Civil Lines", "Ring Road", "Lake View",
        "Shastri Nagar", "Lal Bagh Road", "Brigade Road",
    ])
    city = rng.choice(_CITIES)
    return f"{num} {street}, {city}"


# ══════════════════════════════════════════════════════════════════════════════
# RECEIPTS GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

_VENDORS = [
    "BigBazaar", "DMart", "Reliance Fresh", "More Supermarket",
    "Spencer's", "Star Bazaar", "Hypercity", "Spar India",
    "Easyday", "Nilgiris", "Nature's Basket", "Food Hall",
    "V-Mart", "Walmart India", "Snapdeal Store", "Paytm Mall",
    "Swiggy Instamart", "Blinkit", "Zepto", "Dunzo Daily",
    "Licious", "FreshMenu", "Zomato Market", "Grofers",
    "Apollo Pharmacy", "MedPlus", "Netmeds Outlet",
    "Crossword Books", "Landmark", "Lifestyle Store",
    "Westside", "Pantaloons", "FBB", "Zudio",
    "Domino's Pizza", "McDonald's", "KFC", "Burger King",
    "Pizza Hut", "Subway", "Cafe Coffee Day", "Starbucks India",
    "Chai Point", "Haldiram's", "Bikanervala",
]

_PAYMENT_METHODS = [
    "UPI", "Cash", "Debit Card", "Credit Card",
    "Net Banking", "PhonePe", "Google Pay", "Paytm",
    "Wallets", "EMI",
]

_LINE_ITEM_POOLS = [
    ["1x Bread 45.00", "2x Milk 90.00", "1x Eggs 55.00"],
    ["3x Biscuits 90.00", "1x Juice 85.00", "2x Soap 120.00"],
    ["1x Rice 5kg 320.00", "1x Dal 200g 65.00", "1x Oil 1L 185.00"],
    ["Chicken 500g 280.00", "Paneer 200g 110.00", "Curd 400ml 55.00"],
    ["Shampoo 340.00", "Toothpaste 85.00", "Face wash 220.00"],
    ["Burger 149.00", "Fries 99.00", "Coke 69.00"],
    ["Pizza Margherita 299.00", "Garlic Bread 129.00", "Pepsi 65.00"],
    ["T-Shirt 599.00", "Jeans 1499.00", "Belt 299.00"],
    ["Medicine x3 450.00", "Vitamins 320.00"],
    ["Coffee 180.00", "Sandwich 220.00", "Muffin 120.00"],
    ["Book 399.00", "Notebook 120.00", "Pen set 85.00"],
]


def _generate_receipt(rng: random.Random, index: int) -> dict:
    vendor   = rng.choice(_VENDORS)
    year     = rng.randint(2022, 2024)
    month    = rng.randint(1, 12)
    day      = rng.randint(1, 28)
    dt       = date(year, month, day).isoformat()
    time_str = f"{rng.randint(8,21):02d}:{rng.choice(['00','15','30','45'])}"

    receipt_no = (
        f"RCT-{year}{month:02d}{day:02d}-"
        f"{rng.randint(1, 999):03d}"
    )

    items    = rng.choice(_LINE_ITEM_POOLS)
    subtotal = round(rng.uniform(80, 4500), 2)
    tax_rate = rng.choice([0.05, 0.12, 0.18])
    tax      = round(subtotal * tax_rate, 2)
    total    = round(subtotal + tax, 2)

    return {
        "vendor_name":    vendor,
        "date":           dt,
        "time":           time_str if rng.random() > 0.2 else None,
        "receipt_number": receipt_no if rng.random() > 0.15 else None,
        "subtotal":       subtotal   if rng.random() > 0.2  else None,
        "tax_amount":     tax        if rng.random() > 0.2  else None,
        "total_amount":   total,
        "currency":       "INR",
        "payment_method": rng.choice(_PAYMENT_METHODS) if rng.random() > 0.1 else None,
        "line_items":     ", ".join(items) if rng.random() > 0.25 else None,
    }


# ══════════════════════════════════════════════════════════════════════════════
# INSURANCE GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

_POLICY_TYPES = [
    "Term Life", "Whole Life", "Endowment", "ULIP",
    "Health Insurance", "Critical Illness", "Personal Accident",
    "Motor Comprehensive", "Motor Third Party", "Home Insurance",
    "Travel Insurance", "Group Health",
]

_INSURERS = [
    "LIC", "HDFC Life", "ICICI Prudential", "SBI Life",
    "Max Life", "Bajaj Allianz", "Kotak Life", "Tata AIA",
    "Aditya Birla Sun Life", "Reliance Nippon",
    "Star Health", "Niva Bupa", "Care Health", "ManipalCigna",
    "New India Assurance", "Oriental Insurance", "United India",
    "National Insurance", "IFFCO Tokio",
]

_AGENT_PREFIXES = [
    "AG", "AGT", "LIC", "HDF", "ICR", "SBI", "MX",
    "BJA", "KTK", "TAT",
]

_PAYMENT_FREQS = ["yearly", "monthly", "quarterly", "half-yearly"]


def _generate_insurance(rng: random.Random, index: int) -> dict:
    name    = _random_name(rng)
    gender  = rng.choice(["Male", "Female"])
    dob     = _random_date(rng, date(1955, 1, 1), date(2005, 12, 31))
    phone   = _random_phone(rng)
    address = _random_address(rng)

    policy_no   = (
        f"{rng.choice(_AGENT_PREFIXES)}-"
        f"{rng.randint(100000, 999999)}"
    )
    policy_type = rng.choice(_POLICY_TYPES)
    start       = _random_date(rng, date(2018, 1, 1), date(2024, 6, 30))
    has_end     = rng.random() > 0.4
    end_date    = None
    if has_end:
        start_d  = date.fromisoformat(start)
        duration = rng.choice([5, 10, 15, 20, 25, 30])
        end_date = date(start_d.year + duration, start_d.month, start_d.day).isoformat()

    # Fully random amount - not from a fixed pool a model could memorise
    amount    = round(rng.uniform(300, 30000), 2)
    freq      = rng.choice(_PAYMENT_FREQS)

    agent_name = _random_name(rng)
    agent_id   = (
        f"{rng.choice(_AGENT_PREFIXES)}-"
        f"{rng.randint(100, 999)}"
    )

    return {
        "policyholder": {
            "name":           name,
            "dob":            dob,
            "gender":         gender,
            "contact_number": phone,
            "address":        address,
        },
        "policy": {
            "policy_number": policy_no,
            "policy_type":   policy_type,
            "start_date":    start,
            "end_date":      end_date,
        },
        "premium": {
            "amount":            float(amount),
            "currency":          "INR",
            "payment_frequency": freq,
        },
        "agent": {
            "name":     agent_name if rng.random() > 0.1 else None,
            "agent_id": agent_id   if rng.random() > 0.1 else None,
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# HOSPITAL GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

_DEPARTMENTS = [
    "General Medicine", "Cardiology", "Orthopaedics", "Gynaecology",
    "Paediatrics", "Neurology", "Dermatology", "ENT",
    "Ophthalmology", "Gastroenterology", "Oncology", "Urology",
    "Nephrology", "Endocrinology", "Pulmonology", "Psychiatry",
    "Emergency", "Radiology", "Physiotherapy",
]

# Complaints are generated procedurally - not from a fixed pool.
# This prevents models from guessing values they've seen in training.
_COMPLAINT_SYMPTOMS = [
    "fever", "chest pain", "headache", "abdominal pain", "back pain",
    "cough", "knee pain", "skin rash", "blurred vision", "ear pain",
    "fatigue", "shoulder pain", "ankle pain", "wrist pain", "neck pain",
    "joint pain", "muscle weakness", "breathlessness", "palpitations",
    "nausea", "vomiting", "diarrhoea", "constipation", "bloating",
    "burning sensation", "numbness", "tingling", "hair loss", "dry skin",
    "swelling", "bruising", "redness", "itching", "discharge",
]
_COMPLAINT_QUALIFIERS = [
    "for {n} days", "for {n} weeks", "since {n} days", "since morning",
    "since yesterday", "on and off for {n} days", "worsening over {n} days",
    "recurring for {n} months", "post exertion", "at night", "after meals",
    "radiating to left arm", "radiating to right leg", "with mild fever",
    "with nausea", "with dizziness", "with vomiting", "with swelling",
    "with difficulty walking", "with loss of appetite",
]
_COMPLAINT_PURPOSES = [
    "Routine annual checkup",
    "Follow-up after discharge",
    "Pre-operative evaluation",
    "Post-operative review",
    "Vaccination and growth review",
    "Diabetic management review",
    "Hypertension monitoring",
    "Antenatal checkup",
    "Cancer screening",
    "Second opinion consultation",
]

def _random_complaint(rng: random.Random) -> str:
    """Generate a unique procedural complaint - not drawn from a fixed pool."""
    # 20% chance of a specific purpose visit
    if rng.random() < 0.20:
        return rng.choice(_COMPLAINT_PURPOSES)
    symptom   = rng.choice(_COMPLAINT_SYMPTOMS)
    qualifier = rng.choice(_COMPLAINT_QUALIFIERS).format(n=rng.randint(1, 14))
    # Capitalise and punctuate naturally
    return f"{symptom.capitalize()} {qualifier}"

_BLOOD_GROUPS = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]

_HOSPITALS = [
    "Apollo", "Fortis", "Max", "Manipal", "Aster", "Narayana",
    "Medanta", "Artemis", "Kokilaben", "Lilavati", "Hinduja",
    "Wockhardt", "Ruby Hall", "Sahyadri", "Care",
]

_INSURER_SHORT = [
    "StarHealth", "NivaBupa", "CareHealth", "AyushmanBharat",
    "HDFC Ergo", "ICICI Lombard", "Bajaj Allianz Health",
    "United India Health", "Government CGHS", "ESI",
]

_DOCTOR_TITLES = ["Dr", "Dr.", "Prof Dr"]
_DOCTOR_SPECIALTIES = ["", "MD", "MS", "MBBS", "DM", "MCh"]

_SBP_RANGE = list(range(100, 170, 5))   # systolic
_DBP_RANGE = list(range(60, 105, 5))    # diastolic
_PULSE_RANGE = list(range(55, 115, 1))
_TEMP_VARIANTS = [
    lambda t: f"{t}F",
    lambda t: f"{t} F",
    lambda t: f"{round((t - 32) * 5/9, 1)}C",
    lambda t: f"{round((t - 32) * 5/9, 1)} C",
]


def _generate_hospital(rng: random.Random, index: int) -> dict:
    name    = _random_name(rng)
    gender  = rng.choice(["Male", "Female"])
    dob     = _random_date(rng, date(1940, 1, 1), date(2023, 12, 31))
    blood   = rng.choice(_BLOOD_GROUPS)   if rng.random() > 0.15 else None
    phone   = _random_phone(rng)
    address = _random_address(rng)

    visit_date = _random_date(rng, date(2021, 1, 1), date(2024, 12, 31))
    visit_time = f"{rng.randint(7,20):02d}:{rng.choice(['00','10','15','20','30','45'])}"
    dept       = rng.choice(_DEPARTMENTS)
    complaint  = _random_complaint(rng)

    sbp  = rng.choice(_SBP_RANGE)
    dbp  = rng.choice(_DBP_RANGE)
    bp   = f"{sbp}/{dbp}" if rng.random() > 0.1 else None

    pulse = str(rng.choice(_PULSE_RANGE)) if rng.random() > 0.1 else None

    raw_temp  = round(rng.uniform(97.0, 103.5), 1)
    temp_fmt  = rng.choice(_TEMP_VARIANTS)
    temp      = temp_fmt(raw_temp)           if rng.random() > 0.15 else None

    weight = f"{rng.randint(38, 130)} kg"    if rng.random() > 0.4 else None
    height = f"{rng.randint(140, 195)} cm"   if rng.random() > 0.4 else None

    has_ins   = rng.random() > 0.3
    ins_prov  = rng.choice(_INSURER_SHORT) if has_ins else None
    ins_no    = (
        f"INS-{rng.randint(1000, 9999)}-"
        f"{rng.randint(100000, 999999)}"
        if has_ins else None
    )

    doc_first  = _random_name(rng).split()[0]
    doc_last   = _random_name(rng).split()[-1]
    doc_title  = rng.choice(_DOCTOR_TITLES)
    doc_suffix = rng.choice(_DOCTOR_SPECIALTIES)
    doc_name   = f"{doc_title} {doc_first} {doc_last}"
    if doc_suffix:
        doc_name += f" {doc_suffix}"
    doc_id = (
        f"{doc_last[:2].upper()}"
        f"{rng.choice(['Apollo','Fortis','Max','Manipal','Aster','KH','CH','GH'])[:2].upper()}"
        f"-{rng.randint(1000, 9999)}"
    )

    return {
        "patient": {
            "name":           name,
            "dob":            dob,
            "gender":         gender,
            "blood_group":    blood,
            "contact_number": phone,
            "address":        address,
        },
        "visit": {
            "date":             visit_date,
            "time":             visit_time if rng.random() > 0.15 else None,
            "department":       dept       if rng.random() > 0.05 else None,
            "reason_for_visit": complaint,
        },
        "vitals": {
            "blood_pressure": bp,
            "pulse":          pulse,
            "temperature":    temp,
            "weight":         weight,
            "height":         height,
        },
        "insurance": {
            "provider":      ins_prov,
            "policy_number": ins_no,
        },
        "attending_physician": {
            "name": doc_name if rng.random() > 0.05 else None,
            "id":   doc_id   if rng.random() > 0.1  else None,
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# DISPATCH
# ══════════════════════════════════════════════════════════════════════════════

_GENERATORS = {
    "receipts":  _generate_receipt,
    "insurance": _generate_insurance,
    "hospital":  _generate_hospital,
}


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def generate_batch(
    domain: str,
    n:      int,
    seed:   int = 42,
) -> list[dict]:
    """
    Generate n distinct gt_structs for the given domain.

    Args:
        domain: 'receipts', 'insurance', or 'hospital'
        n:      Number of records to generate
        seed:   Base random seed - same seed always produces the same batch

    Returns:
        List of n gt_struct dicts, each distinct
    """
    if domain not in _GENERATORS:
        raise ValueError(f"Unknown domain '{domain}'. Must be one of: {list(_GENERATORS)}")

    rng = random.Random(seed)
    gen = _GENERATORS[domain]

    return [gen(rng, i) for i in range(n)]


# ══════════════════════════════════════════════════════════════════════════════
# Usage example
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json

    for domain in ["receipts", "insurance", "hospital"]:
        batch = generate_batch(domain, n=5, seed=42)
        print(f"\n{'='*60}")
        print(f"Domain: {domain.upper()}  (5 samples)")
        print('='*60)
        for i, gt in enumerate(batch):
            print(f"\n--- Sample {i+1} ---")
            print(json.dumps(gt, indent=2))

    # Diversity check - make sure samples differ
    print(f"\n{'='*60}")
    print("Diversity check: 300 receipts - unique vendor_names")
    receipts = generate_batch("receipts", n=300, seed=42)
    vendors  = {r["vendor_name"] for r in receipts}
    totals   = {r["total_amount"] for r in receipts}
    print(f"  Unique vendors : {len(vendors)} / {len(_VENDORS)}")
    print(f"  Unique totals  : {len(totals)}")

    print("\n300 insurance - unique policy numbers and types")
    insurance = generate_batch("insurance", n=300, seed=42)
    polnos    = {r["policy"]["policy_number"] for r in insurance}
    poltypes  = {r["policy"]["policy_type"] for r in insurance}
    print(f"  Unique policy numbers : {len(polnos)}")
    print(f"  Unique policy types   : {len(poltypes)} / {len(_POLICY_TYPES)}")

    print("\n300 hospital - unique complaints and departments")
    hospital = generate_batch("hospital", n=300, seed=42)
    comps    = {r["visit"]["reason_for_visit"] for r in hospital}
    depts    = {r["visit"]["department"] for r in hospital}
    print(f"  Unique complaints   : {len(comps)}")
    print(f"  Unique departments  : {len(depts)} / {len(_DEPARTMENTS)}")