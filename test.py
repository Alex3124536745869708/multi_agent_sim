import json
import re
from pprint import pprint
from datetime import datetime

text = """<|channel|>analysis<|message|>We have a patient. Need to decide which diagnostic pathway(s) to run. The features: age 51, sex=0 (0 likely female). cp=0 (chest pain type?) etc. So we need to evaluate risk for diabetes, heart disease, breast cancer.

We must provide JSON with routes: list of diagnostic pathways. "primary" route: the most likely diagnostic pathway. Provide confidence scores for each route. Also notes.

Need to consider features: diabetes: features include Age, Sex, Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function. In dataset, thresholds? We can approximate: high glucose (180). BMI 59.4 huge. diabetic pedigree 2.42. The features strongly indicate diabetes. So diagnostic pathway for diabetes must be run.

{
  "routes": ["diabetes", "heart", "breast_cancer"],
  "primary": "diabetes",
  "confidence": {
    "diabetes": 0.72,
    "heart": 0.46,
    "breast_cancer": 0.21
  },
  "notes": "Patient exhibits classic red‑flag signs for diabetes (age 51, BMI 59.4, fasting glucose 180, high diabetes pedigree function, insulin 14). Chest pain type 0 with exercise‑induced angina, elevated total cholesterol (305) and resting blood pressure (130) raise moderate risk for coronary heart disease, warranting a heart panel. Breast‑cancer risk is lower but not negligible given age 51, female sex, and borderline tumoral features on the radiomics panel; a diagnostic pathway may help rule out malignancy."
}

Heart disease: chest pain features. Age 51, cp 0, trestbps 130, chol 305, fbs0? fbs=0? maybe normal. rest ecg 1, thalach 142, exang 1, oldpeak 1.2, slope1, ca 0. So risk moderate? chest pain, exang=1 (exercise-induced angina). For heart disease model, I'd say moderate to high risk. Not as high as diabetes but still relevant. So "heart" maybe as second.

Breast cancer: features: mean radius 12.270, texture 17.920, perimeter 78.410, area 466.100, smoothness 0.087. For typical dataset (Wisconsin Diagnostic Breast Cancer) mean radius > ~10 indicates malignant? Actually borderline. Usually malignant tumors have larger radius, smoother lower? 0.087 is relatively low smoothness (smoothness measures mean smoothness). Lower might indicate malignant? Not sure. Typical benign mean radius around 10-12; malignant > 14 or so. 12.27 might be borderline but could be benign. So risk moderate but less. So probably not primary.

So we output routes likely: ["diabetes", "heart"]. If we include breast cancer maybe to be thorough? But triage assistant: decide which diagnostic pathways. Might include all if uncertain.

But we could also include breast cancer? The patient age 51 female is at risk for breast cancer. Features are borderline risk. Provide confidence scores: Diabetes high (~0.8), Heart moderate (~0.6), breast cancer lower (~0.3). But we need normalization to sum 1? Not necessary but we can. They had example with equal 0.33 each. So not required. Provide numeric probabilities.

Return JSON.

Also "primary": choose the one with highest confidence. So "diabetes" maybe.

Let's compute approximate confidence: Diabetes nearly certain: 0.8, Heart 0.5, Breast cancer 0.3. But sums to >1. But they didn't ask to sum, but maybe it's fine.

But we could do "diabetes": 0.7, "heart":0.6, "breast_cancer":0.2. That's also fine. Provide notes: mention features.

Ok produce JSON.<|end|><|start|>assistant<|channel|>final<|message|>{
  "routes": ["diabetes", "heart", "breast_cancer"],
  "primary": "diabetes",
  "confidence": {
    "diabetes": 0.72,
    "heart": 0.46,
    "breast_cancer": 0.21
  },
  "notes": "Patient exhibits classic red‑flag signs for diabetes (age 51, BMI 59.4, fasting glucose 180, high diabetes pedigree function, insulin 14). Chest pain type 0 with exercise‑induced angina, elevated total cholesterol (305) and resting blood pressure (130) raise moderate risk for coronary heart disease, warranting a heart panel. Breast‑cancer risk is lower but not negligible given age 51, female sex, and borderline tumoral features on the radiomics panel; a diagnostic pathway may help rule out malignancy."
}<|return|>"""

def find_last_json(text):
    text_len = len(text)
    first = True
    end = 0
    start = 0
    counter = 0
    for i in range(text_len-1, -1, -1):
        if counter == 0 and not first:
            start = i+1
            break
        if text[i] == "}":
            counter += 1
            if first:
                first = False
                end = i+1
        if text[i] == "{":
            if first:
                continue
            counter -= 1

    return text[start:end]

json_text = find_last_json(text)

def find_key_words(json_text):
    return list(re.finditer(r"routes?|primary|confidence|notes?", json_text, re.IGNORECASE))

def parse_json_text(json_text):
    key_words = find_key_words(json_text)
    for idx in range(len(key_words)):
        next_idx = key_words[idx+1].start() if idx+1 < len(key_words) else len(json_text)
        curr = key_words[idx]

        preprocess_domains = lambda x: x.replace(" ", "_").replace("-", "_")
        preprocess_numbers = lambda x: float(x.replace(",", ".") if x.lower() != "none" else 0)

        match curr.group():
            case "routes" | "route":
                between_text = json_text[curr.end()+1:next_idx].strip()
                routes = re.findall(r"diabetes|heart|breast[_ -]cancer", between_text, re.IGNORECASE)
                routes = list(map(preprocess_domains, routes))
            case "primary":
                between_text = json_text[curr.end()+1:next_idx].strip()
                primary_group = re.search(r"diabetes|heart|breast[_ -]cancer", between_text, re.IGNORECASE)
                primary = preprocess_domains(primary_group.group()) if primary_group else None
            case "confidence":
                between_text = json_text[curr.end()+1:next_idx].strip()
                confidence_keys = re.findall(r"diabetes|heart|breast[_ -]cancer", between_text, re.IGNORECASE)
                confidence_values = re.findall(r"\d+[.,]?\d*|none", between_text, re.IGNORECASE)
                confidence = dict(zip(
                    map(preprocess_domains, confidence_keys),
                    map(preprocess_numbers, confidence_values)
                ))
            case "notes" | "note":
                between_text = json_text[curr.end()+1:next_idx].strip()
                notes_match = re.search(r'".*"', between_text, re.DOTALL)
                notes = notes_match.group().strip('"') if notes_match else None

    return {"routes": routes, "primary": primary, "confidence": confidence, "notes": notes}

json_dict = parse_json_text(json_text)

print(json_dict)

print(max(json_dict["confidence"], key=json_dict["confidence"].get))