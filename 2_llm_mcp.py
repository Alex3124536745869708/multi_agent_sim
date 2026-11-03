import pandas as pd
import numpy as np
import uuid
import warnings
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from datetime import datetime
from pathlib import Path
import json
import re
from ucimlrepo import fetch_ucirepo
from transformers import AutoModelForCausalLM, AutoTokenizer

# init logging
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"run_{timestamp}.txt"
def log(input: str, output: str, parsed_output: dict, verdict: str):
    """Log the original inputm, output, parsed output, and verdict of output."""
    with log_file.open("a", encoding="utf-8") as f:
        f.write("=====START=====\n"+ "- INPUT:\n" + input + "\n- OUTPUT:\n" + output + "\n- PARSED OUTPUT:\n" + str(parsed_output) + "\n- VERDICT:\n" + verdict + "\n=====END=====\n")
    return

# init LLM
model_id = "openai/gpt-oss-20b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
LLM = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="cuda",
)

warnings.filterwarnings("ignore")

def call_llm_fn(prompt: str) -> str:
    """Call LLM and return raw text output.
    
    Example output: return '''{
          "routes": ["diabetes", "heart", "parkinson"],
          "primary": "heart",
          "confidence": {"diabetes": 0.35, "heart": 0.72, "breast_cancer": 0.05, "parkinson": 0.20},
          "notes": "Brief triage rationale."
        }'''
    """
    
    messages = [
        {"role": "user", "content": prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        reasoning_effort="medium",
    ).to(LLM.device)
    generated = LLM.generate(**inputs, max_new_tokens=5000)
    return tokenizer.decode(generated[0][inputs["input_ids"].shape[-1]:])

def make_case_text(pid, d_row, b_row, h_row, p_row):
    # d_row, b_row, h_row are 1D arrays (same as you pass to models)
    # Keep it short; it’s triage, not a full report.
    return f"""
Patient {pid}. You are a triage assistant.
You must decide which diagnostic pathway(s) to run: diabetes, breast_cancer or heart.

Patient features:
age={h_row[0]}, sex={h_row[1]}, cp={h_row[2]}, trestbps={h_row[3]}, chol={h_row[4]},
fbs={h_row[5]}, rest ecg={h_row[6]}, thalach={h_row[7]}, exang={h_row[8]}, oldpeak={h_row[9]},
slope={h_row[10]}, ca={h_row[11]},
Pregnancies={d_row[0]}, Glucose={d_row[1]}, Blood Pressure={d_row[2]}, Skin Thickness={d_row[3]},
Insulin={d_row[4]}, BMI={d_row[5]}, Diabetes Pedigree Function={d_row[6]}.
Breast features: mean radius={b_row[0]:.3f}, mean texture={b_row[1]:.3f}, mean perimeter={b_row[2]:.3f}, mean area={b_row[3]:.3f}, mean smoothness={b_row[4]:.3f}.
Movement features: 
MDVP:Fo={p_row[0]:.3f}, MDVP:Fhi={p_row[1]:.3f}, MDVP:Flo={p_row[2]:.3f},
MDVP:Jitter={p_row[3]:.3f}, MDVP:Jitter={p_row[4]:.3f},
MDVP:RAP={p_row[5]:.3f}, MDVP:PPQ={p_row[6]:.3f}, Jitter:DDP={p_row[7]:.3f},
MDVP:Shimmer={p_row[8]:.3f}, MDVP:Shimmer={p_row[9]:.3f},
Shimmer:APQ3={p_row[10]:.3f}, Shimmer:APQ5={p_row[11]:.3f},
MDVP:APQ={p_row[12]:.3f}, Shimmer:DDA={p_row[13]:.3f},
NHR={p_row[14]:.3f}, HNR={p_row[15]:.3f}, RPDE={p_row[16]:.3f},
DFA={p_row[17]:.3f}, spread1={p_row[18]:.3f}, spread2={p_row[19]:.3f},
D2={p_row[20]:.3f}, PPE={p_row[21]:.3f}.

Output JSON only.
Output format (example):
{{
  "routes": ["diabetes", "heart"],
  "primary": "heart",
  "confidence": {{"diabetes": 0.35, "heart": 0.72, "breast_cancer": 0.05, "parkinson": 0.20}},
  "notes": "Brief triage rationale."
}}
""".strip()


# ==========================================
# STEP 1: Define Custom Scheduler (like RandomActivation)
# ==========================================

class CustomScheduler:
    def __init__(self, model):
        self.model = model
        self.agents = []

    def add(self, agent):
        self.agents.append(agent)

    def step(self):
        for agent in self.agents:
            agent.step()

# ==========================================
# STEP 2: Define Agent Classes
# ==========================================

class BaseAgent:
    def __init__(self, unique_id, model):
        self.unique_id = unique_id
        self.model = model

class LLMPlannerAgent(BaseAgent):
    """
    An agent that uses an LLM to triage cases into diagnostic pathways.
    THIS IS WHERE THE AGENT THE ML MODEL CALLED FOR THE PREDICTED DOMAIN(S).
    TODO: implement the step method for this class yourself.
    """
    def __init__(self, unique_id, model, call_llm_fn):
        super().__init__(unique_id, model)
        self.role = "llm_planner"
        self.call_llm = call_llm_fn

    @staticmethod
    def _extract_json(text):
        # robust-ish: find biggest {...} block
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            print("No json!")
            raise ValueError("No JSON object found in LLM output.")
        return json.loads(m.group(0))

    @staticmethod
    def _extract_last_json(text):
        # robust-ish: find last {...} block if multiple json objects are present
        text_len = len(text)
        first = True
        end = 0
        start = 0
        counter = 0
        # Scan backwards and count '}' and '{' with a counter to determine start and end of json object
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

    @staticmethod
    def _find_key_words(json_text):
        """finds the keys in the json text and returns a list of match objects"""
        return list(re.finditer(r"routes?|primary|confidence|notes?", json_text, re.IGNORECASE))

    def _parse_json_text(self, json_text):
        """parse the json text into a dict with keys: routes, primary, confidence, notes"""
        key_words = self._find_key_words(json_text)
        for idx in range(len(key_words)):
            next_idx = key_words[idx+1].start() if idx+1 < len(key_words) else len(json_text)
            curr = key_words[idx]

            # preprocesses possible variations in keys and values to get a consistent output
            preprocess_domains = lambda x: x.replace(" ", "_").replace("-", "_")
            preprocess_numbers = lambda x: float(x.replace(",", ".") if x.lower() != "none" else 0)
    
            match curr.group():
                case "routes" | "route":
                    # create a list with the possible domains
                    between_text = json_text[curr.end()+1:next_idx].strip()
                    routes = re.findall(r"diabetes|heart|breast[_ -]cancer|parkinson", between_text, re.IGNORECASE)
                    routes = list(map(preprocess_domains, routes))
                case "primary":
                    # find the first domain mentioned after the primary key
                    between_text = json_text[curr.end()+1:next_idx].strip()
                    primary_group = re.search(r"diabetes|heart|breast[_ -]cancer|parkinson", between_text, re.IGNORECASE)
                    primary = preprocess_domains(primary_group.group()) if primary_group else None
                case "confidence":
                    # find all domains and their corresponding confidence values
                    between_text = json_text[curr.end()+1:next_idx].strip()
                    confidence_keys = re.findall(r"diabetes|heart|breast[_ -]cancer|parkinson", between_text, re.IGNORECASE)
                    confidence_values = re.findall(r"\d+[.,]?\d*|none", between_text, re.IGNORECASE)
                    confidence = dict(zip(
                        map(preprocess_domains, confidence_keys),
                        map(preprocess_numbers, confidence_values)
                    ))
                case "notes" | "note":
                    # find the string text after the notes key
                    between_text = json_text[curr.end()+1:next_idx].strip()
                    notes_match = re.search(r'".*"', between_text, re.DOTALL)
                    notes = notes_match.group().strip('"') if notes_match else None
    
        return {"routes": routes, "primary": primary, "confidence": confidence, "notes": notes}

    def step(self):
        if not self.model.case_queue:
            print("No queued cases to triage.")
            return

        remaining = []
        for case in self.model.case_queue:
            pid, payload = case  # payload is dict with data per domain + text
            prompt = payload["text"]

            # Call LLM and parse response, LLM has 3 tries per case to give a valid answer
            reps = 0
            max_reps = 3
            while reps < max_reps:
                reps += 1
                response = self.call_llm(prompt)                 # get response from LLM
                response_ = self._extract_last_json(response)    # get last {...} block
                response_dict = self._parse_json_text(response_) # parse the text into a dict

                verdict = "" # init verdict

                # check confidence
                # if no confidence: prompt again!
                if len(response_dict["confidence"]) == 0:
                    if reps < max_reps:
                        verdict = "Missing confidence values. Prompting again."
                    else:
                        verdict = "Missing confidence values. Max retries reached. Lets diagnose everything, better safe then sorry."
                        response_dict["routes"] = ["diabetes", "heart", "breast_cancer", "parkinson"]
                        response_dict["primary"] = response_dict["primary"] if response_dict["primary"] else "diabetes"
                    log(input=prompt, output=response, parsed_output=response_dict, verdict=verdict)
                    continue

                # check and fix primary
                # if no primary, get highest confidence as primary
                if response_dict["primary"] is None:
                    response_dict["primary"] = max(response_dict["confidence"], key=response_dict["confidence"].get)
                    verdict = "Missing primary. Set to highest confidence domain."

                # check and fix routes
                # if no routes, get domain above certain threshold (50%), if still no route: get primary as route
                if len(response_dict["routes"]) == 0:
                    response_dict["routes"] = [domain for domain, conf in response_dict["confidence"].items() if conf >= 0.5]
                    if len(response_dict["routes"]) == 0:
                        response_dict["routes"] = [response_dict["primary"]]
                        verdict += "\nMissing routes. Set to primary."
                    else:
                        verdict += "\nMissing routes. Set to domains with confidence >= 0.5."

                if verdict == "":
                    verdict = "LLM output parsed successfully."
                    
                log(input=prompt, output=response, parsed_output=response_dict, verdict=verdict)
                
                break
                
            # store triage decision in model context
            self.model.triage_log.append({
                    "PatientID": pid,
                    "routes": response_dict["routes"],
                    "primary": response_dict["primary"],
                    "labels": payload["labels"]
                })

            # Route to appropriate executors based on LLM decision
            # add to model.context with correct task_{domain}_{pid} name
            for domain in response_dict["routes"]:
                task_id = f"task_{domain}_{pid}"
                task = {"id": pid, "description": f"Analyze {domain} data", "data": payload["data"][domain], "y_true": payload["labels"][domain]}
                self.model.context[task_id] = task
                print(f"[{self.role}] Planned task {task_id}")

        self.model.case_queue = remaining

class PlannerAgent(BaseAgent):
    def __init__(self, unique_id, model, disease):
        super().__init__(unique_id, model)
        self.disease = disease
        self.role = f"planner_{disease}"

    def step(self):
        if self.model.case_queue:
            remaining = []
            for case in self.model.case_queue:
                pid, data, domain = case
                if domain == self.disease:
                    task_id = f"task_{self.disease}_{pid}"
                    task = {"id": pid, "description": f"Analyze {self.disease} data", "data": data}
                    self.model.context[task_id] = task
                    print(f"[{self.role}] Planned task {task_id}")
                else:
                    remaining.append(case)
            self.model.case_queue = remaining

class ExecutorAgent(BaseAgent):
    def __init__(self, unique_id, model, disease, clf):
        super().__init__(unique_id, model)
        self.disease = disease
        self.role = f"executor_{disease}"
        self.clf = clf

    def step(self):
        for key in list(self.model.context.keys()):
            if key.startswith(f"task_{self.disease}"):
                task = self.model.context.pop(key)
                pred = self.clf.predict([task["data"]])[0]
                result_id = f"result_{self.disease}_{task['id']}"
                self.model.context[result_id] = {
                    "task_id": task["id"],
                    "prediction": int(pred),
                    "agent": self.role
                }
                y_true = int(task.get("y_true", -1))
                y_pred = int(self.clf.predict([task["data"]])[0])

                self.model.results.append({
                    "PatientID": task["id"],
                    "Disease": self.disease,
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "triage_primary": task.get("triage", {}).get("primary", None)
                })
                print(f"[{self.role}] Executed task {task['id']} → prediction: {pred}")

# ==========================================
# STEP 3: Define the Mesa Model
# ==========================================

class MCPMesaModel:
    def __init__(self, X_d, X_b, X_h, X_p, clf_d, clf_b, clf_h, clf_p):
        self.schedule = CustomScheduler(self)
        self.context = {}
        self.case_queue = []
        self.results = []
        self.triage_log = []

        # Add cases to queue (patients)
        for _ in range(100):
            pid = str(uuid.uuid4().int)[:4]
            idx_d = X_d.sample(1).index[0]
            d_row = X_d.loc[idx_d].values
            y_d_true = int(y_d.loc[idx_d])

            idx_b = X_b.sample(1).index[0]
            b_row = X_b.loc[idx_b].values
            y_b_true = int(y_b[idx_b]) 

            idx_h = X_h.sample(1).index[0]
            h_row = X_h.loc[idx_h].values
            y_h_true = int(y_h.loc[idx_h])

            idx_p = X_p.sample(1).index[0]
            p_row = X_p.loc[idx_p].values
            y_p_true = int(y_p.loc[idx_p])

            payload = {
                "text": make_case_text(pid, d_row, b_row, h_row, p_row),
                "data": {
                    "diabetes": d_row,
                    "breast_cancer": b_row,
                    "heart": h_row,
                    "parkinson": p_row
                },
                "labels": {
                    "diabetes": y_d_true,
                    "breast_cancer": y_b_true,
                    "heart": y_h_true,
                    "parkinson": y_p_true
                }
            }
            
            self.case_queue.append((pid, payload))

        # Add agents
        # Replaces 3 planners with 1 LLM planner
        self.schedule.add(LLMPlannerAgent("planner_llm", self, call_llm_fn))

        self.schedule.add(ExecutorAgent("executor_diabetes", self, "diabetes", clf_d))
        self.schedule.add(ExecutorAgent("executor_breast_cancer", self, "breast_cancer", clf_b))
        self.schedule.add(ExecutorAgent("executor_heart", self, "heart", clf_h))
        self.schedule.add(ExecutorAgent("executor_parkinson", self, "parkinson", clf_p))

    def step(self):
        self.schedule.step()


# ==========================================
# STEP 4: Run Simulation
# ==========================================

# Diabetes
diabetes = pd.read_csv("data/pima-indians-diabetes.data.csv",
                       names=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI",
                              "DiabetesPedigreeFunction", "Age", "Outcome"])
X_d, y_d = diabetes.drop("Outcome", axis=1), diabetes["Outcome"]

# Breast Cancer
breast = load_breast_cancer()
X_b = pd.DataFrame(breast.data, columns=breast.feature_names)
y_b = breast.target  # 0 = malignant, 1 = benign

# Heart
heart = pd.read_csv("data/heart.csv")
X_h, y_h = heart.drop("target", axis=1), heart["target"]

# Parkinson
parkinsons = fetch_ucirepo(id=174)
X_p = parkinsons.data.features 
y_p = parkinsons.data.targets

# load models from files
clf_diabetes = joblib.load("model/diabetes.pkl")
clf_breast = joblib.load("model/breast.pkl")
clf_heart = joblib.load("model/heart.pkl")
clf_parkinson = joblib.load("model/parkinson.pkl")

model = MCPMesaModel(X_d, X_b, X_h, X_p, clf_diabetes, clf_breast, clf_heart, clf_parkinson)
model.step()

# ==========================================
# STEP 5: Visualize Results with Seaborn
# ==========================================

# Analyse the triage performance
print("======== Done with scheduling and execution ==========\n")
triage_df = pd.DataFrame(model.triage_log)
print("Triage Collected:", len(triage_df))
print(triage_df.head())

# calculate Triage precision and recall
TP = 0
FP = 0
FN = 0
TN = 0
for index, row in triage_df.iterrows():
    for domain in row["labels"].keys():
        truth_label = row["labels"][domain]

        if domain in row["routes"] and truth_label == 1:
            TP += 1
        elif domain in row["routes"] and truth_label == 0:
            FP += 1
        elif domain not in row["routes"] and truth_label == 1:
            FN += 1
        else:
            TN += 1
# print precision and recall
precision = TP/(TP+FP)
recall = TP/(TP+FN)
print(f"\n====== Triage performance: =====\nTP: {TP}\nFP: {FP}\nTN: {TN}\nFN: {FN}\nPrecision: {precision}\nRecall: {recall}")

# plot confussion matrix
cm = np.array([[TN, FP],[FN, TP]])
sns.heatmap(cm, annot=True, fmt="d", cmap= "viridis",
            xticklabels=['0', '1'],
            yticklabels=['0', '1'])
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Confusion Matrix: triage performance")
plt.savefig("results/confusion_triage.png")


# show first 5 results of diseases
results_df = pd.DataFrame(model.results)
print("Results Collected:", len(results_df))
print(results_df.head())
print()
# save confusion matrix plot per tested disease
if not results_df.empty and "Disease" in results_df.columns and "y_pred" in results_df.columns:
    plt.figure(figsize=(10, 6))
    sns.countplot(data=results_df, x="Disease", hue="y_pred")
    plt.title("Predictions by Disease Type")
    plt.ylabel("Number of Cases")
    plt.xlabel("Disease")
    plt.legend(title="Prediction", labels=["Negative", "Positive"])
    plt.tight_layout()
    plt.savefig("results/output.png")
else:
    print("No results to display.")

# print confusion matrix per tested disease
for disease in results_df["Disease"].unique():
    sub = results_df[results_df["Disease"] == disease].dropna(subset=["y_true", "y_pred"])
    if sub.empty:
        print(f"\n--- {disease}: no labeled results ---")
        continue

    y_true = sub["y_true"].astype(int).values
    y_pred = sub["y_pred"].astype(int).values

    print(f"\n=== {disease} ===")
    print(classification_report(y_true, y_pred, digits=3))

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"Confusion matrix: {disease}")
    plt.tight_layout()
    plt.savefig(f"results/confusion_{disease}.png")
    plt.close()