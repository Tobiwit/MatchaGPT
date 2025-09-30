import pandas as pd
import ollama
from codecarbon import EmissionsTracker
from openpyxl import Workbook
import re

MODEL_NAME = "Gemma3:4b"
SMALL_MODEL_NAME = "Gemma3:1b"
INPUT_FILE = "prompts.csv"
OUTPUT_FILE = "antworten.xlsx"
N_PROMPTS = 100


def extract_rating(text: str) -> int:
    patterns = [
        r'\b([1-9]|10)\b',
        r'([1-9]|10)/10',
        r'rating:?\s*([1-9]|10)',
        r'score:?\s*([1-9]|10)',
        r'complexity:?\s*([1-9]|10)',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            try:
                rating = int(matches[0])
                if 1 <= rating <= 10:
                    return rating
            except ValueError:
                continue
    return -1



def evaluate_complexity(prompt) -> int:
        system_prompt = f"""Analyze the following prompt and rate its complexity from 1 to 10:

        Prompt: {prompt}

        Only respond with the complexity score (1-10)."""
        
        response = ollama.chat(
        model=SMALL_MODEL_NAME,
        messages=[{"role": "user", "content": system_prompt}]
        )

        text = response["message"]["content"].strip()
        complexity = extract_rating(text)
        return complexity
        

def query_model(prompt, complexity):
    if complexity > 5:
        """Fragt das Modell mit einem Prompt ab und gibt die Antwort zurück."""
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}]
        )
    else:
        """Fragt das Modell mit einem Prompt ab und gibt die Antwort zurück."""
        response = ollama.chat(
            model=SMALL_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}]
        )
    return response['message']['content'].strip()

def switch_model(complexity):
    """Stellt dem Modell eine Frage die fast keine Energie zur Antwortgenerierung verbraucht."""
    query_model("What is 1+1?",complexity)

def main():
    # CSV einlesen
    df = pd.read_csv(INPUT_FILE, sep=";")
    df = df.head(N_PROMPTS)

    results = []

    # Tracker starten
    tracker = EmissionsTracker(measure_power_secs=1, log_level="error", save_to_file=False)
    tracker.start()

    from codecarbon.core.units import Energy

    for idx, row in df.iterrows():
        sess_id = row['sess_id']
        prompt = row['prompt']
        try:
            energy1 = tracker._total_energy or Energy(0.0)
            switch_model(1)
            energy2 = tracker._total_energy or Energy(0.0)
            complexity = evaluate_complexity(prompt)
            energy3 = tracker._total_energy or Energy(0.0)
            switch_model(complexity)
            energy4 = tracker._total_energy or Energy(0.0)
            answer = query_model(prompt, complexity)
            energy5 = tracker._total_energy or Energy(0.0)
            
            energy_diff_switch1 = energy2 - energy1
            energy_diff_complexity = energy3 - energy2
            energy_diff_switch2 = energy4 - energy3
            energy_diff_answer = energy5 - energy4
            total_energy_diff = energy5 - energy1
        except Exception as e:
            complexity = -1  # Defaultwert, wenn Bewertung fehlschlägt
            answer = f"[Fehler bei Modellabfrage: {e}]"
            energy_diff_answer = 0;
            energy_diff_complexity = 0;



        results.append({
            "sess_id": sess_id,
            "prompt": prompt,
            "complexity": complexity,
            "answer": answer,
            "energy_switch1": energy_diff_switch1.kWh,
            "energy_complexity": energy_diff_complexity.kWh,
            "energy_switch2": energy_diff_switch2.kWh,
            "energy_answer": energy_diff_answer.kWh
        })

        print(f"✅ {idx+1}/{N_PROMPTS} verarbeitet – Verbrauch: {total_energy_diff.kWh:.6f} kWh")

    # Tracker stoppen
    tracker.stop()
    total_energy = energy5.kWh

    # Ergebnisse in Excel speichern + Summenzeile
    output_df = pd.DataFrame(results)

    # Summenzeile hinzufügen
    total_row = {
        "sess_id": "TOTAL",
        "prompt": "",
        "complexity": "",
        "answer": "",
        "energy_consumed_kWh": total_energy
    }
    output_df = pd.concat([output_df, pd.DataFrame([total_row])], ignore_index=True)

    output_df.to_excel(OUTPUT_FILE, index=False)

    print("\n===================================")
    print(f"Alle {N_PROMPTS} Prompts verarbeitet.")
    print(f"Gesamtstromverbrauch: {total_energy:.6f} kWh")
    print(f"Ergebnisse gespeichert in: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
