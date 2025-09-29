import pandas as pd
import ollama
from codecarbon import EmissionsTracker
from openpyxl import Workbook
from typing import Optional
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

def main():
    # CSV einlesen
    df = pd.read_csv(INPUT_FILE, sep=";")
    df = df.head(N_PROMPTS)

    results = []

    # Tracker starten
    tracker = EmissionsTracker(measure_power_secs=1, log_level="error", save_to_file=False)
    tracker.start()

    from codecarbon.core.units import Energy
    prev_energy = Energy(0.0)  # kein unit-Parameter

    for idx, row in df.iterrows():
        sess_id = row['sess_id']
        prompt = row['prompt']

        try:
            complexity = evaluate_complexity(prompt)
            answer = query_model(prompt, complexity)
        except Exception as e:
            answer = f"[Fehler bei Modellabfrage: {e}]"
            complexity = 0

        # Energie seit Start
        current_energy = tracker._total_energy or Energy(0.0)
        energy_diff = current_energy - prev_energy
        prev_energy = current_energy

        results.append({
            "sess_id": sess_id,
            "prompt": prompt,
            "complexity": complexity,
            "answer": answer,
            "energy_consumed_kWh": energy_diff.kWh
        })

        print(f"✅ {idx+1}/{N_PROMPTS} verarbeitet – Verbrauch: {energy_diff.kWh:.6f} kWh")

    # Tracker stoppen
    tracker.stop()
    total_energy = prev_energy.kWh

    # Ergebnisse in Excel speichern + Summenzeile
    output_df = pd.DataFrame(results)

    output_df.to_excel(OUTPUT_FILE, index=False)

    print("\n===================================")
    print(f"Alle {N_PROMPTS} Prompts verarbeitet.")
    print(f"Gesamtstromverbrauch: {total_energy:.6f} kWh")
    print(f"Ergebnisse gespeichert in: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
