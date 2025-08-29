import pandas as pd
import ollama
from codecarbon import EmissionsTracker
from openpyxl import Workbook

MODEL_NAME = "Deepseek-r1:8b"   # oder 'llama3', 'phi3', etc.
INPUT_FILE = "prompts.csv"
OUTPUT_FILE = "antworten.xlsx"
N_PROMPTS = 100

def query_model(prompt):
    """Fragt das Modell mit einem Prompt ab und gibt die Antwort zurück."""
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}]
    )
    return response['message']['content'].strip()

def main():
    # CSV einlesen
    df = pd.read_csv(INPUT_FILE)
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
            answer = query_model(prompt)
        except Exception as e:
            answer = f"[Fehler bei Modellabfrage: {e}]"

        # Energie seit Start
        current_energy = tracker._total_energy or Energy(0.0)
        energy_diff = current_energy - prev_energy
        prev_energy = current_energy

        results.append({
            "sess_id": sess_id,
            "prompt": prompt,
            "answer": answer,
            "energy_consumed_kWh": energy_diff.kWh
        })

        print(f"✅ {idx+1}/{N_PROMPTS} verarbeitet – Verbrauch: {energy_diff.kWh:.6f} kWh")

    # Tracker stoppen
    tracker.stop()
    total_energy = prev_energy.kWh

    # Ergebnisse in Excel speichern + Summenzeile
    output_df = pd.DataFrame(results)
    output_df.loc[len(output_df.index)] = ["TOTAL", "", "", total_energy]

    output_df.to_excel(OUTPUT_FILE, index=False)

    print("\n===================================")
    print(f"Alle {N_PROMPTS} Prompts verarbeitet.")
    print(f"Gesamtstromverbrauch: {total_energy:.6f} kWh")
    print(f"Ergebnisse gespeichert in: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
