import csv
import time
from datetime import datetime
import pandas as pd
from codecarbon import EmissionsTracker
import ollama
import os

# CSV-Dateien
script_dir = os.path.dirname(os.path.abspath(__file__))
input_csv = os.path.join(script_dir, "bidd1k_2.csv")
output_csv = os.path.join(script_dir, "results.csv")

# Modelle
models = ["gemma2:2b", "mistral:7b"]  # nur verfügbare Modelle

# Prompts einlesen
prompts = []
with open(input_csv, mode='r', encoding='utf-8') as file:
    reader = csv.reader(file, delimiter=';')
    for row in reader:
        if len(row) < 2 or not row[1].strip():  # leere oder unvollständige Zeilen überspringen
            continue
        prompt_id, prompt_text = row[0].strip(), row[1].strip()
        prompts.append((prompt_id, prompt_text))

# Ergebnisse speichern
results = []
global_id = 1  # Startwert für globale ID

for model in models:
    for prompt_id, prompt_text in prompts:  # nur gültige Prompts
        tracker = EmissionsTracker(project_name="model_efficiency_test")
        tracker.start()
        start_time = time.time()

        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt_text}]
        )

        end_time = time.time()
        tracker.stop()

        energy_consumed = tracker.final_emissions_data.emissions  # kWh
        duration = end_time - start_time
        timestamp = datetime.now().isoformat()

        results.append({
            "Global_ID": global_id,
            "Prompt_ID": prompt_id,
            "Model": model,
            "Prompt": prompt_text,
            "Response": response['message']['content'],
            "Energy_kWh": energy_consumed,
            "Duration_s": duration,
            "Timestamp": timestamp
        })

        global_id += 1  # hochzählen

# In CSV speichern
df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)
print(f"Ergebnis CSV gespeichert unter: {output_csv}")