# MatchaGPT
single_model_with_energy_tracker.py runs a single ollama model on a prompt dataset, and measures the energy.

multigemma_with_energytracker.py runs a combination of Gemma3:1b and Gemma3:4b on a prompt dataset, and measures the energy.

evaluate_prompt.py takes in a dataset of prompts and answers to those prompts, and lets Gemma3:27b rate those answers on a scale from 1 to 10.

measure_switch_costs.py runs the multigemma model and separately measures the energy whenever ollama has to switch from Gemma3:1b to Gemma3:4b or vice versa.

The evaluations folder contains measurements for all models performed on our primary workstation owned by Raffael Eger.
The peter pc measurements folder contains measurements for all gemma models and multigemma performed on our secondary workstation owned by Peter Bonart.
The evaluator_tests folder contains several tests to determine which large language model is the most consistent, reliable and valid answer evaluator.