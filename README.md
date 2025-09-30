# MatchaGPT
single_model_with_energy_tracker.py runs a single ollama model on a prompt dataset, and measures the energy.

multigemma_with_energytracker.py runs a combination of Gemma3:1b and Gemma3:4b on a prompt dataset, and measures the energy.

evaluate_prompt.py takes in a dataset of prompts and answers to those prompts, and lets Gemma3:27b rate those answers on a scale from 1 to 10.

The evaluations folder contains measurements for all models performed on our primary workstation owned by Raffael Eger.
The peter pc measurements folder contains measurements for all gemma models and multigemma performed on our secondary workstation owned by Peter Bonart.

The evaluator_tests folder contains several tests to determine which large language model is the most consistent, reliable and valid answer evaluator.

The switching_costs folder contains a script to measure the cost of switching between Gemma3:1b and Gemma3:4b while running multigemma. It contains results for the two workstations.

The llama3_70b_evaluations folder contains evaluations that llama3:70b created for Gemma3:27b and llama3:70b.