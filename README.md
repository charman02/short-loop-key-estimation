# Key Estimation on Short Loops
This project investigates the problem of musical key estimation for short audio loops, a task that presents unique challenges due to the limited tonal context of looped samples. While existing models such as STONE (the first self-supervised tonality estimator) and its extension S-KEY have achieved strong results on full-length tracks, their performance on short loops commonly used in real-world music production workflows has not been extensively studied. To address this gap, we propose a methodology that fine-tunes and adapts S-KEY specifically for short audio segments. Our approach leverages self-supervised learning to exploit large quantities of unlabeled audio, while preprocessing techniques such as constant-Q transform and chroma feature extraction enable robust tonal representation. We evaluate the adapted model on benchmark datasets including FMAKv2 and GiantSteps by measuring performance with standard MIREX key detection metrics.

# Command Line Interface (CLI)
     pip install poetry
     poetry add
     poetry run python eval.py
  
1. Enter the directory
2. Install poetry
3. Add necessary packages in poetry
4. Run evaluation script
5. Check results in eval/metrics.csv and predictions in skey/models/ks_mode/7/basic/MODEL_NAME/predictions.json, for example.

Notes:
- To change the model being evaluated, replace DEFAULT_CHECKPOINT_PATH in the "Load Model" section of eval.py.
