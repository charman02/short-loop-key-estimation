# Key Estimation on Short Loops
This project investigates the problem of musical key estimation for short audio loops, a task that presents unique challenges due to the limited tonal context of looped samples. While existing models such as STONE (the first self-supervised tonality estimator) and its extension S-KEY have achieved strong results on full-length tracks, their performance on short loops commonly used in real-world music production workflows has not been extensively studied. To address this gap, we propose a methodology that fine-tunes and adapts S-KEY specifically for short audio segments. Our approach leverages self-supervised learning to exploit large quantities of unlabeled audio, while preprocessing techniques such as constant-Q transform and chroma feature extraction enable robust tonal representation. We evaluate the adapted model on benchmark datasets including FMAKv2 and GiantSteps by measuring performance with standard MIREX key detection metrics.

# Code Usage
## Environment Setup
     git clone https://github.com/charman02/short-loop-key-estimation
     cd short-loop-key-estimation

## Evaluation
     poetry install
     poetry add
     poetry run python eval.py
  
1. Enter the directory
2. Install poetry
3. Add necessary packages in poetry
4. Run evaluation script
5. Check results in 'eval/metrics.csv' and predictions in '<save_dir>/models/<train_type>/<circle_type>/<exp_name>/<model_name>/predictions.json', for example.

Notes:
- To change the model being evaluated, replace DEFAULT_CHECKPOINT_PATH in the "Load Model" section of eval.py.

## Training
### Example Usage
     poetry run python -m main -n basic -tt ks_mode -s skey -e 20 -ts 20 -vs 10 -g training_utils/config/skey.gin

### Command-Line Arguments
- '-n' '--exp-name', the name of the experiment. If the experiment name was used before and checkpoints are saved, then the training will continue from the checkpoint of the experiment of the same name.
- '-tt' '--train-type', type of training: ks (key signature) for stone12, ks_mode for skey.
- '-c' '--circle-type', the circle where key signature profile is projected. 1 or 7. 1 for circle of semitones and 7 for circle of fifths.
- '-s' '--save-dir', the path where the checkpoint will be saved.
- '-e' '--n-epochs', number of epochs.
- '-ts' '-train-steps', steps of training per epoch.
- '-vs' '--val-steps', steps of validation per epoch.
- '-g' '--gin-file', path to configuration file for training. There are two gin files saved in /config, users can modify them to their own purpose.

Notes:
- Checkpoints are saved in '<save_dir>/models/<train_type>/<circle_type>/<exp_name>/<model_name>/'.

# References
     @article{kong2024stone,
       title={STONE: Self-supervised Tonality Estimator},
       author={Kong, Yuexuan and Lostanlen, Vincent and Meseguer-Brocal, Gabriel and Wong, Stella and Lagrange, Mathieu and Hennequin, Romain},
       journal={International Society for Music Information Retrieval Conference (ISMIR 2024)},
       year={2024}
     }
     @INPROCEEDINGS{kongskey2025,
       author={Kong, Yuexuan and Meseguer-Brocal, Gabriel and Lostanlen, Vincent and Lagrange, Mathieu and Hennequin, Romain},
       booktitle={ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
       title={S-KEY: Self-supervised Learning of Major and Minor Keys from Audio}, 
       year={2025},
       pages={1-5},
       doi={10.1109/ICASSP49660.2025.10890222}}
