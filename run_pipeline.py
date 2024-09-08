# import os
# import logging
# from Bio import SeqIO
# from Bio.Seq import Seq
# from Bio.SeqRecord import SeqRecord
# import numpy as np

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# def main():
#     from data_collection import fetch_multiple_sequences
#     from feature_extraction import perform_alignment, calculate_gc_content, extract_cpg_islands, plot_gc_content
#     from hmm_model import create_hmm_model, train_hmm_model, save_model, load_model, plot_transition_matrix
#     from risk_analysis import predict_risk, evaluate_model
#     from sklearn.model_selection import train_test_split
    
#     # Step 1: Fetch DNA Sequences
#     accession_ids = ["NM_007294", "NM_000059", "NM_000546"]  # BRCA1, BRCA2, TP53
#     sequences = fetch_multiple_sequences(accession_ids)
#     if not sequences:
#         logging.error("No sequences fetched. Exiting pipeline.")
#         return
    
#     # Step 2: Save Sequences to FASTA
#     seq_records = [SeqRecord(Seq(seq), id=acc_id, description=f"{acc_id} Sequence") for acc_id, seq in sequences.items()]
#     input_fasta = "dna_sequences.fasta"
#     with open(input_fasta, "w") as f:
#         SeqIO.write(seq_records, f, "fasta")
#     logging.info(f"Saved {len(seq_records)} sequences to {input_fasta}.")
    
#     # Step 3: Perform Alignment
#     alignment_success = perform_alignment(input_fasta, output_fasta="aligned_sequences.fasta")
#     if not alignment_success:
#         logging.warning("Alignment step failed or was skipped.")
    
#     # Step 4: Feature Extraction and Visualization
#     for acc_id, seq in sequences.items():
#         gc = calculate_gc_content(seq)
#         logging.info(f"{acc_id} GC Content: {gc:.2f}%")
#         cpg_islands = extract_cpg_islands(seq)
#         logging.info(f"{acc_id} CpG Islands Found: {len(cpg_islands)}")
#         plot_gc_content(seq, window=100)
    
#     # Step 5: Prepare Data for HMM
#     # For demonstration, process the first sequence
#     primary_acc_id = accession_ids[0]
#     dna_sequence = sequences[primary_acc_id]
    
#     window_size = 100
#     gc_content_level_seq = [
#         0 if calculate_gc_content(dna_sequence[i:i+window_size]) < 40 
#         else 1 if calculate_gc_content(dna_sequence[i:i+window_size]) < 60 
#         else 2 
#         for i in range(0, len(dna_sequence), window_size)
#     ]
    
#     obs_sequence = np.array(gc_content_level_seq).reshape(-1, 1)
#     lengths = [len(gc_content_level_seq)]
    
#     # Step 6: Create and Train HMM
#     model = create_hmm_model()
#     model = train_hmm_model(model, obs_sequence, lengths)
#     save_model(model)
    
#     # Step 7: Visualize HMM Transition Matrix
#     states = ["Healthy", "At Risk", "Diseased"]
#     plot_transition_matrix(model, states, filename="hmm_transition_matrix.png")
    
#     # Step 8: Risk Prediction and Evaluation
#     # Generate synthetic true labels for demonstration
#     y_true = [0, 1, 2] * (len(gc_content_level_seq) // 3)
#     y_true = y_true[:len(gc_content_level_seq)]  # Ensure same length
    
#     # Split data
#     obs_train, obs_test, y_train, y_test = train_test_split(
#         obs_sequence, y_true, test_size=0.3, random_state=42, stratify=y_true)
    
#     # Retrain model on training data
#     model = create_hmm_model()
#     model = train_hmm_model(model, obs_train, [len(y_train)])
#     save_model(model)
    
#     # Predict and evaluate on test data
#     predicted_states = predict_risk(model, obs_test)
#     logging.info(f"Predicted States: {predicted_states}")
#     evaluate_model(model, y_test, obs_test, states=states, filename_prefix="hmm_evaluation_")

# if __name__ == "__main__":
#     main()


# run_pipeline.py

import numpy as np
import logging
from hmm_model import create_hmm_model, train_hmm_model
from feature_extraction import calculate_gc_content
from data_collection import fetch_dna_sequence
from risk_analysis import predict_risk, evaluate_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Fetch DNA sequence
    dna_sequence = fetch_dna_sequence("NM_007294")
    
    # Define window size
    window_size = 100
    
    # Extract GC content levels
    gc_content_levels = []
    for i in range(0, len(dna_sequence) - window_size + 1, window_size):
        window_seq = dna_sequence[i:i+window_size]
        gc_content = calculate_gc_content(window_seq)
        if gc_content < 40:
            gc_content_levels.append(0)
        elif gc_content < 60:
            gc_content_levels.append(1)
        else:
            gc_content_levels.append(2)
    
    # Prepare observation sequence
    obs_sequence = np.array(gc_content_levels).reshape(-1, 1)
    lengths = [len(gc_content_levels)]
    
    # Define model parameters
    n_components = 3  # Number of hidden states: Healthy, At Risk, Diseased
    n_features = 3    # Number of observation symbols: 0, 1, 2
    
    # Create and train HMM model
    model = create_hmm_model(n_components=n_components, n_features=n_features)
    model = train_hmm_model(model, obs_sequence, lengths)
    
    # Predict risk states
    predicted_states = predict_risk(model, obs_sequence)
    logging.info(f"Predicted States: {predicted_states}")
    
    # Generate true labels for evaluation (this should come from actual data)
    y_true = predicted_states  # Placeholder: replace with actual true labels if available
    
    # Evaluate model performance
    evaluate_model(model, y_true, obs_sequence)

if __name__ == "__main__":
    main()





