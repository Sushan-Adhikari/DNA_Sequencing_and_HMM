# import os
# import logging
# import joblib
# from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
# from sklearn.preprocessing import label_binarize
# import matplotlib.pyplot as plt
# import numpy as np

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# def predict_risk(model, obs_sequence):
#     """
#     Predicts the sequence of hidden states using the Viterbi algorithm.
#     """
#     states_sequence = model.predict(obs_sequence)
#     logging.info("Risk states predicted using Viterbi algorithm.")
#     return states_sequence

# def evaluate_model(model, y_true, obs_sequence, states=["Healthy", "At Risk", "Diseased"], filename_prefix=""):
#     """
#     Evaluates the HMM model using ROC and AUC metrics, plotting the ROC curve.
#     Also plots the confusion matrix.
#     """
#     try:
#         # Obtain log probabilities for each state
#         log_probs = model.predict_proba(obs_sequence)
        
#         # Binarize the true labels for multi-class ROC
#         y_true_binarized = label_binarize(y_true, classes=range(len(states)))
#         n_classes = y_true_binarized.shape[1]
        
#         # Compute ROC curve and AUC for each class
#         fpr = dict()
#         tpr = dict()
#         roc_auc = dict()
#         for i in range(n_classes):
#             fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], log_probs[:, i])
#             roc_auc[i] = auc(fpr[i], tpr[i])
        
#         # Plot ROC curves for each class
#         plt.figure()
#         colors = ['blue', 'green', 'red']
#         for i, color in zip(range(n_classes), colors):
#             plt.plot(fpr[i], tpr[i], color=color, lw=2,
#                      label=f'ROC curve of {states[i]} (area = {roc_auc[i]:0.2f})')
        
#         plt.plot([0, 1], [0, 1], 'k--', lw=2)
#         plt.xlim([0.0, 1.0])
#         plt.ylim([0.0, 1.05])
#         plt.xlabel('False Positive Rate')
#         plt.ylabel('True Positive Rate')
#         plt.title('Receiver Operating Characteristic for HMM States')
#         plt.legend(loc="lower right")
#         roc_filename = f"{filename_prefix}roc_curve.png"
#         plt.savefig(roc_filename)
#         plt.show()
#         logging.info(f"ROC curves plotted and saved as {roc_filename}.")
        
#         # Classification Report
#         print("Classification Report:")
#         print(classification_report(y_true, model.predict(obs_sequence), target_names=states))
        
#         # Confusion Matrix
#         cm = confusion_matrix(y_true, model.predict(obs_sequence))
#         plot_confusion_matrix(cm, classes=states, title='Confusion Matrix', filename=f"{filename_prefix}confusion_matrix.png")
        
#     except Exception as e:
#         logging.error(f"Error during evaluation: {e}")

# def plot_confusion_matrix(cm, classes,
#                           normalize=False,
#                           title='Confusion matrix',
#                           cmap=plt.cm.Blues,
#                           filename="confusion_matrix.png"):
#     """
#     Plots the confusion matrix.
#     """
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         logging.info("Normalized confusion matrix.")
#     else:
#         logging.info("Confusion matrix without normalization.")
    
#     plt.figure(figsize=(6, 6))
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)
    
#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
    
#     for i in range(cm.shape[0]):
#         for j in range(cm.shape[1]):
#             plt.text(j, i, format(cm[i, j], fmt),
#                      ha="center", va="center",
#                      color="white" if cm[i, j] > thresh else "black")
    
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     plt.tight_layout()
#     plt.savefig(filename)
#     plt.show()
#     logging.info(f"Confusion matrix plot saved as {filename}.")

# if __name__ == "__main__":
#     from hmm_model import create_hmm_model, train_hmm_model, save_model, load_model
#     from feature_extraction import calculate_gc_content
#     from data_collection import fetch_dna_sequence
#     from sklearn.model_selection import train_test_split
    
#     # Fetch DNA sequence
#     dna_sequence = fetch_dna_sequence("NM_007294")  # BRCA1
#     if not dna_sequence:
#         logging.error("Failed to fetch DNA sequence. Exiting.")
#         exit(1)
    
#     # Prepare observation sequence
#     window_size = 100
#     gc_content_level_seq = [
#         0 if calculate_gc_content(dna_sequence[i:i+window_size]) < 40 
#         else 1 if calculate_gc_content(dna_sequence[i:i+window_size]) < 60 
#         else 2 
#         for i in range(0, len(dna_sequence), window_size)
#     ]
    
#     obs_sequence = np.array(gc_content_level_seq).reshape(-1, 1)
#     lengths = [len(gc_content_level_seq)]
    
#     # Generate synthetic true labels (for demonstration)
#     # In a real scenario, replace with actual labels
#     y_true = [0, 1, 2] * (len(gc_content_level_seq) // 3)
#     y_true = y_true[:len(gc_content_level_seq)]  # Ensure same length
    
#     # Split data into training and testing
#     obs_train, obs_test, y_train, y_test = train_test_split(
#         obs_sequence, y_true, test_size=0.3, random_state=42, stratify=y_true)
    
#     # Create and train HMM
#     model = create_hmm_model()
#     model = train_hmm_model(model, obs_train, lengths)
    
#     # Save the trained model
#     save_model(model)
    
#     # Predict on test data
#     predicted_states = model.predict(obs_test)
#     logging.info(f"Predicted States: {predicted_states}")
    
#     # Evaluate the model
#     evaluate_model(model, y_test, obs_test, filename_prefix="hmm_")


# risk_analysis.py

import numpy as np
import logging
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def predict_risk(model, obs_sequence):
    logprob, states_sequence = model.decode(obs_sequence, algorithm="viterbi")
    return states_sequence

def evaluate_model(model, y_true, obs_sequence):
    try:
        y_pred = model.predict(obs_sequence)
        
        # Determine unique classes present in y_true and y_pred
        unique_classes = np.unique(np.concatenate((y_true, y_pred)))
        
        if len(unique_classes) == 0:
            logging.error("No matching classes between y_true and y_pred. Evaluation cannot be performed.")
            return
        
        # Classification report
        report = classification_report(y_true, y_pred, labels=unique_classes, target_names=["Healthy", "At Risk", "Diseased"][:len(unique_classes)], zero_division=0)
        logging.info("\nClassification Report:\n" + report)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=unique_classes)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=["Healthy", "At Risk", "Diseased"][:len(unique_classes)],
                    yticklabels=["Healthy", "At Risk", "Diseased"][:len(unique_classes)])
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        plt.show()
        
    except Exception as e:
        logging.error(f"Error occurred during evaluation: {e}")
        raise


