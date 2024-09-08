# from hmmlearn import hmm
# import numpy as np
# import logging

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# def create_hmm_model():
#     states = ["Healthy", "At Risk", "Diseased"]
#     n_states = len(states)

#     observations = ["Low", "Medium", "High"]
#     n_observations = len(observations)

#     start_probability = np.array([0.6, 0.3, 0.1])
#     transition_probability = np.array([
#         [0.7, 0.2, 0.1],
#         [0.3, 0.5, 0.2],
#         [0.2, 0.3, 0.5]
#     ])
#     emission_probability = np.array([
#         [0.8, 0.15, 0.05],
#         [0.2, 0.6, 0.2],
#         [0.1, 0.3, 0.6]
#     ])

#     model = hmm.MultinomialHMM(n_components=n_states, n_iter=100, tol=1e-4)
#     model.startprob_ = start_probability
#     model.transmat_ = transition_probability
#     model.emissionprob_ = emission_probability

#     return model

# def train_hmm_model(model, obs_sequence, lengths):
#     try:
#         model.fit(obs_sequence, lengths)
#         logging.info("Model trained successfully!")
#     except ValueError as e:
#         logging.error(f"Error during model training: {e}")
#     except Exception as e:
#         logging.error(f"Unexpected error during model training: {e}")
#     return model

# if __name__ == "__main__":
#     from feature_extraction import calculate_gc_content
#     from data_collection import fetch_dna_sequence

#     dna_sequence = fetch_dna_sequence("NM_007294")
#     if dna_sequence:
#         gc_content_level_seq = [
#             0 if calculate_gc_content(dna_sequence[i:i + 100]) < 40 
#             else 1 if calculate_gc_content(dna_sequence[i:i + 100]) < 60 
#             else 2 
#             for i in range(0, len(dna_sequence), 100)
#         ]
        
#         obs_sequence = np.array([gc_content_level_seq]).T
#         lengths = [len(gc_content_level_seq)]

#         model = create_hmm_model()
#         model = train_hmm_model(model, obs_sequence, lengths)
# from hmmlearn import hmm
# import numpy as np
# import logging

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# def create_hmm_model():
#     states = ["Healthy", "At Risk", "Diseased"]
#     n_states = len(states)

#     observations = ["Low", "Medium", "High"]
#     n_observations = len(observations)

#     start_probability = np.array([0.6, 0.3, 0.1])
#     transition_probability = np.array([
#         [0.7, 0.2, 0.1],
#         [0.3, 0.5, 0.2],
#         [0.2, 0.3, 0.5]
#     ])
#     emission_probability = np.array([
#         [0.8, 0.15, 0.05],
#         [0.2, 0.6, 0.2],
#         [0.1, 0.3, 0.6]
#     ])

#     model = hmm.MultinomialHMM(n_components=n_states, n_iter=100, tol=1e-4)
#     model.startprob_ = start_probability
#     model.transmat_ = transition_probability
#     model.emissionprob_ = emission_probability

#     return model

# def train_hmm_model(model, obs_sequence, lengths):
#     try:
#         model.fit(obs_sequence, lengths)
#         logging.info("Model trained successfully!")
#     except ValueError as e:
#         logging.error(f"Error during model training: {e}")
#     except Exception as e:
#         logging.error(f"Unexpected error during model training: {e}")
#     return model

# if __name__ == "__main__":
#     from feature_extraction import calculate_gc_content
#     from data_collection import fetch_dna_sequence

#     dna_sequence = fetch_dna_sequence("NM_007294")
#     if dna_sequence:
#         gc_content_level_seq = [
#             0 if calculate_gc_content(dna_sequence[i:i + 100]) < 40 
#             else 1 if calculate_gc_content(dna_sequence[i:i + 100]) < 60 
#             else 2 
#             for i in range(0, len(dna_sequence), 100)
#         ]
        
#         obs_sequence = np.array([gc_content_level_seq]).T
#         lengths = [len(gc_content_level_seq)]

#         model = create_hmm_model()
#         model = train_hmm_model(model, obs_sequence, lengths)

# hmm_model.py

import numpy as np
from hmmlearn import hmm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_hmm_model(n_components, n_features):
    model = hmm.CategoricalHMM(n_components=n_components, n_iter=100, tol=1e-4, init_params="st")
    
    # Initialize start probabilities uniformly or based on domain knowledge
    model.startprob_ = np.full(n_components, 1 / n_components)
    
    # Initialize transition matrix uniformly or based on domain knowledge
    model.transmat_ = np.full((n_components, n_components), 1 / n_components)
    
    # Initialize emission probabilities uniformly or based on domain knowledge
    model.emissionprob_ = np.full((n_components, n_features), 1 / n_features)
    
    return model

def train_hmm_model(model, obs_sequence, lengths):
    try:
        model.fit(obs_sequence, lengths)
        logging.info("Model trained successfully!")
        return model
    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise



