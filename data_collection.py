# import os
# import logging
# from Bio import Entrez, SeqIO
# from Bio.Entrez import HTTPError

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# def fetch_dna_sequence(accession_id):
#     """
#     Fetches the DNA sequence for the given accession number from NCBI.
#     Returns the DNA sequence as a string.
#     """
#     Entrez.email = os.getenv("NCBI_EMAIL")  # Use environment variable for email
#     try:
#         # Fetch the DNA sequence from the NCBI nucleotide database
#         with Entrez.efetch(db="nucleotide", id=accession_id, rettype="gb", retmode="text") as handle:
#             record = SeqIO.read(handle, "genbank")
#             logging.info(f"Successfully fetched sequence for accession ID: {accession_id}")
#             return str(record.seq)
#     except HTTPError as e:
#         logging.error(f"HTTP Error while fetching {accession_id}: {e}")
#         return ""
#     except Exception as e:
#         logging.error(f"Unexpected error while fetching {accession_id}: {e}")
#         return ""

# def fetch_multiple_sequences(accession_ids):
#     """
#     Fetches multiple DNA sequences given a list of accession numbers.
#     Returns a dictionary mapping accession IDs to their sequences.
#     """
#     sequences = {}
#     for acc_id in accession_ids:
#         seq = fetch_dna_sequence(acc_id)
#         if seq:
#             sequences[acc_id] = seq
#     return sequences

# if __name__ == "__main__":
#     # Example usage
#     accession_ids = ["NM_007294", "NM_000059", "NM_000546"]  # BRCA1, BRCA2, TP53
#     sequences = fetch_multiple_sequences(accession_ids)
#     for acc_id, seq in sequences.items():
#         print(f"{acc_id}: Length {len(seq)}")



from Bio import Entrez, SeqIO
from Bio.Entrez import HTTPError
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_dna_sequence(accession_id):
    """
    Fetches the DNA sequence for the given accession number from NCBI.
    Returns the DNA sequence as a string.
    """
    Entrez.email = "sushan.adhikari2060@gmail.com"  # Replace with your email

    try:
        handle = Entrez.efetch(db="nucleotide", id=accession_id, rettype="gb", retmode="text")
        record = SeqIO.read(handle, "genbank")
        handle.close()
        logging.info(f"Successfully fetched sequence for accession ID: {accession_id}")
        return str(record.seq)
    except HTTPError as e:
        logging.error(f"An error occurred while fetching the DNA sequence: {e}")
        return ""
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return ""

if __name__ == "__main__":
    accession_ids = ["NM_007294", "NM_000059", "NM_000546"]  # BRCA1, BRCA2, TP53
    sequences = []
    for accession_id in accession_ids:
        sequence = fetch_dna_sequence(accession_id)
        if sequence:
            sequences.append(sequence)
    
    if sequences:
        with open("dna_sequences.fasta", "w") as output_handle:
            for i, seq in enumerate(sequences):
                SeqIO.write(SeqRecord(Seq(seq), id=accession_ids[i]), output_handle, "fasta")
        logging.info("Saved all sequences to dna_sequences.fasta.")
    else:
        logging.error("Failed to fetch any sequences.")
