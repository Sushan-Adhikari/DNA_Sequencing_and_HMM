# import os
# import logging
# import subprocess
# from Bio import SeqIO
# from Bio.Seq import Seq
# from Bio.SeqRecord import SeqRecord
# import matplotlib.pyplot as plt

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# def perform_alignment(input_fasta, output_fasta="aligned_sequences.fasta", mafft_exe="mafft"):
#     """
#     Performs multiple sequence alignment using MAFFT.
#     """
#     # Check if MAFFT is installed
#     if not shutil.which(mafft_exe):
#         logging.error(f"MAFFT executable '{mafft_exe}' not found. Please install MAFFT and ensure it's in your PATH.")
#         return False
    
#     sequences = list(SeqIO.parse(input_fasta, "fasta"))
#     if len(sequences) < 2:
#         logging.warning("Alignment skipped: At least two sequences are required.")
#         return False
    
#     try:
#         # Execute MAFFT with input and output files
#         result = subprocess.run([mafft_exe, "--auto", input_fasta],
#                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
#         with open(output_fasta, 'w') as f:
#             f.write(result.stdout)
#         logging.info(f"Alignment successful! Output saved to {output_fasta}")
#         return True
#     except subprocess.CalledProcessError as e:
#         logging.error(f"Error during MAFFT alignment: {e.stderr}")
#         return False

# def calculate_gc_content(sequence):
#     """
#     Calculates the GC content percentage of a DNA sequence.
#     """
#     g = sequence.count('G') + sequence.count('g')
#     c = sequence.count('C') + sequence.count('c')
#     return (g + c) / len(sequence) * 100 if len(sequence) > 0 else 0

# def extract_cpg_islands(sequence, window=100, threshold=0.6):
#     """
#     Identifies CpG islands in the DNA sequence.
#     Returns a list of start positions where CpG islands begin.
#     """
#     cpg_islands = []
#     for i in range(0, len(sequence) - window + 1, window):
#         window_seq = sequence[i:i + window]
#         gc_content = calculate_gc_content(window_seq)
#         cpg_ratio = window_seq.count("CG") / window if window > 0 else 0
#         if gc_content > 50 and cpg_ratio > 0.6:
#             cpg_islands.append(i)
#     return cpg_islands

# def plot_gc_content(sequence, window=100):
#     """
#     Plots the GC content across the DNA sequence.
#     """
#     gc_contents = [calculate_gc_content(sequence[i:i + window]) for i in range(0, len(sequence) - window + 1, window)]
#     plt.figure(figsize=(10, 5))
#     plt.plot(gc_contents, color='green')
#     plt.title('GC Content Across DNA Sequence')
#     plt.xlabel('Window Number')
#     plt.ylabel('GC Content (%)')
#     plt.grid(True)
#     plt.savefig('gc_content_plot.png')
#     plt.show()
#     logging.info("GC content plot saved as 'gc_content_plot.png'.")

# if __name__ == "__main__":
#     import shutil
#     from data_collection import fetch_dna_sequence, fetch_multiple_sequences
    
#     # Fetch multiple DNA sequences
#     accession_ids = ["NM_007294", "NM_000059", "NM_000546"]  # Example accession IDs
#     sequences = fetch_multiple_sequences(accession_ids)
    
#     # Create SeqRecords
#     seq_records = [SeqRecord(Seq(seq), id=acc_id, description=f"{acc_id} Sequence") for acc_id, seq in sequences.items()]
    
#     # Save sequences to FASTA
#     input_fasta = "brca_sequences.fasta"
#     with open(input_fasta, "w") as f:
#         SeqIO.write(seq_records, f, "fasta")
#     logging.info(f"Saved {len(seq_records)} sequences to {input_fasta}.")
    
#     # Perform alignment
#     alignment_success = perform_alignment(input_fasta)
    
#     if alignment_success:
#         # Parse aligned sequences
#         aligned_sequences = list(SeqIO.parse("aligned_sequences.fasta", "fasta"))
#         logging.info(f"Aligned {len(aligned_sequences)} sequences.")
    
#     # Calculate GC content for each sequence
#     for record in seq_records:
#         gc = calculate_gc_content(record.seq)
#         logging.info(f"{record.id} GC Content: {gc:.2f}%")
    
#     # Plot GC content distribution
#     plot_gc_content(sequences["NM_007294"])  # Example for BRCA1
    
#     # Extract and log CpG islands
#     cpg_islands = extract_cpg_islands(sequences["NM_007294"])
#     logging.info(f"Found {len(cpg_islands)} CpG islands in BRCA1.")


from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import subprocess
import shutil
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# feature_extraction.py
def perform_alignment(input_fasta, output_fasta="aligned_sequences.fasta"):
    if not shutil.which("mafft"):
        logging.error("MAFFT is not installed or not found in your PATH.")
        return False

    sequences = list(SeqIO.parse(input_fasta, "fasta"))
    if len(sequences) < 2:
        logging.warning("Alignment skipped: At least two sequences are required.")
        return False
    
    # Proceed with alignment
    try:
        with open(output_fasta, 'w') as output_handle:
            subprocess.run(["mafft", "--auto", input_fasta], stdout=output_handle, check=True)
        logging.info("Alignment successful!")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Error occurred during alignment: {e.stderr}")
        return False

def calculate_gc_content(sequence):
    g = sequence.count('G')
    c = sequence.count('C')
    return (g + c) / len(sequence) * 100

def extract_cpg_islands(sequence):
    return [i for i in range(len(sequence) - 1) if sequence[i:i + 2] == "CG"]

if __name__ == "__main__":
    from data_collection import fetch_dna_sequence
    dna_sequence = fetch_dna_sequence("NM_007294")
    
    seq_record = SeqRecord(Seq(dna_sequence), id="BRCA1", description="BRCA1 Gene")
    dummy_seq = SeqRecord(Seq("ATGCGTACGTAGCTAGCTAGCTAGCTAGC"), id="Dummy", description="Dummy Sequence")
    
    with open("brca1_sequences.fasta", "w") as output_handle:
        SeqIO.write([seq_record, dummy_seq], output_handle, "fasta")
    
    alignment_success = perform_alignment("brca1_sequences.fasta")
    
    if alignment_success:
        gc_content = calculate_gc_content(dna_sequence)
        cpg_islands = extract_cpg_islands(dna_sequence)
        logging.info(f"GC Content: {gc_content:.2f}%")
        logging.info(f"Number of CpG Islands: {len(cpg_islands)}")
