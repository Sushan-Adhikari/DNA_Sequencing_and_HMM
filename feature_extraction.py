from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align.Applications import ClustalwCommandline

def perform_alignment(input_fasta):
    clustalw_exe = r"clustalw2"  # Adjust the path if clustalw is not in your PATH
    clustalw_cline = ClustalwCommandline(clustalw_exe, infile=input_fasta)
    stdout, stderr = clustalw_cline()
    print(stdout)

def calculate_gc_content(sequence):
    g = sequence.count('G')
    c = sequence.count('C')
    return (g + c) / len(sequence) * 100

def extract_cpg_islands(sequence):
    cpg_islands = [i for i in range(len(sequence) - 1) if sequence[i:i+2] == "CG"]
    return cpg_islands

if __name__ == "__main__":
    # Load DNA sequence
    from data_collection import fetch_dna_sequence
    dna_sequence = fetch_dna_sequence()
    
    # Create SeqRecord
    seq_record = SeqRecord(Seq(dna_sequence), id="BRCA1", description="BRCA1 Gene")
    
    # Save to FASTA for alignment
    with open("brca1_sequences.fasta", "w") as output_handle:
        SeqIO.write(seq_record, output_handle, "fasta")
    
    # Perform alignment
    perform_alignment("brca1_sequences.fasta")

    # Extract features
    gc_content = calculate_gc_content(dna_sequence)
    cpg_islands = extract_cpg_islands(dna_sequence)
    
    print("GC Content: {:.2f}%".format(gc_content))
    print("Number of CpG Islands:", len(cpg_islands))
