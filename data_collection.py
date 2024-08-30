from Bio import Entrez, SeqIO

def fetch_dna_sequence():
    Entrez.email = "sushan.adhikari2060@gmail.com"
    handle = Entrez.efetch(db="nucleotide", id="NM_007294", rettype="gb", retmode="text")
    record = SeqIO.read(handle, "genbank")
    handle.close()
    return str(record.seq)

if __name__ == "__main__":
    dna_sequence = fetch_dna_sequence()
    print("DNA Sequence Length:", len(dna_sequence))
