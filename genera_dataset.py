from Bio import Entrez, SeqIO
Entrez.email = "a.ture@studenti.unisa.it"
# esempio: cromosoma di E. coli
handle = Entrez.efetch(db="nucleotide",
                       id="NC_000913.3",
                       rettype="fasta",
                       retmode="text")
records = SeqIO.parse(handle, "fasta")
SeqIO.write(records, "ecoli_ref.fasta", "fasta")
handle.close()
