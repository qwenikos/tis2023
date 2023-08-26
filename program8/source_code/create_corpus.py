import sys
import argparse

from misc import clean_sequences, subsequences, k_mers, read_fasta_file


def main():
    parser = argparse.ArgumentParser(description="""Construction of corpus""")
    parser.add_argument("--file", help="Path of the file that contains gene sequences")
    # parser.add_argument("--type", help="Type of the genes, i.e positive or negative ")
    parser.add_argument("--subregion", default=None, help="Starting and finishing positions of the selected subregion of a gene")
    parser.add_argument("--k", help="k value for k-mers")
    parser.add_argument("--overlapping", default='non-overlapping', choices=['overlapping', 'non-overlapping'], help="overlapping k-mers")

    args = parser.parse_args()
    
    file = args.file
    # typegenes = args.type
    subregion = args.subregion

    start_point = 60
    end_point = 120

    if args.subregion != None:
        subregion = args.subregion.split(",")

        if len(subregion) < 2 or subregion[1] == '':
            sys.exit('The given arguments for the flag --subregion are incorrect')
        else:
            start_point = int(subregion[0])
            end_point = int(subregion[1])


    k = args.k
    k = int(k)

    overlapping = args.overlapping

    if file == None:
        sys.exit('Please give the path of the file that contains the genes sequences')

    # if typegenes == None:
    #     sys.exit('Please specify the type of the file (positive or negative)')

    genes = read_fasta_file(file)

    subseqs, window_size = subsequences(genes, [start_point, end_point], overlapping, k)


    subseqs = clean_sequences(subseqs, window_size)

    file = 'corpus_text.txt'

    f = open(file,'w')

    for subseq in subseqs.T:

        kmers = k_mers(subseq, k, overlapping)

        for kmer in kmers:
            f.write(kmer)
            f.write(' ')

    f.write('\n')

    f.close()


if __name__ == '__main__':
	main()