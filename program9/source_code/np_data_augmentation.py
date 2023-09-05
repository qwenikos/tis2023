def reverse_complement(sequence):
    complement_dict = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C','N': 'N' }
    reverse_sequence = sequence[::-1]
    reverse_complement_sequence = ''.join(complement_dict[base] for base in reverse_sequence)
    return reverse_complement_sequence


def reverse_complement_fasta(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file , 'w') as outfile:
        current_sequence = ''
        for line in infile:
            line = line.strip()
            if line.startswith('>'):
                # This is a header line, write it as is
                if current_sequence:
                    # Write the reverse complement of the previous sequence
                    rc_sequence = reverse_complement(current_sequence)
                    outfile.write(rc_sequence + '\n')
                    current_sequence = ''
                outfile.write(line + '\n')
            else:
                # Accumulate the DNA sequence
                current_sequence += line
        # Write the reverse complement of the last sequence
        if current_sequence:
            rc_sequence = reverse_complement(current_sequence)
            outfile.write(rc_sequence + '\n')

def concatenate_files(file1_path, file2_path, output_path):
    try:
        with open(file1_path, 'r') as file1, open(file2_path, 'r') as file2, open(output_path, 'w') as output_file:
            for line in file1:
                output_file.write(line)
            for line in file2:
                output_file.write(line)
        print(f"Concatenated '{file1_path}' and '{file2_path}' into '{output_path}'.")
    except FileNotFoundError:
        print("One or more input files not found.")
            

train_pos="../datasets/training/positive/positive_trainingSet_Flank-100.fa"
train_neg="../datasets/training/negative/negative_trainingSet_Flank-100.fa"
reverse_compl_train_pos="../datasets/training/positive/reverse_compl_positive_trainingSet_Flank-100.fa"
reverse_compl_train_neg="../datasets/training/negative/reverse_compl_negative_trainingSet_Flank-100.fa"
both_train_pos="../datasets/training/positive/both_positive_trainingSet_Flank-100.fa"
both_train_neg="../datasets/training/negative/both_negative_trainingSet_Flank-100.fa"
reverse_complement_fasta(train_pos,reverse_compl_train_pos )
reverse_complement_fasta(train_neg,reverse_compl_train_neg )

concatenate_files(train_pos, reverse_compl_train_pos, both_train_pos)
concatenate_files(train_neg,reverse_compl_train_neg , both_train_neg)





