from sys import argv
import json

def parse_fasta(fh):
    """Parses a RefSeq fasta file.
    Fasta headers are expected to be like:
    >NR_165790.1 Tardibacter chloracetimidivorans strain JJ-A5 16S ribosomal RNA, partial sequence

    Args:
      fh: filehandle to the fasta file

    Returns:
      A seq_dict like {'name': ['seq1', 'seq2'], etc. }
    """
    fasta_dict = {}
    current_species = ''
    for line in fh:
        line = line.strip()
        if line.startswith('>'): # it's a header line
            line_parts = line.split()
            current_species = line_parts[1] + ' ' + line_parts[2]
            # Add current species to dict if it doesn't exist
            if not current_species in fasta_dict:
                fasta_dict[current_species] = ['']
            else:
                # add a new entry for this species
                fasta_dict[current_species].append('')
        else: # Sequence line
            fasta_dict[current_species][-1] += line
    return fasta_dict

def print_fasta_stats(seq_dict):
    """Prints some statistics of a sequence dictionary.

    Args:
      seq_dict: dictionary of structure {'name': ['seq1', 'seq2'], etc. }

    Side Effect:
      Prints to standard out.
    """
    print('PARSED NUM SPECIES: ', len(seq_dict))
    for key in sorted(seq_dict.keys()):
        lengths = ''
        # for seq in seq_dict[key]:
            # lengths += str(len(seq)) + ' '
        print(key, len(seq_dict[key]), lengths)


def generate_all_options(seq_dict, window_length=1200):
    """Generates all possible sequence options for every entry in a seq_dict.
    Calculates according to a moving window of size window_length, like:
    'BAAAC' -> '[BAAA]C' + 'B[AAAC] -> ['BAAA', 'AAAC']
    
    Args:
      seq_dict: A sequence dict from parse_fasta
      window_length: int, size of the moving window
    Returns:
      A seq_dict but with all possible options
    """
    all_sequence_dict = {}
    for species_key in fasta_dict:
        all_sequence_dict[species_key] = []
        for sequence in fasta_dict[species_key]:
            seq_length = len(sequence)
            if seq_length < window_length:  # can't get window
                continue
            num_windows = seq_length - window_length + 1
            for win_start in range(num_windows):
                win_end = win_start + window_length
                window = sequence[win_start:win_end]
                all_sequence_dict[species_key].append(window)
    return all_sequence_dict

if __name__ == "__main__":
    input_fh = open(argv[1])
    output_fh = open(argv[2], 'w')
    print('Reading from', argv[1])

    fasta_dict = parse_fasta(input_fh)
    print_fasta_stats(fasta_dict)

    all_options = generate_all_options(fasta_dict)
    json.dump(all_options, output_fh)
    print('Done writing to', argv[2])
