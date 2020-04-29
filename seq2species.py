import numpy as np
import pickle
from sys import argv

def parse_fasta(fh):
    # >NR_165790.1 Tardibacter chloracetimidivorans strain JJ-A5 16S ribosomal RNA, partial sequence
    # ACTG
    # TGAC
    # >New header like 0
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

def encode_base_string(base_string):
    # One-hot encode base string
    # returns [[0, 0, 0, 1], [1, 0, 0, 0]]
    bases = []
    for raw_base in base_string:
        # Could be sped up
        if raw_base == 'A':
            bases.append([1, 0, 0, 0])
        elif raw_base == 'C':
            bases.append([0, 1, 0, 0])
        elif raw_base == 'T':
            bases.append([0, 0, 1, 0])
        elif raw_base == 'G':
            bases.append([0, 0, 0, 1])
        else:
            bases.append([0, 0, 0, 0])
    return bases

def one_hot_species(species):
    encoding_table = list(set(species)) # not memory-efficient
    one_hotted_species = []
    for species_name in species:
        one_hot_species_name = [0] * len(encoding_table)
        species_index = encoding_table.index(species_name)
        one_hot_species_name[species_index] = 1
        one_hotted_species.append(one_hot_species_name)
    return np.array(one_hotted_species)

def create_dataset(fasta_dict):
    sequences = []
    species = []
    for species_name in fasta_dict:
        for sequence in fasta_dict[species_name]:
            # Test set
            sequence = sequence[:1300]
            one_hot_sequence = encode_base_string(sequence)
            species.append(species_name)
            sequences.append(one_hot_sequence)
    # One-hot encode the species (labels) as well
    species = one_hot_species(species)
    return np.array(sequences), species

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=2, kernel_size=1, input_shape=(1300, 4)),
        tf.keras.layers.LeakyReLU(alpha=0.05),
        tf.keras.layers.Dense(107, activation='softmax')
    ])
    model.compile(
        optimizer='adam', # stochastic gradient descent method that just works well
        # easiest to use loss function sparse_categorical_crossentropy, easiest to understand mean_squared_error
        loss='mean_squared_error',
        metrics=['accuracy'] # Log accuracy
    )
    return model


fasta_dict = parse_fasta(open(argv[1]))
print('LENGTH', len(fasta_dict))
for key in sorted(fasta_dict.keys()):
    lengths = ''
    for seq in fasta_dict[key]:
        lengths += str(len(seq)) + ' '
    print(key, len(fasta_dict[key]), lengths)

exit(0)
import tensorflow as tf

print('STARTING to learn NN')
fasta_dict = parse_fasta(open('./part_bacteria_rrna.fna'))
train_sequences, train_labels = create_dataset(fasta_dict)

train_sequences = train_sequences.astype(float)
train_labels = train_labels.astype(float)
print(train_sequences.shape)
print(train_labels.shape)
# (107, 1300, 4)
# (107, 107)
model = create_model()
model.fit(train_sequences, train_labels, epochs=10)
    