from __future__ import print_function
import numpy as np
import pandas as pd
import re

#generate a pandas dataframe from an alignment file
def pdataframe_from_alignment_file(filename,num_reads=200000):    
    data=pd.DataFrame(columns=["name","sequence"])
    with open(filename) as datafile:
        serotype=""
        sequence=""
        dump=False
        count=0
        dataf=datafile.readlines()
        for line in dataf:
            if line.startswith(">"):
                if count>=num_reads:
                    break
                if dump:
                    row=pd.DataFrame([[serotype,sequence]],columns=["name","sequence"])
                    data=data.append(row,ignore_index=True)
                dump=True
                serotype=line[1:].strip("\n")
                sequence=""
                count+=1
            else:
                sequence+=line.strip("\n")
        row=pd.DataFrame([[serotype,sequence]],columns=["name","sequence"])
        data=data.append(row,ignore_index=True)
    return data


#Find the indices of aligned columns, Note that indices are 0-indexed
def index_of_non_lower_case_dot(sequence):
    output=[]
    for s in range(len(sequence)):
        if sequence[s]!="." and not (sequence[s].islower()):
            output.append(s)
    return output

#Drop columns that are not part of the alignment
def prune_seq(sequence):
    output=""
    for s in sequence:
        if s!="." and not (s.islower()):
            output+=s
    return output

#Helper function to translate string to one_hot
def translate_string_to_one_hot(sequence,order_list):
    out=np.zeros((len(order_list),len(sequence)))
    for i in range(len(sequence)):
        out[order_list.index(sequence[i])][i]=1
    return out

#generate single mutants for those positions that experimental data and alignment are available
def mutate_single(wt,mutation_data,offset=0,index=0):
    mutants=[]
    prev=int(mutation_data[0][1])-offset
    for md in mutation_data:
        if prev!=int(md[1])-offset:
            index+=1
            prev=int(md[1])-offset 
        mutant=[md[2] if i==index else wt[i] for i in range (len(wt))]
        mutants.append(mutant)
    return mutants


#Invariants
ORDER_KEY="XILVAGMFYWEDQNHCRKSTPBZ-"[::-1]
ORDER_LIST=list(ORDER_KEY)
print(ORDER_LIST)

data = pdataframe_from_alignment_file('BLAT.a2m')
print(data.head())
print("number of data points:", len(data))
print("length of sequence:", len(data.iloc[0]["sequence"]))
print("sample sequence:", data.iloc[0]["sequence"])

#Indices that align (at least 50% of sequences are not gaps)
indices = index_of_non_lower_case_dot(data.iloc[0]["sequence"])
data["seq"] = list(map(prune_seq, data["sequence"]))
PRUNED_SEQ_LENGTH = len(data.iloc[0]["seq"])
print("pruned sequence length:", PRUNED_SEQ_LENGTH)

#get one-hot encoded sequences for calculating sequence weights
training_data_one_hot=[]
for i, row in data.iterrows():
    training_data_one_hot.append(translate_string_to_one_hot(row["seq"],ORDER_LIST))
print(len(training_data_one_hot))

training_data = np.array([np.array(list(sample.T.flatten())) for sample in training_data_one_hot])
train_data_one_hot = training_data.reshape([len(training_data), PRUNED_SEQ_LENGTH, 24])
print('train_data_one_hot:',train_data_one_hot.shape)

exp_data_full=pd.read_csv("betalactamase.csv", sep=",")
print ("number of mutants:", len(exp_data_full))
print(exp_data_full.head())

OFFSET=24
#Deciding offset requires investigating the dataset and alignment.
exp_data_singles=pd.DataFrame(columns=exp_data_full.columns)
#decide starting index depending on how the file is "headered"
for i,row in exp_data_full[0:].iterrows():
    pos=re.split(r'(\d+)', row.mutant) 
    if int(pos[1])-OFFSET in indices:
        exp_data_singles=exp_data_singles.append(row)
exp_data_singles=exp_data_singles.reset_index()
target_values=list(exp_data_singles["linear"])
print('number of target values:', len(target_values))

mutation_data=[re.split(r'(\d+)', s) for s in exp_data_singles.mutant]
wt_sequence=data.iloc[0].seq
mutants=mutate_single(wt_sequence,mutation_data,offset=0,index=0)
#sanity checks
print(len(mutants),len(exp_data_singles))
#the mutant should be in the correct place
print(list(zip(wt_sequence,mutants[0]))[:10])
print(list(zip(wt_sequence,mutants[1]))[:10])

#Test data with wt at 0 index
one_hot_mutants=[]
mutants_plus=mutants
for mutant in mutants:
    one_hot_mutants.append(translate_string_to_one_hot("".join(mutant),ORDER_LIST))

test_data_plus=np.array([np.array(list(sample.T.flatten())) \
                         for sample in one_hot_mutants])
print(test_data_plus.shape)
#print(test_data_plus[0])
test_data_one_hot = test_data_plus.reshape([len(test_data_plus), PRUNED_SEQ_LENGTH, 24])
print('test_data_one_hot:',test_data_one_hot.shape)

np.save('train_data', train_data_one_hot, allow_pickle=False)
np.save('test_data', test_data_one_hot, allow_pickle=False)
np.save('target_values', target_values, allow_pickle=False)

print(np.load('train_data.npy'))
print(np.load('test_data.npy'))
print(np.load('target_values.npy'))