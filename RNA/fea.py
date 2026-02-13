import numpy as np
import pandas as pd

def construct_kmer():
	ntarr = ("A","C","G","T")

	kmerArray = []


	for n in range(4):
		kmerArray.append(ntarr[n])

	for n in range(4):
		str1 = ntarr[n]
		for m in range(4):
			str2 = str1 + ntarr[m]
			kmerArray.append(str2)
#############################################
	for n in range(4):
		str1 = ntarr[n]
		for m in range(4):
			str2 = str1 + ntarr[m]
			for x in range(4):
				str3 = str2 + ntarr[x]
				kmerArray.append(str3)
#############################################
#change this part for 3mer or 4mer
	for n in range(4):
		str1 = ntarr[n]
		for m in range(4):
			str2 = str1 + ntarr[m]
			for x in range(4):
				str3 = str2 + ntarr[x]
				for y in range(4):
					str4 = str3 + ntarr[y]
					kmerArray.append(str4)
############################################
	for n in range(4):
		str1 = ntarr[n]
		for m in range(4):
			str2 = str1 + ntarr[m]
			for x in range(4):
				str3 = str2 + ntarr[x]
				for y in range(4):
					str4 = str3 + ntarr[y]
					for z in range(4):
						str5 = str4 + ntarr[z]
						kmerArray.append(str5)
####################### 6-mer ##############
	for n in range(4):
		str1 = ntarr[n]
		for m in range(4):
			str2 = str1 + ntarr[m]
			for x in range(4):
				str3 = str2 + ntarr[x]
				for y in range(4):
					str4 = str3 + ntarr[y]
					for z in range(4):
						str5 = str4 + ntarr[z]
						for t in range(4):
							str6 = str5 + ntarr[t]
							kmerArray.append(str6)
    
	return kmerArray

def kmer_encode(seq,kmerarray):
    result = np.zeros((len(seq),len(kmerarray)))
    for i in range(len(seq)):
        for j in range(len(kmerarray)):
            result[i,j] = seq[i].count(kmerarray[j])/len(seq[i])
    return result

def mer_sin(seq, nc_m, c_m, kmerarray, x):   
    l = len(seq) - x + 1
    if l <= 0:
        return 0.0
    log_r = np.zeros((l))
    for i in range(l):
        tempseq = seq[i:i+x]        
        if 'N' in tempseq:          
            log_r[i] = 0
            continue
        idx = kmerarray.index(tempseq)
        Fc = c_m[int(idx)]
        Fnc = nc_m[int(idx)]
        if Fc == 0 and Fnc == 0:
            log_r[i] = 0
        elif Fc == 0 and Fnc != 0:
            log_r[i] = -1
        elif Fnc == 0 and Fc != 0:
            log_r[i] = 1
        else:
            log_r[i] = np.log(Fc / Fnc)
    miu = sum(log_r) / l
    return miu
   
def mer_score(seq,nc_m,c_m,kmerarray,x):

    miu = np.zeros((len(seq)))
    for i in range(len(seq)):
        miu[i] = mer_sin(seq[i],nc_m,c_m,kmerarray,x)
        
    miu0 = np.expand_dims(miu, axis=1) 
    return miu0
    
def generate_features(sequences):

    # ---- Load static reference matrices (log values) ----
    pos = pd.read_csv('embed/mer_rnapos_mean.csv', header=None).values
    neg = pd.read_csv('embed/mer_rnaneg_mean.csv', header=None).values
    
    kmerArray = construct_kmer()

    # ---- k-mer encodings ----
    kmer1 = kmer_encode(sequences, kmerArray[0:4])      # 4 features
    kmer2 = kmer_encode(sequences, kmerArray[4:20])     # 16 features
    kmer3 = kmer_encode(sequences, kmerArray[20:84])    # 64 features
        
    # ---- mer scores ----
    merscore1 = mer_score(sequences, pos[0:4], neg[0:4], kmerArray[0:4], 1)
    print('merscore1')
    merscore2 = mer_score(sequences, pos[4:20], neg[4:20], kmerArray[4:20], 2)
    print('merscore2')
    merscore3 = mer_score(sequences, pos[20:84], neg[20:84], kmerArray[20:84], 3)
    print('merscore3')
    merscore4 = mer_score(sequences, pos[84:340], neg[84:340], kmerArray[84:340], 4)
    print('merscore4')
    merscore5 = mer_score(sequences, pos[340:1364], neg[340:1364], kmerArray[340:1364], 5)
    print('merscore5')
    merscore6 = mer_score(sequences, pos[1364:5460], neg[1364:5460], kmerArray[1364:5460], 6)
    print('merscore6')

    return merscore1, merscore2, merscore3, merscore4, merscore5, merscore6,kmer1, kmer2, kmer3