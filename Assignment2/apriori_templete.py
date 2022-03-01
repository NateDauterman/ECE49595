from __future__ import print_function
import sys



def apriori(dataset, min_support=0.5, verbose=False):
    """Implements the Apriori algorithm.

    The Apriori algorithm will iteratively generate new candidate
    k-itemsets using the frequent (k-1)-itemsets found in the previous
    iteration.

    Parameters
    ----------
    dataset : list
        The dataset (a list of transactions) from which to generate
        candidate itemsets.

    min_support : float
        The minimum support threshold. Defaults to 0.5.

    Returns
    -------
    F : list
        The list of frequent itemsets.

    support_data : dict
        The support data for all candidate itemsets.

    References
    ----------
    .. [1] R. Agrawal, R. Srikant, "Fast Algorithms for Mining Association
           Rules", 1994.

    """
    C1 = create_candidates(dataset)
    D = list(map(set, dataset))
    F1, support_data = get_freq(D, C1, min_support, verbose=False) # get frequent 1-itemsets
    F = [F1] # list of frequent itemsets; initialized to frequent 1-itemsets
    k = 2 # the itemset cardinality
    while (len(F[k - 2]) > 0):
        Ck = apriori_gen(F[k-2], k) # generate candidate itemsets
        Fk, supK  = get_freq(D, Ck, min_support) # get frequent itemsets
        support_data.update(supK)# update the support counts to reflect pruning
        F.append(Fk)  # add the frequent k-itemsets to the list of frequent itemsets
        k += 1

    if verbose:
        # Print a list of all the frequent itemsets.
        for kset in F:
            for item in kset:
                print(""                     + "{"                     + "".join(str(i) + ", " for i in iter(item)).rstrip(', ')                     + "}"                     + ":  sup = " + str(round(support_data[item], 3)))

    return F, support_data

def create_candidates(dataset, verbose=False):
    """Creates a list of candidate 1-itemsets from a list of transactions.

    Parameters
    ----------
    dataset : list
        The dataset (a list of transactions) from which to generate candidate
        itemsets.

    Returns
    -------
    The list of candidate itemsets (c1) passed as a frozenset (a set that is
    immutable and hashable).
    """
    c1 = [] # list of all items in the database of transactions
    for transaction in dataset:
        for item in transaction:
            if not [item] in c1:
                c1.append([item])
    c1.sort()

    if verbose:
        # Print a list of all the candidate items.
        print(""             + "{"             + "".join(str(i[0]) + ", " for i in iter(c1)).rstrip(', ')             + "}")

    # Map c1 to a frozenset because it will be the key of a dictionary.
    return list(map(frozenset, c1))

def get_freq(dataset, candidates, min_support, verbose=False):
    """

    This function separates the candidates itemsets into frequent itemset and infrequent itemsets based on the min_support,
	and returns all candidate itemsets that meet a minimum support threshold.

    Parameters
    ----------
    dataset : list
        The dataset (a list of transactions) from which to generate candidate
        itemsets.

    candidates : frozenset
        The list of candidate itemsets.

    min_support : float
        The minimum support threshold.

    Returns
    -------
    freq_list : list
        The list of frequent itemsets.

    support_data : dict
        The support data for all candidate itemsets.
    """

    min_sup = len(dataset) * min_support
    #print(min_sup)
    #print(candidates)
    #print(dataset)

    support_data = {}
    freq_list = []

    for cand in candidates:
        support = 0
        for data in dataset:
            if cand.issubset(data):
                support += 1
        if support >= min_sup:
            freq_list.append(cand)
        support_data[cand] = support

    #print(freq_list)
    #print(support_data)

    #if verbose:

    print("Candidate:")
    print(len(candidates))
    print()
    print("frequent list")
    print(len(freq_list))
    print()

    return freq_list, support_data



def apriori_gen(freq_sets, k):
    """Generates candidate itemsets (via the F_k-1 x F_k-1 method).

    This part generates new candidate k-itemsets based on the frequent
    (k-1)-itemsets found in the previous iteration.

    The apriori_gen function performs two operations:
    (1) Generate length k candidate itemsets from length k-1 frequent itemsets
    (2) Prune candidate itemsets containing subsets of length k-1 that are infrequent

    Parameters
    ----------
    freq_sets : list
        The list of frequent (k-1)-itemsets.

    k : integer
        The cardinality of the current itemsets being evaluated.

    Returns
    -------
    candidate_list : list
        The list of candidate itemsets.
    """


    ##  CANDIDATE LIST GENERATION

    #for i in range(len(freq_sets)):
    #    for j in range[]

    #print(freq_sets)
    candidate_list = []
    #print(freq_sets)
    if k == 2:
        for i in range(len(freq_sets)):
            for j in range(i + 1, len(freq_sets)):
                candidate_list.append(freq_sets[i] | freq_sets[j])
    else:
        for i in range(len(freq_sets)):
            for j in range(i + 1, len(freq_sets)):
                one = sorted(list(freq_sets[i]))
                two = sorted(list(freq_sets[j]))
                #print(one[:k - 2], two[:k - 2])
                #print()

                if one[:k - 2] == two[:k - 2]:
                    candidate_list.append(freq_sets[i] | freq_sets[j])


    ## CANDIDATE LIST PRUNING
    remove = []

    #print(freq_sets)
    #print(candidate_list)

    for new_set in candidate_list:
        unfrozenset = set(new_set)
        #print(list(unfrozenset))
        for item in list(unfrozenset):
            unfrozenset.discard(item)

            #print(unfrozenset)

            if unfrozenset not in freq_sets:
                #print(candidate_list)
                #print(new_set)
                #print('found')
                if new_set in candidate_list:
                    remove.append(new_set)
            unfrozenset.add(item)

    for item in remove:
        candidate_list.remove(item)

        #print(candidate_list)
        # remove = []
        # for i in range(len(candidate_list)):
        #     for freq in freq_sets:
        #         #print(freq)
        #         #print(candidate_list[i])
        #         #print(not freq.issubset(candidate_list[i]))
        #         if freq.issubset(candidate_list[i]):
        #             break
        #
        #
        #
        # remove = sorted(set(remove), reverse=True)
        # #print(remove)
        # for i in remove:
        #     candidate_list.pop(i)

    #print(candidate_list)

    return candidate_list


def loadDataSet(fileName, delim=','):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    return stringArr



def run_apriori(data_path, min_support, verbose=False):
    dataset = loadDataSet(data_path)
    F, support = apriori(dataset, min_support=min_support, verbose=verbose)
    return F, support



def bool_transfer(input):
    ''' Transfer the input to boolean type'''
    input = str(input)
    if input.lower() in ['t', '1', 'true' ]:
        return True
    elif input.lower() in ['f', '0', 'false']:
        return False
    else:
        raise ValueError('Input must be one of {T, t, 1, True, true, F, F, 0, False, false}')




if __name__ == '__main__':
    if len(sys.argv)==3:
        F, support = run_apriori(sys.argv[1], float(sys.argv[2]))
    elif len(sys.argv)==4:
        F, support = run_apriori(sys.argv[1], float(sys.argv[2]), bool_transfer(sys.argv[3]))
    else:
        raise ValueError('Usage: python apriori_templete.py <data_path> <min_support> <is_verbose>')
    print(len(F))
    print(len(support))

    '''
    Example:

    python apriori_templete.py market_data_transaction.txt 0.5

    python apriori_templete.py market_data_transaction.txt 0.5 True

    '''
