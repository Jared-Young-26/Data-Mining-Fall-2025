"""
Apriori Algorithm Implementation in Python
Author: Jared Young
Course: Fall 2025 DATA MINING TECHNIQUES (CS-63015-001)
Notes: Implemented in Python using sets, frozensets, and itertools.
Comments included to explain the data structures and algorithim logic step-by-step.
Utilized Python documentation for itertools, class slides, pseudocode, and external tutorials for assistance in this project.
Verified on dataset 'data.txt' with support thresholds 10%, 20%, 30%, and 50%
"""
# Import the following modules for system parameters, generating combinations, and calculating the ceiling value
import sys
from math import ceil
from itertools import combinations

# Input parameters: min_support, transactions
# Output: frequent itemsets

def load_transactions(filename):
    """ Load transactions from a text file.
    Each line is split into individual items and stored as a set."""
    
    transactions = []
    with open(filename, 'r') as file:
        for line in file:
            items = line.strip().split()
            transactions.append(set(items))
    return transactions

def apriori(transactions, min_support_percent):
    """Run the Apriori algorithim on a list of transactions.
    Returns a list of frequent itemsets and their support counts."""
    
    # Define the thresholds
    total_transactions = len(transactions)
    min_support_count = ceil((int(min_support_percent) * total_transactions)/ 100.0)
    
    # Count the items for F1
    items_count = {} # Initialize a hash map for the item counts to be stored
    for transaction in transactions: # Iterate through transactions
        for item in transaction: # Iterate through each item in each of the transactions
            key = frozenset([item]) # Set has to be immutable to make it a key
            items_count[key] = items_count.get(key, 0) + 1 # If the key exists, add 1; Otherwise, initialize to 0 and increment 1

    # Filter out the items that do not meet the minimum support count
    supported_data = {itemset: count for itemset, count in items_count.items() if count >= min_support_count}
    F1 = set(supported_data.keys()) # F1 is the set of all frequent 1-itemsets

    # Create a container to hold all of the frequent sets and initialize the first iteration of frequent items
    frequent_sets = []
    if F1: # If the F1 set is not empty
        frequent_sets.append(F1) # Add it to the list of frequent sets

    k = 2 # Start with k = 2
    F0 = F1 # Make the iteration of F1 the new initial set to iterate through
    while F0: # While the set isn't empty
        
        # Generate the new list of candidates from the last set of frequent items using combinations
        Ck = set() # Initialize an empty set for generated candidates to be stored
        for set1, set2 in combinations(F0, 2): # Iterate through all pairs of sets in F0
            candidate_set = set1.union(set2) # Join the two sets to create a new candidate
            if len(candidate_set) == k: # If the new candidate is equal to the value of k
                Ck.add(candidate_set) # Add that candidate to the k-candidate set
        
        # Check for pruning in the candidate pool
        pruned_Ck = set() # Initialize an empty set for valid candidates to be stored
        for candidate in Ck: # For each candidate in the generated candidates
            subset_frequent = True # Assume the candidate is a valid subset 
            for subset in combinations(candidate, k-1): # generate all of the (k-1)-subsets
                if frozenset(subset) not in F0: # If the subset is not in the list of valid candidates
                    subset_frequent = False # It is not a frequent subset
                    break # Quit processing and prune the candidate
            if subset_frequent: # If the candidate is a frequent subset 
                pruned_Ck.add(candidate) # Add it to the list of candidates that survived pruning

        Ck = pruned_Ck # New candidate set post-pruning

        count_candidates = {c: 0 for c in Ck} # Initialize a dictionary with keys for all Ck candidates
        for transaction in transactions: # Scan through the transaction data
            for candidate in Ck: # Iterate through each candidate in the pruned set
                if candidate.issubset(transaction): # If the candidate is a subset of the transaction 
                    count_candidates[candidate] += 1 # Increase the count
        
        # Create the next k-itemset for testing based on the min_support of candidates in Ck
        Fk = {candidate for candidate, count in count_candidates.items() if count >= min_support_count}
        
        for candidate in Fk: # For each candidate that exceeded the min_support threshold
            supported_data[candidate] = count_candidates[candidate] # Update the counts for the candidate pool

        if not Fk: # If the k-itemset is empty
            break # Done calculating k-itemsets, return the values

        frequent_sets.append(Fk) # Otherwise, add the current k-itemset to the list of all frequent sets
        F0 = Fk # Make the k-itemset the new initial set
        k += 1 # Increment k
    return frequent_sets, supported_data # Return the frequent sets and their associated count values



# Main Function 
if __name__ == "__main__":
   transactions = load_transactions(sys.argv[1]) # Loads the file containing transactions from the first position after the program file is called
   min_support_percent = sys.argv[2] # Takes in the desired minimum support percentage as a whole number
   frequent_sets, supported_data = apriori(transactions, min_support_percent) # Finds all the frequent itemsets based on the support threshold
   
   # Iterate through the outputs in an organized format
   for k, Fk in enumerate(frequent_sets, start=1): # Going through each k-itemset 
       print(f"\nFrequent {k}-itemsets:") # Shows the frequent k-itemset
       for item in Fk: # Iterates through each item showing the set & its counted value
           print(f"Item: {sorted(item)}, Count: {supported_data[item]}")
    