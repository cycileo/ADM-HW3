# ========================================================
#                 SECOND EXERCISE: SEARCH ENGINE
# ========================================================

from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from nltk.stem import SnowballStemmer
import re
import json
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import pandas as pd

# Function to preprocess restaurant descriptions by removing stopwords, cleaning punctuation, and applying stemming to improve search efficiency.
def preprocess_text(descriptions):
    """
    Preprocesses a list of restaurant descriptions by:
    - Tokenizing each description
    - Removing stopwords
    - Cleaning tokens of punctuation
    - Stemming each word to its root form
    
    Args:
    descriptions (list of str): List of text descriptions to preprocess
    
    Returns:
    list of list of str: A list where each element is a list of processed tokens for a description
    """
    
    processed_descriptions = []  # Holds the final processed tokens for each description
    stop_words = set(stopwords.words('english'))  # Load English stopwords set
    stemmer = SnowballStemmer('english')  # Initialize the Snowball stemmer for English

    # Process each description individually
    for description in descriptions:
        # Tokenize the description into words/punctuation using wordpunct_tokenize
        tokens = wordpunct_tokenize(description)
        
        # Remove stopwords and lowercase each word for uniformity
        tokens_without_stopwords = [word for word in tokens if word.lower() not in stop_words]
        
        # Remove punctuation by substituting any non-word characters with an empty string
        cleaned_tokens = [re.sub(r'[^\w\s]', '', token) for token in tokens_without_stopwords if re.sub(r'[^\w\s]', '', token)]
        
        # Stem each word to its root form
        stemmed_tokens = [stemmer.stem(word) for word in cleaned_tokens]
        
        # Append the processed tokens for this description to the list
        processed_descriptions.append(stemmed_tokens)

    return processed_descriptions

# Function to create 'vocabulary.csv' file
def create_vocabulary(processed_descriptions):
    """
    Checks if 'vocabulary.csv' exists. If it does, loads it as a DataFrame; if not, creates a vocabulary file in CSV format, mapping each unique word (term) in the processed texts to a unique integer ID.

    Args:
    processed_texts (list of list of str): A list of lists, where each sublist contains tokenized and processed words from a description.

    Returns:
    pd.DataFrame: A DataFrame containing the vocabulary, with each word mapped to a unique integer ID.
    """
    
    # Check if the vocabulary file already exists
    if os.path.exists('vocabulary.csv'):
        print("Loading existing vocabulary file.")
        vocabulary_df = pd.read_csv('vocabulary.csv')
    else:
        print("Creating new vocabulary file.")
        # Flatten the list of lists into a single list and convert it to a set to keep only unique words
        unique_terms = list(set([word for description in processed_descriptions for word in description]))
        
        # Create a DataFrame with term IDs and terms
        vocabulary_df = pd.DataFrame({
            'term_id': range(len(unique_terms)),  # Assign a unique integer ID to each term
            'term': unique_terms
        })
        
        # Save the vocabulary DataFrame to a CSV file named 'vocabulary.csv' without including the index
        vocabulary_df.to_csv('vocabulary.csv', index=False)
    
    return vocabulary_df

# Function to create the inverted index dictionary and save it as 'inverted_index.json'
def create_inverted_index(processed_descriptions, vocabulary_df, file_path="inverted_index.json"):
    """
    Creates or loads an inverted index for a collection of documents (processed descriptions).
    
    The inverted index maps term IDs to lists of document indices containing the term. 
    Document IDs are derived from the row index of the `processed_descriptions` list,
    meaning the document ID corresponds to the index of the description in the list.
    
    Args:
    processed_descriptions (list of list of str): A list of processed document descriptions, 
                                                  where each description is a list of terms (strings).
    vocabulary_df (pandas.DataFrame): A DataFrame containing 'term' and 'term_id' columns. 
                                      It maps each term to a unique term_id.
    file_path (str): Path to the file where the inverted index is stored. Default is "inverted_index.json".
    
    Returns:
    dict: An inverted index, where keys are term IDs and values are lists of document indices
          (rows) that contain each term. Document IDs correspond to the indices of the 
          descriptions in the `processed_descriptions` list.
    """
    
    # Check if the inverted index file exists
    if os.path.exists(file_path):
        # If the file exists, load the inverted index from the file
        with open(file_path, 'r') as f:
            print("Loading inverted index from file.")
            inverted_index = []
            inverted_index = json.load(f)
            inverted_index = {int(k): v for k, v in inverted_index.items()}
    else:
        # If the file does not exist, create the inverted index
        print("Creating inverted index...")
        
        # Create a mapping of terms to term_ids for fast lookup
        term_to_id = {term: term_id for term, term_id in zip(vocabulary_df['term'], vocabulary_df['term_id'])}
        
        # Initialize an empty dictionary for the inverted index
        inverted_index = {term_id: [] for term_id in vocabulary_df['term_id']}
        
        # Iterate over the documents
        for doc_idx, description in enumerate(processed_descriptions):
            # Use a set to avoid duplicate terms in a single document
            unique_terms = set(description)
            
            # For each unique term in the document, add the document index to the inverted index
            for term in unique_terms:
                # If the term exists in the vocabulary, add the document index to its term_id's list
                if term in term_to_id:
                    inverted_index[term_to_id[term]].append(doc_idx)
        
        # Save the inverted index to a JSON file
        with open(file_path, 'w') as f:
            json.dump(inverted_index, f, indent=4)  # Save with indentation for readability
            print(f"Inverted index saved to {file_path}.")
    
    return inverted_index

# Function to execute a search query by finding documents that contain all terms in the query.
def execute_conjunctive_query(query, inverted_index, vocabulary_df):
    """
    Executes a search query on an inverted index to find documents that contain all the terms in the query.
    
    Args:
    query (str): The search query, typically a string of words.
    inverted_index (dict): The inverted index where keys are term_ids and values are lists of document indices (IDs).
    vocabulary_df (pd.DataFrame): A DataFrame that maps terms to their unique term_ids.

    Returns:
    list: A list of document IDs that contain all the terms in the query.
    """
    
    # Preprocess the query to tokenize and clean the terms
    # Assumes query is a single string, and preprocesses it to get a list of terms
    query_list = preprocess_text([query])[0]  # preprocess_text returns a list of lists, we get the first (and only) list

    # Get the term_ids corresponding to the terms in the query
    # 'isin' checks if each term in the query is present in the vocabulary DataFrame
    # 'term_id' is the column in the vocabulary that maps each term to a unique integer ID
    terms_id = (vocabulary_df[vocabulary_df['term'].isin(query_list)]['term_id'].astype(int)).tolist()
    
    # Initialize a list to store the document sets for each term in the query
    documents_id = []
    
    # For each term_id from the query, retrieve the set of document IDs from the inverted index
    for term_id in terms_id:
        # Convert the term_id into a set of document IDs
        documents_id.append(set(inverted_index[term_id]))
   
    # Start with the set of document IDs for the first term
    intersection_result = documents_id[0]
    
    # Perform an intersection between all the document sets
    # The intersection operator '&=' finds common elements between sets
    for s in documents_id[1:]:
        intersection_result &= s  # Keep only the documents that contain all terms in the query
   
    # Return the list of document IDs that match all query terms
    return list(intersection_result)


# Function to compute the TF-IDF scores for terms in the given processed descriptions, and constructs an inverted index.
def compute_tfIdf_scores(processed_descriptions, terms, file_path="tfIdf_inverted_index.json"):
    """
    If the inverted index with TF-IDF scores already exists in a JSON file, it loads it. Otherwise, it computes 
    the TF-IDF scores from scratch and saves the inverted index to the specified file.

    Args:
        processed_descriptions (list of list of str): A list of processed restaurant descriptions, 
                                                      where each description is a list of terms (strings).
        terms (pandas.Series): A list or pandas Series containing the terms in the vocabulary, 
                               which will be used for calculating the TF-IDF scores.
        file_path (str): The file path where the inverted index with TF-IDF scores will be saved or loaded from. 
                         Default is "tfIdf_inverted_index.json".

    Returns:
        dict: An inverted index where each key is a term ID and the value is a list of tuples. Each tuple contains:
              - document ID (doc_idx): The index of the document in the `processed_descriptions` list.
              - TF-IDF score: The score representing the relevance of the term to the document.
    """

    # Check if the inverted index file exists
    if os.path.exists(file_path):
        # If the file exists, load the inverted index from the file
        with open(file_path, 'r') as f:
            print("Loading inverted index with TF-IDF scores from file.")
            tfIdf_inverted_index = {}
            tfIdf_inverted_index = json.load(f)

            # Convert the values in the inverted index from lists to tuples (doc_idx, score) for consistency
            tfIdf_inverted_index = {term: [(int(doc_idx), score) for doc_idx, score in docs] 
                                    for term, docs in tfIdf_inverted_index.items()}
    else:
        # Transform each description into a single string without modifying the original list of terms
        processed_descriptions_str = [' '.join(description) for description in processed_descriptions]

        # Initialize and apply the TfidfVectorizer using the specified vocabulary
        tfIdf_model = TfidfVectorizer(vocabulary=terms.tolist())
        tfIdf_scores = tfIdf_model.fit_transform(processed_descriptions_str).toarray()
        
        # Get the feature names (unique terms in the vocabulary)
        words_set = tfIdf_model.get_feature_names_out()

        # Build the inverted index with TF-IDF scores
        tfIdf_inverted_index = {}
        
        # Iterate through all terms in the vocabulary
        for term_idx, _ in enumerate(words_set):
            tfIdf_inverted_index[term_idx] = []
            
            # For each term, add document IDs and corresponding TF-IDF scores to the inverted index
            for doc_idx, score in enumerate(tfIdf_scores[:, term_idx]):
                if score > 0:  # Consider only non-zero scores
                    tfIdf_inverted_index[term_idx].append((doc_idx, score))
        
        # Save the inverted index to a JSON file for future use
        with open(file_path, 'w') as f:
            json.dump(tfIdf_inverted_index, f, indent=4)  # Save with indentation for readability
            print(f"Inverted index with TF-IDF scores saved to {file_path}.")

    # Return the computed (or loaded) inverted index with TF-IDF scores
    return tfIdf_inverted_index


def execute_ranked_query(query, inverted_index, vocabulary_df, k):
    return k

import numpy as np
from collections import defaultdict

def cosine_similarity(vector1, vector2):
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    return dot_product / (norm1 * norm2) if norm1 != 0 and norm2 != 0 else 0

def ranked_query(query_terms, inverted_index, vocabulary_df, top_k=1):
    """
    Execute a ranked search query.
    
    Args:
    - query_terms (list of str): Preprocessed list of terms in the query.
    - inverted_index (dict): Inverted index with TF-IDF scores.
    - vocabulary_df (pandas.DataFrame): DataFrame with 'term' and 'term_id' columns.
    - top_k (int): Number of top documents to return.
    
    Returns:
    - List of tuples (document_id, score) sorted by score in descending order.
    """
    # Map terms to term IDs using vocabulary
    query_list = preprocess_text([query_terms])[0]  # preprocess_text returns a list of lists, we get the first (and only) list

    # Get the term_ids corresponding to the terms in the query
    # 'isin' checks if each term in the query is present in the vocabulary DataFrame
    # 'term_id' is the column in the vocabulary that maps each term to a unique integer ID
    query_term_ids = (vocabulary_df[vocabulary_df['term'].isin(query_list)]['term_id'].astype(int)).tolist()
    
    # Construct query vector (TF-IDF values for query terms)
    query_vector = np.zeros(len(query_term_ids))
    for i, term_id in enumerate(query_term_ids):
        # Use a simple IDF value of 1 if you don't have specific query TF-IDF scores
        query_vector[i] = inverted_index[term_id]
        print(query_vector[i])
    
    # Initialize document vectors based on query terms found in documents
    document_vectors = defaultdict(lambda: np.zeros(len(query_term_ids)))
    
    # Populate document vectors with the TF-IDF scores from the inverted index
    for i, term_id in enumerate(query_term_ids):
        if term_id in inverted_index:
            for doc_id, tfidf_score in inverted_index[term_id]:
                document_vectors[doc_id][i] = tfidf_score
    
    # Calculate cosine similarity for each document
    scores = {}
    for doc_id, doc_vector in document_vectors.items():
        scores[doc_id] = cosine_similarity(query_vector, doc_vector)
    
    # Sort documents by similarity score in descending order and get top-k
    ranked_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    return ranked_results