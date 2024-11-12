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
import numpy as np
import math
from collections import defaultdict

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


def get_tfIdf(term, document, corpus):
    """
    Calculate the TF-IDF (Term Frequency-Inverse Document Frequency) score for a given term in a document.
    
    TF-IDF is a statistic used to evaluate the importance of a term within a document relative to a corpus of documents.
    The formula is:
        TF-IDF = TF * IDF

    Where:
        - TF (Term Frequency) measures how frequently a term appears in a document.
        - IDF (Inverse Document Frequency) measures the rarity of the term across the entire corpus.

    Parameters:
    - term (str): The term for which the TF-IDF score is being calculated.
    - document (list of str): The list of words (terms) in the document being analyzed.
    - corpus (list of list of str): The entire collection of documents, each represented as a list of words.

    Returns:
    - float: The calculated TF-IDF score for the term in the given document.

    Detailed explanation of the computation:
    
    1. **Term Frequency (TF):**
       TF is calculated as the ratio of the count of the `term` in the `document` to the total number of words in that document.
       This gives a measure of how important the term is within the context of the document.
       
       Formula:
       TF = count of the term in the document / total number of words in the document

    2. **Inverse Document Frequency (IDF):**
       IDF is calculated to measure the importance of the `term` across the entire `corpus`. A term that appears in many documents is considered less informative, 
       while a term that appears in fewer documents is considered more informative.
       IDF is calculated by taking the logarithm of the total number of documents divided by the number of documents containing the term. 
       The `+1` in both the numerator and denominator ensures that terms that appear in every document do not result in a division by zero.

       Formula:
       IDF = log10(total number of documents / number of documents containing the term)

    3. **TF-IDF Calculation:**
       The TF and IDF values are multiplied together to give the TF-IDF score.

    """
    
    # Compute Term Frequency (TF)
    tf = document.count(term) / len(document)  # How often the term appears in the document, normalized by document length
    
    # Compute the total number of documents in the corpus (adding 1 for the "+1" term in the IDF formula)
    count_of_documents = len(corpus) + 1
    
    # Compute how many documents contain the term (adding 1 for the "+1" term in the IDF formula)
    count_of_documents_with_term = sum([1 for doc in corpus if term in doc]) + 1
    
    # Compute Inverse Document Frequency (IDF)
    idf = np.log10(count_of_documents / count_of_documents_with_term)  # Logarithmic scaling of document frequency
    
    # Return the TF-IDF score
    return tf * idf  # The TF-IDF score is the product of TF and IDF


def create_tfIdf_inverted_index(inverted_index, vocabulary, processed_description, file_path="tfIdf_inverted_index.json"):
    """
    Create or load a TF-IDF inverted index for a given corpus of documents, based on the term frequency (TF)
    and inverse document frequency (IDF) scores. The inverted index will map terms to the documents in which
    they appear along with their corresponding TF-IDF scores.

    If the inverted index already exists (stored in a JSON file), it will be loaded. If not, it will be generated
    from the vocabulary, the processed descriptions of the documents, and the pre-existing inverted index.
    
    Parameters:
    - inverted_index (dict): A dictionary where the keys are term IDs and the values are lists of document IDs
                              in which the term appears.
    - vocabulary (DataFrame): A DataFrame containing the terms in the corpus, where each term has a corresponding
                              unique term ID.
    - processed_description (list of list of str): A list of processed documents, each represented as a list of terms.
    - file_path (str): The file path to save or load the inverted index with TF-IDF scores. Defaults to "tfIdf_inverted_index.json".
    
    Returns:
    - tfIdf_inverted_index (dict): A dictionary where the keys are term IDs, and the values are lists of tuples
                                    (document ID, TF-IDF score) representing the importance of each term in each document.
    """

    # Check if the inverted index file exists
    if os.path.exists(file_path):
        # If the file exists, load the inverted index with TF-IDF scores from the file
        with open(file_path, 'r') as f:
            print("Loading inverted index with TF-IDF scores from file.")
            tfIdf_inverted_index = json.load(f)
            
            # Convert the values in the inverted index from lists to tuples (doc_idx, score) for consistency
            # Ensure that all keys are converted to integers (term IDs) and the document IDs and scores are also integers
            tfIdf_inverted_index = {int(term): [(int(doc_idx), score) for doc_idx, score in docs] 
                                    for term, docs in tfIdf_inverted_index.items()}
    else:
        print("Creating inverted index with TF-IDF scores...")
        
        # Initialize an empty dictionary to store the inverted index with TF-IDF scores
        tfIdf_inverted_index = {}
        
        # Iterate through all terms in the vocabulary
        for term in vocabulary['term']:
            # Get the term ID from the vocabulary DataFrame
            term_id = int(vocabulary[vocabulary['term'] == term]['term_id'].iloc[0])
            
            # Initialize an empty list to store document IDs and TF-IDF scores for the current term
            tfIdf_inverted_index[term_id] = []

            # For each document that contains the current term, calculate the TF-IDF score
            for doc_id in inverted_index[term_id]:
                # Compute the TF-IDF score for the current term in the current document
                tf_idf_score = get_tfIdf(term, processed_description[doc_id], processed_description)
                
                # Append the document ID and its corresponding TF-IDF score to the list for the current term
                tfIdf_inverted_index[term_id].append((doc_id, tf_idf_score))

        # Save the created inverted index to a JSON file for future use
        with open(file_path, 'w') as f:
            json.dump(tfIdf_inverted_index, f, indent=4)
            print(f"Inverted index with TF-IDF scores saved to {file_path}.")
    
    # Return the generated or loaded inverted index with TF-IDF scores
    return tfIdf_inverted_index


def cosine_similarity(doc_vector, query_vector):
    """
    Calculate the cosine similarity between two vectors.

    Cosine similarity is a metric used to measure how similar two vectors are, 
    regardless of their magnitude, by calculating the cosine of the angle between them.
    The cosine similarity value ranges from -1 (completely opposite) to 1 (completely similar).
    A value of 0 indicates orthogonality or no similarity.

    Args:
        doc_vector (numpy array): A vector representing the document.
        query_vector (numpy array): A vector representing the query.

    Returns:
        float: The cosine similarity score between the document vector and the query vector.
               Returns 0 if the denominator is 0 (i.e., if either of the vectors is a zero vector).
    """
    
    # Calculate the dot product between the document and query vectors
    dot_product = np.dot(doc_vector, query_vector)
    
    # Calculate the L2 norm of the document vector
    doc_vector_norm = np.sqrt(np.dot(doc_vector, doc_vector))
    
    # Calculate the L2 norm of the query vector
    query_vector_norm = np.sqrt(np.dot(query_vector, query_vector))
    
    # Calculate the denominator (the product of the norms of the two vectors)
    denominator = doc_vector_norm * query_vector_norm
    
    # Return the cosine similarity score if the denominator is not zero, otherwise return 0
    return dot_product / denominator if denominator != 0 else 0


def execute_ranked_query(query_terms, inverted_index, vocabulary_df, processed_description, top_k):
    """
    Execute a ranked query by calculating the cosine similarity between the query vector (TF-IDF)
    and document vectors. The function retrieves the top-k most similar documents to the query.

    Parameters:
    - query_terms (str): The query input as a space-separated string of terms.
    - inverted_index (dict): A dictionary where each key is a term ID and each value is a list of tuples (document ID, TF-IDF score),
                              representing the inverted index for the documents.
    - vocabulary_df (DataFrame): A DataFrame containing the vocabulary of terms, where each term has a corresponding term ID.
    - processed_description (list of list of str): A list of processed documents, each represented as a list of terms.
    - top_k (int): The number of top-ranked documents to return based on similarity.

    Returns:
    - ranked_results (list): A list of tuples, where each tuple contains a document ID and its corresponding similarity score.
    """
    
    # Preprocess the query terms (tokenize, remove stopwords, etc.)
    query_list = preprocess_text([query_terms])[0]  # preprocess_text returns a list of lists, so we get the first list
    
    # Get the term_ids corresponding to the terms in the query
    # 'isin' checks if each term in the query is present in the vocabulary DataFrame
    # 'term_id' is the column in the vocabulary that maps each term to a unique integer ID
    query_term_ids = (vocabulary_df[vocabulary_df['term'].isin(query_list)]['term_id'].astype(int)).tolist()
    
    # Construct the query vector by calculating TF-IDF values for the query terms
    # The query vector is initialized as a zero vector of the same length as the vocabulary
    query_vector = np.zeros(vocabulary_df.shape[0])
    for i in range(0, len(query_list)):
        # For each term in the query, we calculate the TF-IDF value for that term and update the query vector
        query_vector[query_term_ids[i]] = get_tfIdf(query_list[i], query_list, processed_description)
        
    # Initialize a dictionary to store document vectors, using a default value of zero vector
    document_vectors = defaultdict(lambda: np.zeros(vocabulary_df.shape[0]))
    
    # Populate document vectors with the TF-IDF scores for the query terms
    for term_id in query_term_ids:
        if term_id in inverted_index.keys():  # Check if the term exists in the inverted index
            for doc_id, tfidf_score in inverted_index[term_id]:
                # Update the document vector with the TF-IDF score for the current document and term
                document_vectors[doc_id][term_id] = tfidf_score

    # Calculate cosine similarity for each document vector against the query vector
    scores = {}
    for doc_id, doc_vector in document_vectors.items():
        # Compute the cosine similarity between the document vector and the query vector
        scores[doc_id] = cosine_similarity(doc_vector, query_vector)
    
    # Sort the documents based on the cosine similarity scores in descending order
    ranked_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # If the number of results is greater than the desired top_k, slice the list to get only the top_k results
    if len(ranked_results) > top_k:
        ranked_results = ranked_results[:top_k]
    
    # Return the ranked results (document ID and similarity score)
    return ranked_results
