
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import string
import re
import glob
from tqdm import tqdm
import os
import json
from tabulate import tabulate
import math

# Text preprocessing setup
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))


# Function to create the dataframe for the dataset
def create_combined_dataframe_ID(folder_path):
    """
    Creates a combined DataFrame from all .tsv files in a specified folder.

    Parameters:
    - folder_path: Path to the folder containing .tsv files.
    - separator: Delimiter used in the .tsv files (default: ',').

    Returns:
    - DataFrame containing all combined data.
    """
    # Find all .tsv files in the specified folder
    all_files = glob.glob(os.path.join(folder_path, "TSVs/*.tsv"))

    df_list = []

    # Load each .tsv file as a DataFrame, saving the restaurant ID, and store in a list
    for file in all_files:
        restaurant_id = int(file.split('_')[-1].split('.')[0])  # Extract unique ID
        data = pd.read_csv(file, sep='\t')
        data['restaurant_id'] = int(restaurant_id)  # Add ID as a new column
        df_list.append(data)

    # Concatenate all DataFrames in the list into a single DataFrame
    combined_df = pd.concat(df_list, ignore_index=True)
    df = combined_df.set_index('restaurant_id')

    return df

# Function to preprocess text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Replace punctuation with a space
    text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
    # Tokenize text
    words = word_tokenize(text)
    # Remove stop words, stem the tokens, and filter out single punctuation or non-alphanumeric tokens
    words = [stemmer.stem(word) for word in words if word.isalnum() and word not in stop_words]
    return words

# Build and save the vocabolary from specified column of dataframe, also return it as a dictionary
def vocab_from_df(df, column='processed_description', data_folder='DATA'):
    # Create Vocabulary
    unique_words = pd.Series([word for words in df['processed_description'] for word in words]).unique()
    vocab = {word: idx for idx, word in enumerate(unique_words)}

    # Save Vocabulary
    pd.DataFrame(list(vocab.items()), columns=['word', 'term_id']).to_csv(f'{data_folder}/vocabulary.csv', index=False)

    return vocab

# Build and save the inverted index from vocab and df, saves it as .json
def get_inverted_index(vocab, df, column='processed_description', data_folder='DATA'):
    # Build Inverted Index
    inverted_index = {}
    
    # Iterate over the DataFrame rows, using the index (restaurant_id)
    for restaurant_id, row_data in df.iterrows():
        words = row_data[column]  # Access the words in the 'processed_description' column for the current row
        
        for word in words:
            term_id = vocab.get(word)  # Use .get() to avoid errors for non-existing terms
            if term_id is None:
                continue  # Skip if the term is not in the vocab
            
            if term_id not in inverted_index:
                inverted_index[term_id] = []
            
            # Append restaurant_id to the term_id list, only if not already present
            if restaurant_id not in inverted_index[term_id]:
                inverted_index[term_id].append(restaurant_id)

    # Sort each list of document IDs in the inverted index
    for term_id in inverted_index:
        inverted_index[term_id].sort()

    # Save Inverted Index
    with open(f'{data_folder}/inverted_index.json', 'w') as f:
        json.dump(inverted_index, f)
    
    # Return the inverted index dictionary
    return inverted_index

# Function to perform conjunctive search
def conjunctive_search(query, vocab, inverted_index):
    """
    Perform a conjunctive (AND) search over the inverted index.

    Parameters:
    - query (str): The query string to search for.
    - vocab (dict): Mapping of terms to their term IDs.
    - inverted_index (dict): Mapping of term IDs to lists of document IDs (sorted).
    - preprocess_text (function): The text preprocessing function to apply to the query.

    Returns:
    - set: A set of document IDs that contain all query terms.
    """
    # Preprocess the query
    query_terms = preprocess_text(query)

    # Convert query terms to their corresponding term IDs, handle missing terms gracefully
    term_ids = [vocab[term] for term in query_terms if term in vocab]

    if not term_ids:
        return set()  # No terms found in vocab, return empty set

    # Retrieve the document lists for each term ID
    doc_lists = [inverted_index[term_id] for term_id in term_ids]

    if not doc_lists:
        return set()  # No documents contain the terms

    # Perform conjunctive (AND) search by finding the intersection of all doc lists
    result_set = set(doc_lists[0])
    for doc_list in doc_lists[1:]:
        result_set.intersection_update(doc_list)

    return result_set

def display_restaurants(matched_restaurant_ids, df):
    """
    Given a list of matched restaurant IDs and the dataframe `df`, 
    display a formatted table with restaurant information.

    Args:
        matched_restaurant_ids (list): List of restaurant IDs that match the query.
        df (pd.DataFrame): DataFrame containing restaurant information (name, address, description, website).

    Returns:
        none
    """
    # Filter the dataframe to get only the rows where the restaurant_id is in the matched_restaurant_ids
    matched_df = df[df.index.isin(matched_restaurant_ids)]

    # Select relevant columns to display
    display_columns = ['restaurantName', 'address', 'city', 'phoneNumber', 'website', 'description']
    result_df = matched_df[display_columns]

    headers = ['Restaurant Name', 'Address', 'City', 'Phone Number', 'Website', 'Description']
    coalign = tuple(["left"]*6)
    print(tabulate(result_df, headers=headers, tablefmt='pretty', showindex=False, colalign=coalign))

    

def get_idf(inverted_index, df, column): 
    IDF = {}
    TOT = len(df[column])
    for term_id, rest_ids in inverted_index.items():
        DF = len(rest_ids)  # Document Frequency: number of documents (i.e. restaurant descriptions) containing the term
        IDF[term_id] = math.log(TOT / DF)
    return IDF


def get_tf_idf(inverted_index, vocab, df, column='processed_description', data_folder='DATA'):

    IDF = get_idf(inverted_index, df, column)

    TF_IDF_inverted_index = {}
    for term, term_id in vocab.items():
        # For each term, calculate the TF-IDF score for each document (restaurant description)
        for doc_id in inverted_index[term_id]:
            # Get the term frequency for the current document
            term_count = df.loc[doc_id, 'processed_description'].count(term)
            total_terms = len(df.loc[doc_id, 'processed_description'])
            TF = term_count / total_terms if total_terms > 0 else 0

            # Get the TF-IDF score
            TF_IDF = TF * IDF[term_id]

            # Update the inverted index with the term_id and tf-idf score
            if term_id not in TF_IDF_inverted_index:
                TF_IDF_inverted_index[term_id] = []

            TF_IDF_inverted_index[term_id].append((doc_id, TF_IDF))

    # Save Inverted Index
    with open(f'{data_folder}/TF_IDF_inverted_index.json', 'w') as f:
        json.dump(TF_IDF_inverted_index, f)

    return TF_IDF_inverted_index



def get_query_tf_idf(query, inverted_index, vocab, df):

    # Step 1: Preprocess the query
    processed_query = preprocess_text(query)
    
    # Step 2: Get IDF values using the existing get_idf function
    idf = get_idf(inverted_index, df, 'processed_description')  # Assuming 'processed_description' is the column name
    
    # Step 3: Calculate TF for each term in the query
    term_freq = {}
    total_terms = len(processed_query)  # Total terms in the query
    
    for term in processed_query:
        term_freq[term] = term_freq.get(term, 0) + 1  # Count occurrences of each term in the query
    
    # Step 4: Calculate TF-IDF for each term in the query
    query_tf_idf = {}
    
    for term, count in term_freq.items():
        if term in vocab:  # Only compute TF-IDF for terms that exist in the vocab
            term_id = vocab[term]
            tf = count / total_terms  # Calculate the Term Frequency
            query_tf_idf[term_id] = tf * idf.get(term_id, 0)  # Multiply TF by IDF
    
    return query_tf_idf



def cosine_search(query, TF_IDF_inverted_index, IDF, k=5):
    # Step 1: Preprocess the query
    query_terms = preprocess_text(query)
    
    # Step 2: Calculate the TF-IDF for the query
    IDF = get_idf(inverted_index, df, column)
    query_tf_idf = 
    
    # Step 3: Calculate cosine similarity for each restaurant
    similarity_scores = []
    for restaurant_id in df.index:
        similarity = compute_cosine_similarity(query_tf_idf, restaurant_id, inverted_index, df)
        similarity_scores.append((restaurant_id, similarity))
    
    # Step 4: Sort and return the top `k` restaurants
    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    top_k_restaurants = similarity_scores[:k]
    
    # Return the top k restaurant information
    results = []
    for restaurant_id, score in top_k_restaurants:
        restaurant_info = df.loc[restaurant_id]
        results.append({
            'restaurant_name': restaurant_info['restaurant_name'],
            'address': restaurant_info['address'],
            'description': restaurant_info['description'],
            'website': restaurant_info['website'],
            'similarity_score': score
        })
    
    return results