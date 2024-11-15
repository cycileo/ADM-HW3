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
import ast
import ipywidgets as widgets
from IPython.display import display, clear_output, Image
from ipywidgets import Layout, Checkbox, GridBox, Label, IntRangeSlider
from tabulate import tabulate


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
def create_vocabulary(processed_descriptions, file_name=''):
    """
    Checks if 'vocabulary.csv' exists. If it does, loads it as a DataFrame; if not, creates a vocabulary file in CSV format, mapping each unique word (term) in the processed texts to a unique integer ID.

    Args:
    processed_texts (list of list of str): A list of lists, where each sublist contains tokenized and processed words from a description.

    Returns:
    pd.DataFrame: A DataFrame containing the vocabulary, with each word mapped to a unique integer ID.
    """

    if file_name:
        file_path = 'DATA/vocabulary_' + file_name + '.csv'
    else: 
        file_path = 'DATA/vocabulary.csv' 

    # Check if the vocabulary file already exists
    if os.path.exists(file_path):
        print("Loading existing vocabulary file.")
        vocabulary_df = pd.read_csv(file_path)
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
        vocabulary_df.to_csv(file_path, index=False)
    
    return vocabulary_df

# Function to create the inverted index dictionary and save it as 'inverted_index.json'
def create_inverted_index(processed_descriptions, vocabulary_df, file_name=""):
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
    
    # 
    if file_name:
        file_path = 'DATA/inverted_index_' + file_name + '.json'
    else: 
        file_path = 'DATA/inverted_index.json'

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


def create_tfIdf_inverted_index(inverted_index, vocabulary, processed_description, file_name=""):
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

    if file_name:
        file_path = 'DATA/tfIdf_inverted_index_' + file_name + '.json'
    else: 
        file_path = 'DATA/tfIdf_inverted_index.json'

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



def get_bonus_lists(df): 
    df['processedName'] = preprocess_text(df['restaurantName'].astype(str))
    df['processedCity'] = preprocess_text(df['city'])
    df['processedCuisine'] = preprocess_text(df['cuisineType'])
    name_voc = create_vocabulary(df['processedName'], 'name')
    city_voc = create_vocabulary(df['processedCity'], 'city')
    cuis_voc = create_vocabulary(df['processedCuisine'], 'cuis')
    name_inv_ind = create_inverted_index(df['processedName'], name_voc, 'name')
    city_inv_ind = create_inverted_index(df['processedCity'], city_voc, 'city')
    cuis_inv_ind = create_inverted_index(df['processedCuisine'], cuis_voc, 'cuis')
    name_tfidf = create_tfIdf_inverted_index(name_inv_ind, name_voc, df['processedName'], 'name')
    city_tfidf = create_tfIdf_inverted_index(city_inv_ind, city_voc, df['processedCity'], 'city')
    cuis_tfidf = create_tfIdf_inverted_index(cuis_inv_ind, cuis_voc, df['processedCuisine'], 'cuis')
    tfidf_list = [name_tfidf, city_tfidf, cuis_tfidf]
    voc_list = [name_voc, city_voc, cuis_voc]
    processed_list = [df['processedName'], df['processedCity'], df['processedCuisine']]
    return (tfidf_list, voc_list, processed_list)



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



def compute_query_similarity(query_terms, inverted_index, vocabulary_df, processed_texts):
    """
    Executes a ranked query by calculating the cosine similarity between a query vector (TF-IDF)
    and document vectors, using only the terms from the query that exist in the vocabulary.

    The function performs the following steps:
    - Tokenizes and processes the input query.
    - Filters out query terms that are not found in the vocabulary.
    - Creates a query vector using the TF-IDF values for the terms present in the vocabulary.
    - Calculates cosine similarity between the query vector and the document vectors.

    Args:
        query_terms (str): Query input as a space-separated string of terms.
        inverted_index (dict): Dictionary where each key is a term ID, and each value is a list of tuples 
                               (document ID, TF-IDF score) representing the inverted index for documents.
        vocabulary_df (DataFrame): DataFrame containing the vocabulary terms and their corresponding term IDs.
        processed_texts (list of list of str): List of preprocessed texts, where each text is a list of terms.
        
    Returns:
        np.ndarray: Array of cosine similarity scores for each document.
    """
    
    # Preprocess the input query and tokenize it
    query_list = preprocess_text([query_terms])[0]
    
    # Filter out query terms that are not present in the vocabulary
    query_list = [term for term in query_list if term in vocabulary_df['term'].values]

    # Map query terms to their corresponding term IDs from the vocabulary
    query_term_ids = (vocabulary_df[vocabulary_df['term'].isin(query_list)])
    query_term_ids = query_term_ids.set_index('term').loc[query_list].reset_index()['term_id'].astype(int).tolist()

    # Initialize the query vector with zeros, one for each term in the vocabulary
    query_vector = np.zeros(vocabulary_df.shape[0])
    for i in range(len(query_term_ids)):
        # Set the TF-IDF value for each query term
        query_vector[query_term_ids[i]] = get_tfIdf(query_list[i], query_list, processed_texts)
        
    # Initialize document vectors as a defaultdict with zero vectors for each term
    document_vectors = defaultdict(lambda: np.zeros(vocabulary_df.shape[0]))
    
    # Populate document vectors with the TF-IDF values from the inverted index
    for term_id in vocabulary_df['term_id']:
        if term_id in inverted_index:
            for doc_id, tfidf_score in inverted_index[term_id]:
                document_vectors[doc_id][term_id] = tfidf_score
    
    # Calculate cosine similarity between the query vector and each document vector
    scores = {doc_id: cosine_similarity(doc_vector, query_vector) for doc_id, doc_vector in document_vectors.items()}

    # Find the maximum document ID for creating the final score vector
    max_doc_id = max(scores.keys())
    
    # Create an array to store the similarity scores for each document
    scores_vec = np.zeros(max_doc_id + 1)
    for doc_id, score in scores.items():
        scores_vec[doc_id] = score

    return scores_vec


def multiple_ranked_query(query_list, tfidf_list, voc_list, processed_list, df):
    """
    Executes ranked queries for multiple fields with weighted importance, calculates cosine similarity 
    for each field-specific query, and filters the documents.

    Args:
        query_list (list of str): A list of field-specific queries. Each query corresponds to a specific field.
        tfidf_list (list of dict): A list of TF-IDF inverted indices, each corresponding to one field.
        voc_list (list of DataFrame): A list of vocabulary DataFrames for the terms of each field.
        processed_list (list of list of str): A list of preprocessed terms for each field's query.
        df (DataFrame): The dataset of restaurants, containing details such as name, city, and cuisine type.
        weights (list of float, optional): A list of weights for each field. Default is [0.5, 0.3, 0.2].

    Returns:
        DataFrame: A filtered DataFrame with documents that match the queries. Includes a new column, `score`, 
                   with the weighted similarity score across all fields.
    """
    # Weights for the search field matches
    weights = [0.55, 0.30, 0.15]

    scores = []

    # Iterate over the queries corresponding to different fields
    for i in range(len(query_list)):
        if query_list[i]:  # Skip empty queries
            field_score = compute_query_similarity(query_list[i], tfidf_list[i], voc_list[i], processed_list[i])
            scores.append(weights[i] * field_score)  # Apply weight to the score
    
    # Calculate the weighted sum of scores, skipping missing queries
    score = np.sum(scores, axis=0)

    # Find indices of documents with non-zero similarity scores
    non_zero_indices = np.nonzero(score)[0]

    # Filter the dataset to include only relevant documents
    filtered_df = df.loc[non_zero_indices].copy()

    # Add a 'score' column with the computed similarity scores
    filtered_df['score'] = score[non_zero_indices]

    return filtered_df




# Funzione per convertire una stringa in lista
def convert_to_list(value):
    try:
        return ast.literal_eval(value)  # Converte la stringa in una lista
    except (ValueError, SyntaxError):  # Gestisce eventuali errori di conversione
        return value  # Se la conversione fallisce, restituisce il valore originale (potrebbe non essere una lista)

def display_results_and_filters(df_res, top_k=10):
    # Contenitore separato per i risultati
    results_output = widgets.Output()

    # Definire i filtri manualmente, se necessario
    regions_filter = []  # Impostazione di un filtro vuoto per la regione (o puoi definirlo con un altro valore)
    list_reg = sorted(df_res["region"].unique())

    # Creazione delle checkbox per le regioni
    region_checkboxes = [
        Checkbox(value=(region in regions_filter), description=region)
        for region in list_reg
    ]

    # Etichetta per il filtro delle regioni
    filter_label = Label(value="Filter by Region")

    # Doppio slider per il filtro del prezzo
    price_range_slider = IntRangeSlider(
        value=[1, 4],
        min=1,
        max=4,
        step=1,
        description='Price Range:',
        style={'description_width': 'initial'},
        continuous_update=False
    )

    # Filtro per i servizi e carte di credito
    facilities_filter = []
    credit_cards_filter = []

    # Estrai le stringhe da convertire in liste (se sono stringhe di liste)
    df_res['facilitiesServices'] = df_res['facilitiesServices'].apply(convert_to_list)
    df_res['creditCards'] = df_res['creditCards'].apply(convert_to_list)

    # Crea le checkbox per i servizi
    list_facilities = sorted(set([item for sublist in df_res['facilitiesServices'] for item in sublist]))
    facilities_checkboxes = [
        Checkbox(value=(facility in facilities_filter), description=facility)
        for facility in list_facilities
    ]

    # Crea le checkbox per le carte di credito
    list_credit_cards = sorted(set([item for sublist in df_res['creditCards'] for item in sublist]))
    credit_cards_checkboxes = [
        Checkbox(value=(card in credit_cards_filter), description=card)
        for card in list_credit_cards
    ]

    # Etichetta per il filtro dei servizi
    facilities_label = Label(value="Filter by Services")

    # Etichetta per il filtro delle carte di credito
    credit_cards_label = Label(value="Filter by Credit Cards")

    # Funzione per aggiornare i risultati
    def update_results(_=None):
        # Filtro per le regioni
        selected_regions = [
            checkbox.description for checkbox in region_checkboxes if checkbox.value
        ]
        filtered_df = (
            df_res[df_res["region"].isin(selected_regions)]
            if selected_regions
            else df_res
        )

        # Filtro per la lunghezza del prezzo (basato sul range dello slider)
        min_length, max_length = price_range_slider.value
        filtered_df = filtered_df[filtered_df['priceRange'].apply(lambda x: min_length <= len(x) <= max_length)]

        # Filtro per i servizi (modificato per essere esclusivo)
        selected_facilities = [
            checkbox.description for checkbox in facilities_checkboxes if checkbox.value
        ]
        if selected_facilities:
            # Ogni ristorante deve avere **tutti** i servizi selezionati
            filtered_df = filtered_df[filtered_df['facilitiesServices'].apply(lambda x: all(facility in x for facility in selected_facilities))]

        # Filtro per le carte di credito (modificato per essere esclusivo)
        selected_cards = [
            checkbox.description for checkbox in credit_cards_checkboxes if checkbox.value
        ]
        if selected_cards:
            # Ogni ristorante deve avere **tutti** i tipi di carte selezionate
            filtered_df = filtered_df[filtered_df['creditCards'].apply(lambda x: all(card in x for card in selected_cards))]

        # Pulisce solo la parte dei risultati
        with results_output:
            clear_output(wait=True)
            
            # Calcola il numero di ristoranti che soddisfano i criteri
            num_restaurants = len(filtered_df)
            
            # Mostra il messaggio con il numero di ristoranti
            print(f"\n Found {num_restaurants} restaurant(s) matching the criteria.\n")
            
            # Se ci sono ristoranti da mostrare, ordina e mostra i top_k
            if num_restaurants > 0:
                top_restaurants = filtered_df.nlargest(top_k, 'score')
                # Seleziona le colonne da visualizzare
                display_columns = ['restaurantName', 'address', 'city', 'region', 'priceRange', 'cuisineType', 'website']
                disp_df = top_restaurants[display_columns]

                headers = ['Restaurant Name', 'Address', 'City', 'Region', 'Price', 'Cuisine Type', 'Website']
                coalign = tuple(["left"]*6)
                print(tabulate(disp_df, headers=headers, tablefmt='pretty', showindex=False, colalign=coalign))


    # Osserva i cambiamenti delle checkbox e dello slider
    for checkbox in region_checkboxes:
        checkbox.observe(update_results, names="value")
    
    price_range_slider.observe(update_results, names="value")
    for checkbox in facilities_checkboxes:
        checkbox.observe(update_results, names="value")
    for checkbox in credit_cards_checkboxes:
        checkbox.observe(update_results, names="value")

    # Griglia per i filtri delle regioni
    region_grid = GridBox(
        children=region_checkboxes,
        layout=Layout(
            grid_template_columns="repeat(4, 1fr)",  # 3 colonne
            grid_gap="10px",
            border="1px solid #ccc",
            padding="10px",
        ),
    )

    # Griglia per i filtri dei servizi
    facilities_grid = GridBox(
        children=facilities_checkboxes,
        layout=Layout(
            grid_template_columns="repeat(4, 1fr)",  # 3 colonne
            grid_gap="10px",
            border="1px solid #ccc",
            padding="10px",
        ),
    )

    # Griglia per i filtri delle carte di credito
    credit_cards_grid = GridBox(
        children=credit_cards_checkboxes,
        layout=Layout(
            grid_template_columns="repeat(4, 1fr)",  # 3 colonne
            grid_gap="10px",
            border="1px solid #ccc",
            padding="10px",
        ),
    )

    # Mostra i widget dei filtri
    display(price_range_slider, filter_label, region_grid, facilities_label, facilities_grid, credit_cards_label, credit_cards_grid)

    # Mostra i risultati iniziali all'interno del container separato
    display(results_output)
    update_results()  # Mostra subito i risultati iniziali



def bonus_serach(tfidf_list, voc_list, processed_list, df): 
    
    # Definizione dei widget
    query1 = widgets.Text(description="Restaurant:")
    query2 = widgets.Text(description="City:")
    query3 = widgets.Text(description="Cuisine Type:")
    search_button = widgets.Button(description="Search")

    # Menu a discesa per il numero di risultati (con una larghezza più piccola)
    results_dropdown = widgets.Dropdown(
        options=[10, 20, 50, 100],
        value=10,  # Valore predefinito
        description='Results:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='150px')  # Imposta una larghezza fissa
    )

    # Contenitore per organizzare i widget in una riga (search_box con il dropdown a destra)
    search_box = widgets.HBox([widgets.VBox([query1, query2, query3]), results_dropdown, search_button])
    display(search_box)

    # Callback per il pulsante
    def on_search_clicked(_):
        clear_output(wait=True)
        display(search_box)
        query = [query1.value, query2.value, query3.value]
        top_k = results_dropdown.value  # Ottieni il numero di risultati selezionato
        if query1.value=='' and query2.value=='' and query3.value=='':
            display(Image(url="https://staticfanpage.akamaized.net/wp-content/uploads/sites/6/2019/09/math-lady-1200x675.jpg", width=600, height=340))
            print("Can't read your mind \nNo result matching an empty query :(")
        else:
            # Chiamata alla funzione di ricerca
            df_res = multiple_ranked_query(query, tfidf_list, voc_list, processed_list, df)
            display_results_and_filters(df_res, top_k=top_k)  # Passa top_k per determinare il numero di risultati

    search_button.on_click(on_search_clicked)