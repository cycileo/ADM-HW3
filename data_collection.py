import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import glob



# ========================================================
#                 FIRST PART: COLLECTING THE LINKS
# - This section handles scraping restaurant links 
#   from the Michelin Guide website.
# - It gets the total number of pages from the 
#   starting page, then iterates over all the pages
#   collectin restaurant links
# ========================================================


# Function to get the restaurant links from all the pages
def save_links(start_url, data_folder = 'DATA', file_name = 'restaurant_links.txt'): 

    # Skip the link collection if the file already exists
    file_path = os.path.join(data_folder, file_name)
    if os.path.exists(file_path):
        print("Links already collected.")
        with open(file_path, 'r') as f:
            links_num = sum(1 for line in f)
        print(f"There are {links_num} link already collected")
        return None
    
    # List to save all restaurant links
    restaurant_links = []

    # Request the first page to determine the total number of pages
    response = requests.get(start_url)
    if response.status_code != 200:
        print(f"Error loading the first page: {response.status_code}")
        print("Please check your internet connection or try again in a few minutes.")
        return None
    
    # Parse the page
    soup = BeautifulSoup(response.text, "lxml")
    
    # Find the last page from the pagination section
    pagination = soup.find("ul", class_="pagination")
    last_page_link = pagination.find_all("a")[-2]  # Take the second-to-last link, which is the last page number
    max_pages = int(last_page_link.text)

    print(f"Detected number of pages: {max_pages}")

    # Iterate over all detected pages
    for page_num in tqdm(range(1, max_pages + 1), desc="Links Scraping"):
        url = f"{start_url}/page/{page_num}"
        response = requests.get(url)
        
        if response.status_code != 200:
            print(f"Error on page {page_num}: {response.status_code}")
            print("Please check your internet connection or try again in a few minutes.")
            return None
        
        # Parse the page
        soup = BeautifulSoup(response.text, "lxml")
        
        # Search only in the search results section
        section = soup.find("section", class_="section-main search-results search-listing-result")
        for a_tag in section.find_all("a", class_="link", href=True):
            restaurant_link = "https://guide.michelin.com" + a_tag['href']
            restaurant_links.append(restaurant_link)

    # Print how many links are found
    print(f"Found {len(restaurant_links)} restaurant links")

    # Create the folder to store the file if it doesn't alreday exists
    os.makedirs(data_folder, exist_ok=True)
    
    # Save the links to a .txt file
    with open(file_path, "w") as f:
        for link in restaurant_links:
            f.write(link + "\n")



# ========================================================
#                 SECOND PART: DOWNLOADING THE PAGES
# - This section downloads the restaurant pages using 
#   ThreadPoolExecutor to make it faster
# ========================================================


# Function to read restaurant links from a file
def read_links_from_file(file_path):
    with open(file_path, "r") as f:
        return [line.strip() for line in f.readlines()]


# Function to download and save the HTML of a single restaurant
def download_html(link, page_folder):
    restaurant_name = link.split("/")[-1]
    file_path = os.path.join(page_folder, f"{restaurant_name}.html")

    # Skip if the file already exists
    if os.path.exists(file_path):
        return None  # No problem, the file already exists
    
    # Download the restaurant's HTML
    response = requests.get(link)
    response.raise_for_status()  # Raise an exception if there is an error
    
    # Save the HTML to the specific page folder
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(response.text)
    return None


# Function to download and save the HTMLs of each restaurant in parallel
def download_html_parallel(restaurant_links, data_folder):
    # Create a subfolder for the HTMLs files
    html_folder = os.path.join(data_folder, 'HTMLs')
    os.makedirs(html_folder, exist_ok=True)
    # Create a progress bar for all the links
    with tqdm(total=len(restaurant_links), desc="Download HTMLs") as progress_bar:
        with ThreadPoolExecutor(max_workers=30) as executor:
            futures = []
            
            # Process each link, adding it to the parallel execution queue
            for link in restaurant_links:
                # Find the page number from the link to create the appropriate folder
                page_number = (restaurant_links.index(link) // 20) + 1
                page_folder = os.path.join(html_folder, f"page_{page_number}")
                if not os.path.exists(page_folder):
                    os.mkdir(page_folder)
                
                futures.append(executor.submit(download_html, link, page_folder))
            
            # Handle task completion with interruption in case of error
            for future in as_completed(futures):
                try:
                    result = future.result()  # Retrieve any exceptions
                except requests.exceptions.RequestException as e:
                    print(f"Error during download: {e}")
                    executor.shutdown(wait=False)  # Stop all ongoing downloads
                    raise e  # Re-raise the error to interrupt the program
                else:
                    progress_bar.update(1)

# Main function to encapsulate everything
def download_html_from_link_file(file_name = 'restaurant_links.txt', data_folder = 'DATA'):
    # Read the restaurant links from the file
    file_path = os.path.join(data_folder, file_name)
    restaurant_links = read_links_from_file(file_path)
    # Start the parallel download for all the links
    download_html_parallel(restaurant_links, data_folder)    
    print("All html files have been saved.")



# ========================================================
#                 THIRD PART: EXTRACTING DATAS INTO TSVs
# - This section gets the relevant datas from the downloaded htmls
#   and store them in tsv files
# ========================================================


# Function to extract data from an HTML
def extract_info_from_html(html):
    soup = BeautifulSoup(html, 'lxml')
    
    # Extract the restaurant name
    title_element = soup.find('h1', class_="data-sheet__title")
    restaurant_name = title_element.text.strip() if title_element else None
    
    # Find the divs with class="data-sheet__block--text"
    blocks = soup.find_all('div', class_="data-sheet__block--text", limit=2)
    
    # Initialize fields
    address = city = postal_code = country = ""
    
    # Extract the address text from the first `div`
    address_text = blocks[0].text.strip()
    
    # Split the components
    components = address_text.split(", ")
    length = len(components)
    address = ', '.join([comp.strip().replace('\n', ' ') for comp in components[0:length-3]])   
    city = components[-3].strip()                                               
    postal_code = components[-2].strip()                                       
    country = components[-1].strip()                                           

    # Handle the second block to extract price range and cuisine type
    price_range = cuisine_type = None

    price_cuisine_text = blocks[1].text.strip()

    price_cuisine_parts = price_cuisine_text.split("·")
    if len(price_cuisine_parts) >= 2:
            price_range = price_cuisine_parts[0].strip()  # Price range
            cuisine_type = price_cuisine_parts[1].strip() # Cuisine type

    # Extract the description
    description_element = soup.find('div', class_='data-sheet__description')
    description = description_element.text.strip().replace('\n', ' ') if description_element else "No description available"

    facilities_services = []

    # Extract services from the ul list under `restaurant-details__services`
    services_section = soup.find('div', class_='restaurant-details__services')
    if services_section:
        # Find the first ul in the section
        ul_element = services_section.find('ul')
        if ul_element:
            for li in ul_element.find_all('li'):
                facilities_services.append(li.text.strip())
    
    credit_cards = []
    
    card_section = services_section.find('div', class_='list--card')
    if card_section:
        for img in card_section.find_all('img', {'data-src': True}):
            # Extract the name of the icon from the image path
            icon_path = img['data-src']
            icon_name = icon_path.split('/')[-1].split('-')[0]  # Get the name before the first '-'
            credit_cards.append(icon_name)
    
    phone_number = None
    phone_tag = soup.find('a', {'data-event': 'CTA_tel'})
    if phone_tag and 'href' in phone_tag.attrs:
        phone_number = phone_tag['href'].replace('tel:', '').strip()  # Remove the 'tel:' prefix

    website = None
    website_tag = soup.find('a', {'data-event': 'CTA_website'})
    if website_tag and 'href' in website_tag.attrs:
        website = website_tag['href'].strip()  # Get the URL and remove extra spaces

    # Return the dictionary with the extracted data
    return {
        'restaurantName': restaurant_name,
        'address': address,
        'city': city,
        'postalCode': postal_code,
        'country': country,
        'priceRange': price_range,
        'cuisineType': cuisine_type,
        'description': description,
        'facilitiesServices': facilities_services, 
        'creditCards': credit_cards, 
        'phoneNumber': phone_number,
        'website': website
    }


# Function to process a single file
def process_file(file_path, restaurant_index, tsv_folder):
    # Define the TSV file path
    tsv_filename = os.path.join(tsv_folder, f'restaurant_{restaurant_index}.tsv')
    
    # Skip processing if the TSV file already exists
    if os.path.exists(tsv_filename):
        return

    with open(file_path, "r", encoding="utf-8") as f:
        html = f.read()
        data = extract_info_from_html(html)

    # Write the data directly to the TSV file
    with open(tsv_filename, "w", encoding="utf-8") as tsv_file:
        # Write the header (keys from the dictionary)
        tsv_file.write('\t'.join(data.keys()) + '\n')
        # Write the values (values from the dictionary)
        tsv_file.write('\t'.join(map(str, data.values())))


# Function to get all the HTML files in the main HTMLs folder and subfolders
def get_html_files_in_directory(directory):
    page_files = []
    
    # Use os.walk() to explore all subfolders and collect the HTML files
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".html"):
                file_path = os.path.join(root, filename)
                page_files.append(file_path)
    
    return page_files


# Function to iterate over all htmls
def html_to_tsv(data_folder='DATA', max_workers=4):
    # Check if the folder with the HTML files exists, if not exit
    html_folder = os.path.join(data_folder, 'HTMLs')
    if not os.path.exists(html_folder):
        print("No 'HTMLs' folder found, unable to process the files.")
        return None

    # Ensure the 'TSV' folder exists
    tsv_folder = os.path.join(data_folder, 'TSVs')
    os.makedirs(tsv_folder, exist_ok=True)

    # Get all the HTML files in the 'HTML' folder (including subfolders)
    html_files = get_html_files_in_directory(html_folder)

    # Define a function to process a single file
    def process_file_wrapper(file_tuple):
        file_path, index = file_tuple
        process_file(file_path, index, tsv_folder)

    # Create a list of tuples containing file paths and their indices
    file_tuples = [(file_path, index) for index, file_path in enumerate(html_files, start=1)]

    # Use ThreadPoolExecutor to parallelize the processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Use tqdm to show the progress bar for the parallelized task
        list(tqdm(executor.map(process_file_wrapper, file_tuples), total=len(html_files), desc="Processing HTMLs"))

    print("All files have been processed and saved.")

# Function to create the dataframe for the dataset
def create_combined_dataframe(folder_path, separator):
    """
    Creates a combined DataFrame from all .tsv files in a specified folder.

    Parameters:
    - folder_path: Path to the folder containing .tsv files.
    - separator: Delimiter used in the .tsv files.

    Returns:
    - DataFrame containing all combined data.
    """
    # Find all .tsv files in the specified folder
    all_files = glob.glob(os.path.join(folder_path, "*.tsv"))

    # Load each .tsv file as a DataFrame and store in a list
    df_list = [pd.read_csv(file, sep=separator) for file in all_files]
    
    # Concatenate all DataFrames in the list into a single DataFrame
    combined_df = pd.concat(df_list, ignore_index=True)

    return combined_df


# ========================================================
#                 SECOND EXERCISE: SEARCH ENGINE
# ========================================================


from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from nltk.stem import SnowballStemmer
import re
import json

# Function to preprocess restaurant descriptions by removing stopwords, cleaning punctuation, and applying stemming to improve search efficiency.
def preprocess_text(descriptions):
    """
    Preprocesses a list of restaurant descriptions by:
    - Tokenizing each description
    - Removing stopwords
    - Cleaning tokens of punctuation
    - Stemming each word to its root form
    
    Parameters:
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
    Creates a vocabulary file in CSV format, mapping each unique word (term) in the processed texts to a unique integer ID.

    Parameters:
    processed_texts (list of list of str): A list of lists, where each sublist contains tokenized and processed words from a description.

    Returns:
    pd.DataFrame: A DataFrame containing the vocabulary, with each word mapped to a unique integer ID.
    """
    
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
            inverted_index = json.load(f)
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
def execute_query(query, inverted_index, vocabulary_df):
    """
    Executes a search query on an inverted index to find documents that contain all the terms in the query.
    
    Parameters:
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
    terms_id = vocabulary_df[vocabulary_df['term'].isin(query_list)]['term_id'].astype(str).tolist()
    
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
