import requests
from bs4 import BeautifulSoup
import time
from tqdm import tqdm
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import random
import sys

import aiohttp
import asyncio
from aiohttp import ClientSession



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
    address = ', '.join([comp.strip() for comp in components[0:length-3]])      # address
    city = components[-3].strip()                                               # city
    postal_code = components[-2].strip()                                        # postal code
    country = components[-1].strip()                                            # country

    # Handle the second block to extract price range and cuisine type
    price_range = cuisine_type = ""

    price_cuisine_text = blocks[1].text.strip()

    price_cuisine_parts = price_cuisine_text.split("·")
    if len(price_cuisine_parts) >= 2:
            price_range = price_cuisine_parts[0].strip()  # Price range
            cuisine_type = price_cuisine_parts[1].strip() # Cuisine type

    # Extract the description
    description_element = soup.find('div', class_='data-sheet__description')
    description = description_element.text.strip() if description_element else "No description available"

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

# # Function to process a single file
# def process_file(file_path, restaurant_index, tsv_folder):
#     with open(file_path, "r", encoding="utf-8") as f:
#         html = f.read()
#         data = extract_info_from_html(html)
        
#     # Create a DataFrame for each restaurant and save it as a TSV file
#     df = pd.DataFrame([data])
#     tsv_filename = os.path.join(tsv_folder, f'restaurant_{restaurant_index}.tsv')
#     df.to_csv(tsv_filename, sep='\t', index=False, header=True)


# Function to process a single file
def process_file(file_path, restaurant_index, tsv_folder):
    with open(file_path, "r", encoding="utf-8") as f:
        html = f.read()
        data = extract_info_from_html(html)

    # Define the TSV file path
    tsv_filename = os.path.join(tsv_folder, f'restaurant_{restaurant_index}.tsv')

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



def html_to_tsv(data_folder='DATA'):
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

    # Process each file with a progress bar
    for index, file_path in tqdm(enumerate(html_files, start=1), total=len(html_files), desc="Processing HTMLs"):
        process_file(file_path, index, tsv_folder)

    print("All files have been processed and saved.")






    