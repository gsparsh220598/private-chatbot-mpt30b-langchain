import requests
from bs4 import BeautifulSoup
import os

# Define the URL of the webpage to scrape
url = "https://distill.pub/"

# Send a GET request to the webpage
response = requests.get(url)

# Parse the HTML content using BeautifulSoup
soup = BeautifulSoup(response.content, "html.parser")

# Find the div element with class "posts-list l-page"
div_element = soup.find("div", class_="posts-list l-page")

# Find all the div elements within the main div
inner_divs = div_element.find_all("div")

# Create a directory to store the downloaded HTML files
download_dir = "./source_documents/distillpub/"
os.makedirs(download_dir, exist_ok=True)

# Extract and download the hyperlinks
for div in inner_divs:
    # Find all the anchor tags within the div
    anchor_tags = div.find_all("a")

    for anchor_tag in anchor_tags:
        # Get the href attribute value
        href = anchor_tag.get("href")

        if href:
            if "issues" not in href:
                # Download the file
                file_url = url + href + "/"  # Assuming the href is a relative URL
                file_name = os.path.join(download_dir, href.split("/")[-1]) + ".html"
                response = requests.get(file_url)

                with open(file_name, "wb") as f:
                    f.write(response.content)

                print(f"Downloaded: {file_name}")
