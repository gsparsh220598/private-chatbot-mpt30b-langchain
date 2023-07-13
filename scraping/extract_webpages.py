import requests
from bs4 import BeautifulSoup
import os

# Define the URL of the webpage to scrape
url = "https://www.neelnanda.io/mechanistic-interpretability/"

# Send a GET request to the webpage
response = requests.get(url)

# Parse the HTML content using BeautifulSoup
soup = BeautifulSoup(response.content, "html.parser")

# Change the tag below to your desired tag
div_element = soup.find("div", class_="blog-basic-grid collection-content-wrapper")

# Find all the div elements within the main div
inner_divs = div_element.find_all("article")

# Create a directory to store the downloaded HTML files
download_dir = "./source_documents/neelnanda/mechanistic-interpretability/"
os.makedirs(download_dir, exist_ok=True)

href_list = []
# Extract and download the hyperlinks
for div in inner_divs:
    # Find all the anchor tags within the div
    anchor_tags = div.find_all("a")

    for anchor_tag in anchor_tags:
        # Get the href attribute value
        href = anchor_tag.get("href")
        if href not in href_list:
            if "http" not in href:
                href_list.append(href)
                # print(href.split("/")[-1])
                # Download the file
                file_url = (
                    url + href.split("/")[-1]
                )  # Assuming the href is a relative URL
                # print(file_url)
                file_name = os.path.join(download_dir, href.split("/")[-1]) + ".html"
                response = requests.get(file_url)

                with open(file_name, "wb") as f:
                    f.write(response.content)

                print(f"Downloaded: {file_name}")
