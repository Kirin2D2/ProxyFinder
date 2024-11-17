#scraper
import requests
from bs4 import BeautifulSoup

# Base URL for the department faculty pages
base_url = "https://cla.umn.edu/{department}/people/faculty"

# Departments to scan (you can add more department slugs as needed)
departments = ['sociology']

# Keywords to search for
keywords = ["marx", "marxist", "marxism"]

def get_faculty_bios(department_url):
    response = requests.get(department_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find all links to faculty bios
    faculty_links = soup.find_all('a', href=True)
    
    bios = []
    
    for link in faculty_links:
        if 'profile' in link['href']:  # Adjust this condition based on actual URL structure
            full_url = link['href']
            if not full_url.startswith("http"):  # Handle relative URLs
                full_url = "https://cla.umn.edu" + full_url
            bios.append(scrape_bio(full_url))
    
    return bios

def scrape_bio(bio_url):
    response = requests.get(bio_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Extract the main text of the bio
    bio_text = soup.get_text()
    
    return bio_url, bio_text

def search_for_keywords(bios, keywords):
    matches = []
    
    for bio_url, bio_text in bios:
        if any(keyword.lower() in bio_text.lower() for keyword in keywords):
            matches.append(bio_url)
    
    return matches

def main():
    matched_faculty = []

    for department in departments:
        department_url = base_url.format(department=department)
        bios = get_faculty_bios(department_url)
        matches = search_for_keywords(bios, keywords)
        matched_faculty.extend(matches)
    
    print("Faculty with keywords in their bios:")
    for match in matched_faculty:
        print(match)

if __name__ == "__main__":
    main()
