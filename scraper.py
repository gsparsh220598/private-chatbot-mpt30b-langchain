import requests
from bs4 import BeautifulSoup
import os
import zipfile
import shutil

## !wget -r -A.html -P lillog https://lilianweng.github.io

base_url = "https://ocw.mit.edu/courses/"

course_list = [
    "18-657-mathematics-of-machine-learning-fall-2015",
    "18-06sc-linear-algebra-fall-2011",
    "15-450-analytics-of-finance-fall-2010",
    # "18-01-calculus-i-single-variable-calculus-fall-2020",
    "18-02sc-multivariable-calculus-fall-2010",
    "18-03sc-differential-equations-fall-2011",
    "2-087-engineering-math-differential-equations-and-linear-algebra-fall-2014",
    "6-871-knowledge-based-applications-systems-spring-2005",
    "6-036-introduction-to-machine-learning-fall-2020",
    "9-520-statistical-learning-theory-and-applications-spring-2003",
    "18-650-statistics-for-applications-fall-2016",
    "6-801-machine-vision-fall-2020",
    "15-s08-fintech-shaping-the-financial-world-spring-2020",
    "18-s096-topics-in-mathematics-with-applications-in-finance-fall-2013",
    "2-035-special-topics-in-mathematics-with-applications-linear-algebra-and-the-calculus-of-variations-spring-2007",
    "6-041sc-probabilistic-systems-analysis-and-applied-probability-fall-2013",
    "6-262-discrete-stochastic-processes-spring-2011",
    "6-034-artificial-intelligence-fall-2010",
    "18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018",
    "res-ll-005-mathematics-of-big-data-and-machine-learning-january-iap-2020",
    "15-071-the-analytics-edge-spring-2017",
    "6-253-convex-analysis-and-optimization-spring-2012",
    "15-093j-optimization-methods-fall-2009",
    "6-863j-natural-language-and-the-computer-representation-of-knowledge-spring-2003",
    "9-98-language-and-mind-january-iap-2003",
    "6-864-advanced-natural-language-processing-fall-2005",
    "9-913-pattern-recognition-for-machine-vision-fall-2004",
]


def generate_course_urls(course_names):
    urls = []

    for course_name in course_names:
        course_slug_ls = course_name.split("-")
        course_slug = (
            course_slug_ls[0]
            + "."
            + course_slug_ls[1]
            + "-"
            + course_slug_ls[-2]
            + "-"
            + course_slug_ls[-1]
            + ".zip"
        )

        # Generate the URL for the course
        course_url = base_url + course_name + "/" + course_slug
        # Append the URL to the list
        urls.append(course_url)

    return urls


def download_course_files(course_list):
    course_urls = generate_course_urls(course_list)
    count = 0
    for url, name in zip(course_urls, course_list):
        response = requests.get(url)
        # Ensure that the request was successful
        if response.status_code == 200:
            # check if directory exists
            if not os.path.exists(f"source_documents/MITCourse/{name}"):
                # Create a temporary file to save the ZIP contents
                with open(f"source_documents/MITCourse/zips/{name}.zip", "wb") as file:
                    file.write(response.content)

                # Extract the contents of the ZIP file
                with zipfile.ZipFile(
                    f"source_documents/MITCourse/zips/{name}.zip", "r"
                ) as zip_ref:
                    zip_ref.extractall(f"source_documents/MITCourse/{name}/")

                print(f"ZIP file downloaded and extracted successfully for {name}.")
                count += 1
            else:
                print(f"Course {name} already exists")
        else:
            print(f"Failed to download the ZIP file for course: {name} with url {url}.")
    return count


def delete_files_except_directory(directory_path, exception_directory):
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            if os.path.isdir(file_path):
                continue  # Skip directories

            if not file_path.startswith(exception_directory):
                os.remove(file_path)

        for dir in dirs:
            dir_path = os.path.join(root, dir)
            if dir_path != exception_directory:
                shutil.rmtree(dir_path)

    print("Deletion completed.")


if __name__ == "__main__":
    n = download_course_files(course_list)
    if n == 0:
        print("No new courses to download")
        # provide a list of courses as input to download
        courses = input("Provide course slug for the course you want to download: ")
        course_list = courses.split(",")
        n = download_course_files(course_list)
        print(f"Downloaded {n} new courses")
    else:
        print(f"Downloaded {n} courses")
