# -*- coding: utf-8 -*-
"""scraper

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1B2nariu1hNyFmLAqD-tB4sUbfQgpA1sZ
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

def scrape_karkidi_jobs(keyword="data science", pages=1):
    headers = {'User-Agent': 'Mozilla/5.0'}
    base_url = "https://www.karkidi.com/Find-Jobs/{}/all/India?search={}"
    jobs_list = []

    for page in range(1, pages + 1):
        url = base_url.format(page, keyword.replace(' ', '%20'))
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, "html.parser")

        job_blocks = soup.find_all("div", class_="ads-details")
        for job in job_blocks:
            try:
                title = job.find("h4").get_text(strip=True)
                company = job.find("a", href=lambda x: x and "Employer-Profile" in x).get_text(strip=True)
                location = job.find("p").get_text(strip=True)
                experience = job.find("p", class_="emp-exp").get_text(strip=True)
                skills_tag = job.find("span", string="Key Skills")
                skills = skills_tag.find_next("p").get_text(strip=True) if skills_tag else ""
                summary_tag = job.find("span", string="Summary")
                summary = summary_tag.find_next("p").get_text(strip=True) if summary_tag else ""

                jobs_list.append({
                    "Title": title, "Company": company, "Location": location,
                    "Experience": experience, "Summary": summary, "Skills": skills
                })
            except Exception as e:
                print("Error parsing:", e)
                continue

        time.sleep(1)
    return pd.DataFrame(jobs_list)