#Quick script to scrape recipie page html's

import os, sys, re
import wget, urllib

from_ind = int(sys.argv[1])
to_ind = int(sys.argv[2])

url_base = "https://www.allrecipes.com/recipe/"
for ext in range(from_ind,to_ind):
    url =  url_base + str(ext)

    #for url in infile:
    print(url)
    try:
        filename = wget.download(url.replace("\n",""))
        print(filename)
    except urllib.error.HTTPError:
        print("FAILED TO DOWNLOAD!")

