from __future__ import print_function

import os
import requests
import re
from zipfile import ZipFile
from io import BytesIO, StringIO

from riksmot import DATA_PATH

print("Dowloading motioner into:\n\t{}".format(DATA_PATH))

riksdag_doc_url = "http://data.riksdagen.se/data/dokument/"
content = requests.get(riksdag_doc_url).content
match = re.findall('href="([a-z./]+mot[0-9\-]+\.text\.zip)',
                   str(content))

print("Getting archives")
for zip_url in match[:1]:
    print("\thttp:" + zip_url)
    zip_content = requests.get("http:" + zip_url).content

    with ZipFile(BytesIO(zip_content)) as zip:
        zip.extractall(DATA_PATH)

print("Done")
