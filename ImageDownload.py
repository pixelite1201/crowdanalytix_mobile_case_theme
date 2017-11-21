from bs4 import BeautifulSoup
import requests
import urllib2
import os
import json
import time
from PIL import Image
import cStringIO



TRAIN='./Train'
VALIDATION='./Validation'
TRAIN_DOWNLOAD='./Train_Download'
VAL_DOWNLOAD='./Val_Download'
if not os.path.exists(TRAIN_DOWNLOAD):
    os.mkdir(TRAIN_DOWNLOAD)
if not os.path.exists(VAL_DOWNLOAD):
    os.mkdir(VAL_DOWNLOAD)
header={'User-Agent':"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36"
}

def get_soup(url,header):
    return BeautifulSoup(urllib2.urlopen(urllib2.Request(url,headers=header)),'html.parser')

for theme in os.listdir(VALIDATION):
    path=os.path.join(VALIDATION,theme)
    DIR = os.path.join(VAL_DOWNLOAD, theme)
    if not os.path.exists(DIR):
        os.mkdir(DIR)
    if len(os.listdir(path)) > 90:
        max_limit=10
    else:
        max_limit=15
    for my_img in os.listdir(path):
        filePath=os.path.join(path,my_img)
        searchUrl = 'http://www.google.hr/searchbyimage/upload'
        multipart = {'encoded_image': (filePath, open(filePath, 'rb')), 'image_content': ''}
        response = requests.post(searchUrl, files=multipart, allow_redirects=False)
        fetchUrl = response.headers['Location']
        url=fetchUrl
        soup = get_soup(url,header)
        ActualImages=[]# contains the link for Large original images, type of  image
        for a in soup.find_all("div",{"class":"rg_meta"}):
            link =json.loads(a.text)["ou"]
            if len(ActualImages) >= max_limit:
                break
            ActualImages.append(link)
        print  "there are total" , len(ActualImages),"images"
        for i , img in enumerate( ActualImages):
            try:
                print img
                req = urllib2.Request(img, headers={'User-Agent' : header})
                raw_img = cStringIO.StringIO(urllib2.urlopen(req).read())
                im = Image.open(raw_img).convert('RGBA')
                bg = Image.new("RGBA", im.size, (255, 255, 255))
                bg.paste(im,im)
                bg.save(os.path.join(DIR, "Augment" + my_img.split('.')[0]+str(i)+".jpg"))
            except Exception as e:
                print "could not load : "+img
                print e
        time.sleep(2)
