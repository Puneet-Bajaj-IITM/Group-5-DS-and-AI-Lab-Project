import json, sys, time, urllib.parse, urllib.request, re
API="https://commons.wikimedia.org/w/api.php"
UA={"User-Agent":"SerpentGuardStudentProject/1.0 (university coursework)"}
def get(p):
    url=API+"?"+urllib.parse.urlencode({**p,"format":"json"})
    for k in range(8):
        try:
            r=json.load(urllib.request.urlopen(urllib.request.Request(url,headers=UA),timeout=40)); time.sleep(3); return r
        except Exception as e:
            print("retry",e,file=sys.stderr,flush=True); time.sleep(10*(k+1))
titles=[l.strip() for l in open(sys.argv[1]) if l.strip()]
out={}
for t in titles:
    r=get({"action":"query","titles":t,"prop":"imageinfo|categories","iiprop":"url|size|extmetadata|mime","iimetadataversion":"latest","cllimit":50})
    for pid,pg in r["query"]["pages"].items():
        ii=(pg.get("imageinfo") or [{}])[0]; em=ii.get("extmetadata",{})
        g=lambda k: re.sub("<[^>]+>","",em.get(k,{}).get("value",""))[:600]
        out[t]={"url":ii.get("url"),"size":ii.get("size"),"dur":(ii.get("duration")),"lic":g("LicenseShortName"),"licurl":g("LicenseUrl"),"artist":g("Artist"),"desc":g("ImageDescription"),"date":g("DateTimeOriginal"),"cats":[c["title"] for c in pg.get("categories",[])]}
        print(t,"|",out[t]["lic"],"|",out[t]["artist"][:50],"|",out[t]["desc"][:300],"|",[c[9:] for c in out[t]["cats"] if not c.startswith(("Category:CC","Category:License","Category:Self","Category:WebM","Category:Video","Category:Uploaded","Category:Flickr"))] ,"\n",flush=True)
json.dump(out,open(sys.argv[2],"w"),indent=1,ensure_ascii=False)
