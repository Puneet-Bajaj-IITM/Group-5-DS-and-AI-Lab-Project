import json, time, urllib.parse, urllib.request
UA={"User-Agent":"SerpentGuardStudentProject/1.0 (university coursework)"}
def get(url):
    for k in range(6):
        try:
            r=json.load(urllib.request.urlopen(urllib.request.Request(url,headers=UA),timeout=60)); time.sleep(2); return r
        except Exception as e:
            print("retry",e,flush=True); time.sleep(8*(k+1))
terms=["cobra","krait","Russell's viper","saw-scaled viper","rat snake","python","keelback","vine snake","king cobra","sea snake","pit viper","snake","Naja","Bungarus","Daboia","Echis","Ptyas","Ahaetulla","Python molurus","Boiga","Trimeresurus","Hydrophis","Lycodon","Fowlea","Amphiesma","Oligodon","Eryx"]
out={}
for t in terms:
    q=f'(title:({t}) OR subject:({t}) OR description:({t})) AND mediatype:movies AND (licenseurl:*creativecommons.org* OR licenseurl:*publicdomain*)'
    u="https://archive.org/advancedsearch.php?"+urllib.parse.urlencode({"q":q,"fl[]":["identifier","title","licenseurl","creator"],"rows":100,"output":"json"},doseq=True)
    r=get(u)
    docs=r["response"]["docs"] if r else []
    out[t]=docs
    print(t,len(docs),flush=True)
    json.dump(out,open("ia_search.json","w"),indent=1)
