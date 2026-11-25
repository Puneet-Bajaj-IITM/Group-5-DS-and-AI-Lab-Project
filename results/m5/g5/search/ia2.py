import json, time, urllib.parse, urllib.request
UA={"User-Agent":"SerpentGuardStudentProject/1.0 (university coursework)"}
def get(url):
    for k in range(6):
        try:
            r=json.load(urllib.request.urlopen(urllib.request.Request(url,headers=UA),timeout=60)); time.sleep(2); return r
        except Exception as e:
            print("retry",e,flush=True); time.sleep(8*(k+1))
out={}
for t in ["milking venom slow motion","Venom in Slow Motion","snake venom milking","Bungarus","krait milking","viper milking","cobra venom","Indian cobra","monocled cobra","Russell","Daboia russelii","Echis carinatus","Ptyas mucosa","Python molurus","Python bivittatus","reticulated python","Ophiophagus hannah","Trimeresurus","Hydrophis","Boiga","Dendrelaphis","Lycodon","Coelognathus","Fowlea piscator","checkered keelback","green vine snake","sand boa","snake rescue India","snake charmer"]:
    q=f'(title:("{t}") OR subject:("{t}") OR description:("{t}")) AND mediatype:movies AND (licenseurl:*creativecommons.org* OR licenseurl:*publicdomain*)'
    u="https://archive.org/advancedsearch.php?"+urllib.parse.urlencode({"q":q,"fl[]":["identifier","title","licenseurl","creator"],"rows":60,"output":"json"},doseq=True)
    r=get(u); docs=r.get("response",{}).get("docs",[]) if r else []
    out[t]=docs; print(t,len(docs),flush=True)
    for x in docs[:60]:
        if any(w in x["title"].lower() for w in ["snake","cobra","krait","viper","python","venom","naja","keelback","boa","serpent","reptile"]): print("   ",x["identifier"],"|",x["title"][:90],"|",x.get("licenseurl"))
    json.dump(out,open("ia2_search.json","w"),indent=1)
