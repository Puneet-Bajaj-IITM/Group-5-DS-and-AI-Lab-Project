import json, time, urllib.parse, urllib.request
API="https://commons.wikimedia.org/w/api.php"
UA={"User-Agent":"SerpentGuardStudentProject/1.0 (university coursework)"}
def get(p):
    url=API+"?"+urllib.parse.urlencode({**p,"format":"json"})
    for k in range(8):
        try:
            r=json.load(urllib.request.urlopen(urllib.request.Request(url,headers=UA),timeout=40)); time.sleep(3)
            if "query" in r: return r
            print("noquery",str(r)[:200],flush=True)
        except Exception as e:
            print("retry",e,flush=True); time.sleep(10*(k+1))
    return None
qs=["snake","snakes","serpent","ular","India snake","Thailand snake","Sri Lanka snake","Nepal snake","Bangladesh snake","python","viper","krait","cobra snake","slithering","reptile","herpetology","snake rescue","venomous snake","non-venomous snake","sea snake","water snake","tree snake","whip snake","Naja","Bungarus","Daboia","Echis","Trimeresurus","Hydrophis","Ahaetulla","Boiga","Lycodon","Dendrelaphis","Coelognathus","Ptyas","Fowlea","Eryx","Gongylophis","Oligodon","Sibynophis","Amphiesma","Atretium","Enhydris","Cerberus","Homalopsis","checkered keelback","green keelback","chrysopelea flying snake","flying snake","blind snake","reptile house","snake farm","snake park","zoo snake"]
out={}
for q in qs:
    titles=[]; off=0
    while True:
        r=get({"action":"query","list":"search","srnamespace":6,"srsearch":f"{q} filetype:video","srlimit":100,"sroffset":off})
        if not r: break
        titles+= [h["title"] for h in r["query"]["search"]]
        if "continue" in r and off<300: off=r["continue"]["sroffset"]
        else: break
    out[q]=titles; print(q,len(titles),flush=True)
    json.dump(out,open("cs2_search.json","w"),indent=1)
