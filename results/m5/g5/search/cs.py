import json, csv, time, urllib.parse, urllib.request
API="https://commons.wikimedia.org/w/api.php"
UA={"User-Agent":"SerpentGuardStudentProject/1.0 (university coursework)"}
def get(p):
    url=API+"?"+urllib.parse.urlencode({**p,"format":"json"})
    for k in range(8):
        try:
            r=json.load(urllib.request.urlopen(urllib.request.Request(url,headers=UA),timeout=40)); time.sleep(2.5); return r
        except Exception as e:
            print("retry",e,flush=True); time.sleep(10*(k+1))
sp=[r["species"] for r in csv.DictReader(open("m5-g5/data/classes.csv"))]
common=["cobra","krait","Russell's viper","saw-scaled viper","rat snake","python snake","keelback","vine snake","wolf snake","kukri snake","sea snake","pit viper","snake charmer","snake swimming","king cobra","snake slithering","sand boa","trinket snake","checkered keelback","snake feeding","viper","water snake"]
out={}
for q in sp+common:
    r=get({"action":"query","list":"search","srnamespace":6,"srsearch":f'"{q}" filetype:video',"srlimit":50})
    hits=[h["title"] for h in r["query"]["search"]]
    out[q]=hits
    if hits: print(q,hits,flush=True)
    json.dump(out,open("cs_search.json","w"),indent=1)
