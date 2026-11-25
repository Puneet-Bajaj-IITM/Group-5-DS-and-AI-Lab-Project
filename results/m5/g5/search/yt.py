import json, subprocess, time, urllib.parse, sys, os
qs=["Naja naja","Naja kaouthia monocled cobra","Ophiophagus hannah king cobra","Bungarus caeruleus common krait","Bungarus fasciatus banded krait","Daboia russelii Russell's viper","Echis carinatus saw-scaled viper","Ptyas mucosa rat snake","Python molurus Indian rock python","Python bivittatus Burmese python","Malayopython reticulatus reticulated python","Ahaetulla nasuta green vine snake","Boiga trigonata","Lycodon aulicus wolf snake","Dendrelaphis tristis bronzeback","Fowlea piscator checkered keelback","Coelognathus helena trinket snake","Coelognathus radiatus copperhead rat snake","Trimeresurus pit viper","Craspedocephalus malabaricus Malabar pit viper","Hydrophis sea snake","Hydrophis schistosus beaked sea snake","Hydrophis curtus","Eryx johnii sand boa","Eryx conicus rock sand boa","Indotyphlops braminus brahminy blind snake","Chrysopelea ornata flying snake","Gonyosoma oxycephalum","Elaphe taeniura","Oligodon arnensis banded kukri","Boiga forsteni cat snake","Sibynophis subpunctatus","Atretium schistosum olive keelback","Amphiesma stolatum buffstriped keelback","Xenochrophis cerberus","Cerberus rynchops dog-faced water snake","Enhydris enhydris","Hypsiscopus plumbeus","Rhabdophis subminiatus red-necked keelback","Pareas monticola","Psammodynastes pulverulentus","Calamaria","Lycodon fasciatus banded wolf snake","Naja oxiana","Hydrophis platurus yellow-bellied sea snake","Laticauda sea krait","Bungarus niger greater black krait","Hemibungarus","Oreocryptophis porphyraceus","Dryocalamus","Ahaetulla prasina","Boiga ceylonensis","Python snake India CC BY","snake Kerala","snake Sri Lanka","snake Thailand wild","snake Nepal","snake Bangladesh","snake Assam","snake Western Ghats","snake Andaman","rat snake India","cobra India wild"]
seen={}
if os.path.exists("yt_cands.json"): seen=json.load(open("yt_cands.json"))
done=set(seen.get("_q",[])) if seen else set()
seen.setdefault("_q",[])
for q in qs:
    if q in done: continue
    u="https://www.youtube.com/results?search_query="+urllib.parse.quote_plus(q)+"&sp=EgIwAQ%253D%253D"
    p=subprocess.run(["yt-dlp","--flat-playlist","--playlist-end","12","-J",u],capture_output=True,text=True)
    try: d=json.loads(p.stdout)
    except Exception: print("fail",q,p.stderr[:200],flush=True); time.sleep(5); continue
    n=0
    for e in d["entries"]:
        vid=e["id"]
        if vid in seen: seen[vid]["queries"].append(q); continue
        if (e.get("duration") or 0)>1500 or (e.get("duration") or 0)<5: 
            seen[vid]={"skip":"duration","title":e.get("title"),"dur":e.get("duration"),"queries":[q]}; continue
        time.sleep(1.5)
        m=subprocess.run(["yt-dlp","--skip-download","-J","--no-warnings","--ignore-no-formats-error","--extractor-args","youtube:player_client=android",f"https://www.youtube.com/watch?v={vid}"],capture_output=True,text=True)
        try: j=json.loads(m.stdout)
        except Exception: j=None
        if not j: seen[vid]={"skip":"meta-fail","queries":[q]}; continue
        seen[vid]={"title":j.get("title"),"license":j.get("license"),"uploader":j.get("uploader"),"dur":j.get("duration"),"desc":(j.get("description") or "")[:700],"tags":(j.get("tags") or [])[:25],"date":j.get("upload_date"),"queries":[q]}
        n+=1
    seen["_q"].append(q); print(q,len(d["entries"]),"new",n,flush=True)
    json.dump(seen,open("yt_cands.json","w"),indent=1,ensure_ascii=False)
