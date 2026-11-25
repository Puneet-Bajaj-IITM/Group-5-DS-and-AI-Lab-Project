import json,sys,time,urllib.request,os
m=json.load(open("search/meta1.json"))
want=sys.argv[1:]
for t,v in m.items():
    if not any(w in t for w in want): continue
    fn="raw/"+t[5:].replace("/","_")
    if os.path.exists(fn) and os.path.getsize(fn)>0: continue
    req=urllib.request.Request(v["url"],headers={"User-Agent":"SerpentGuardStudentProject/1.0 (university coursework)"})
    for k in range(8):
        try:
            b=urllib.request.urlopen(req,timeout=120).read(); open(fn,"wb").write(b); print("ok",fn,flush=True); break
        except Exception as e: print("retry",e,flush=True); time.sleep(60*(k+1))
    time.sleep(8)
