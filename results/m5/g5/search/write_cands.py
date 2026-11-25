import csv, json, re
G="results/m5/g5/"
acc={r["clip"]:r for r in csv.DictReader(open(G+"new_clips.csv"))}
neg={r["clip"]:r for r in csv.DictReader(open(G+"new_negatives.csv"))}
acc_urls={r["source_url"]:"ACCEPT (snake) as "+c for c,r in acc.items()}
acc_urls.update({r["source_url"]:"ACCEPT (negative) as "+c for c,r in neg.items()})
rows=[]
# Commons
allt=json.load(open(G+"search/all_titles.json"))
kw=re.compile(r"snake|serpent|cobra|krait|viper|python|naja|keelback|\bboa\b|ular\b|ophi|ahaetulla|boiga|lycodon|dendrel|ptyas|bungarus|echis|daboia|trimeres|hydrophis|reptile|slither|constrictor|anaconda|eel|limax|lizard|Maerim|Homalopsis|Zmija|Nyoka",re.I)
skip=re.compile(r"Kin Wikipedia|WP25|Cobra Gold|Juniper Cobra|State of the Union|Python \d|Python Perf|Python for|Python[- ]based|in Python|Python vs|Python Fac|Python lect|Python and|Python day|Pythonic|Python script|Python plug|Python,|Python \(|MicroPython|^File:.*(Algorithm|Алгоритм|Практика)|Street Fighter|Kombat|AH-1|Viper (Demo|25)|F-16|Cobra's Curse|Von Karman|CIRA|Snake River|Snake Island|Snakeboat|Snake Boat|Snake Pit",re.I)
reasons={
"Maerim":"reject: species not stated (Elapidae, snake show); snake behind wire mesh / dim; not viewed beyond contact sheet",
"Homalopsis":"reject: Homalopsis buccata not in class list",
"Boiga cynodon":"reject: Boiga cynodon not in class list",
"Sea snakes and marine turtles":"reject: Laticauda laticaudata page, but contact sheet of 31418 shows no snake in frame at 3 s steps (underwater reef)",
"Snakes twisted":"reject: mating snakes in leaf litter, species unknown, poorly visible",
"GREEN VINE SNAKE":"reject: only category 'Snakes of Kerala'; genus Ahaetulla but species (nasuta/borealis) not clear; snake visible only late in clip",
"Green vine snake (Ahaetulla nasuta)":"reject: Ahaetulla nasuta in broad old sense (Seattle zoo)",
"Green vine snake.webm":"reject: already used in old test set",
"Rat snake swimming":"reject: already used","Cobra video 2017":"reject: already used","Snake charmer in Jaipur":"reject: already used",
"Burmesepython":"reject: already used","Colubrine sea krait":"reject: already used","Sea Snake eating Moray":"reject: already used",
"Red-tailed green ratsnake":"reject: already used","Milking Banded Krait":"reject: already used (old set)",
"Natuurwonderen":"reject: already used","Milking Adult Malayan":"reject: Calloselasma rhodostoma not in class list","Milking Juvenile Malayan":"reject: Calloselasma rhodostoma not in class list",
"spitting cobra (Naja sputatrix)":"reject: Naja sputatrix not in class list",
"crossing a path":"reject: same Singapore session as accepted king cobra clip; snake thin and far from camera",
"Two snakes.ogv":"reject: unidentified Serpentes (southern India)",
"Snake charmer":"reject: performer/people","Snake queen":"reject: performer, species unknown",
"THAILAND - Snakes":"reject: travel compilation, species unknown",
"Fowlea melanzostus":"reject: Fowlea melanzostus not in class list",
"Rat snake drinking":"ACCEPT as ptyas_mucosa_india_drinking (species inferred)",
"Hierophis|Natrix|Thamnophis|Vipera|Crotalus|rattlesnake|garter|Nerodia|Elaphe climacophora|Elaphe obsoleta|Pituophis|Malpolon|Pseudonaja|Eunectes|Boa constrictor|Boa Strike|Boa eating|Heterodon|Morelia|Antaresia|Bitis|Cerastes|Pseudocerastes|Spilotes|Micrurus|Dice snake|Python regius|Ball python|Python royal|Pythons royaux|Cannibal python":"reject: species not in class list (non-South-Asian)",
"2 eels":"ACCEPT as negative_eels_hawaii","Limax":"ACCEPT as negative_slug_austria",
}
for t in allt:
    if not kw.search(t) or skip.search(t): continue
    why=None
    for k,v in reasons.items():
        if re.search(k,t): why=v; break
    if t.startswith("File:King Cobra (Ophiophagus hannah) hunting"): why="ACCEPT as ophiophagus_hannah_singapore_hunting"
    if not why: why="reject: not a snake video of an in-list species (unrelated, other taxon, or non-wildlife)"
    rows.append(["Commons",t,"https://commons.wikimedia.org/wiki/"+t.replace(" ","_"),why])
nc=len(rows)
# Internet Archive
ia={}
for fn in ["ia_search.json","ia2_search.json"]:
    for q,docs in json.load(open(G+"search/"+fn)).items():
        for x in docs: ia.setdefault(x["identifier"],x)
kwi=re.compile(r"snake|cobra|krait|viper|python|venom|naja|keelback|serpent|reptile|Fowlea",re.I)
for i,x in ia.items():
    if not kwi.search(x["title"]): continue
    lic=x.get("licenseurl","")
    if re.search(r"by-nc|by-nd|nc-sa",lic): why="reject: NC/ND licence"
    elif re.search(r"Street Fighter|Kombat|Twin Cobra|King of Fighters|Garou|JoJo|Sega|Nintendo|cobra-kai|Python (Recumbent|Performance)|pyc|pygotham|Serverless|slapd|Mustang|Chrysler|Muppet|NBC|Marvel|THE SNAKE WOMAN",x["title"],re.I) or re.search(r"pyconza|pytx|pygotham|JFS|diamond-cobra|twin-cobra",i,re.I): why="reject: not a real snake video (game, talk, car, show)"
    elif i=="snake-milking": why="reject: Bungarus fasciatus/Calloselasma milking, same session as already used source (Scenoptica); long documentary with people"
    elif "ephemeral-pond" in i: why="reject: Fowlea melanzostus / not in class list"
    elif "milking" in i: why="reject: milking video; Calloselasma not in class list or already-used source"
    elif "garter" in i: why="reject: garter snake (not in class list)"
    elif "keelbacks" in i: why="reject: Hong Kong yellow-spotted keelback (Fowlea flavipunctatus) not in class list"
    elif "spitting-cobra" in i: why="reject: Naja sputatrix not in class list"
    elif "Flickr-6114673404" in i: why="reject: Pit viper at Arenal volcano, Costa Rica"
    else: why="reject: not a usable in-list snake video"
    rows.append(["Internet Archive",x["title"][:150],"https://archive.org/details/"+i,why+" ("+lic+")"])
ni=len(rows)-nc
# YouTube
y1=json.load(open(G+"search/yt_cands.json")); y2=json.load(open(G+"search/yt_cands2.json"))
yt={**{k:v for k,v in y1.items() if k!="_q"},**{k:v for k,v in y2.items() if k!="_q"}}
spec={ # viewed/decided manually
"oKSy7NlOv-0":"reject: viewed, king cobra barely visible","XgMJq6u99NA":"reject: educational clip, not viewed","p7T_T2xJG0E":"reject: viewed, mating snakes look like rat snakes not cobras (ID doubt)","-A3NbpidsnI":"reject: two species (krait and Russell's viper)",
"l-ex95PtJz4":"reject: viewed, ID doubtful from footage","eTkt5rTVxYc":"reject: viewed, mostly person with gloves, ID doubtful","GufTUWnClmE":"reject: viewed, multi-snake compilation",
"OuxqGvMHe3o":"reject: viewed, TV news footage of people holding a python","KwlmfX9jpc4":"reject: viewed, snake not visible","XxpTjzNs5hY":"reject: viewed, person talking, snake barely shown",
"_2kI_uS_Ha0":"reject: viewed, mostly news anchor","hCCW35uYLFw":"reject: viewed, news banner and people, snake only briefly in bucket","6hsHXITNz2Q":"reject: viewed, borderline: snake partly visible, text overlay and people",
"euYLFRk93wk":"reject: viewed, tiny snake at night with watermark and black frame (borderline)","ZHqfMbABXes":"reject: viewed, TV news layout, mostly banners",
"Lv-GonLLFqI":"reject: viewed, multi-species compilation","cyT5mxzsTWA":"reject: download failed (format unavailable)","kGpbhCCjNVo":"reject: Trimeresurus stejnegeri not in class list",
"Rnd_dQgy1Us":"reject: Hydrophis belcheri not in class list","Vwq_sUVpvso":"reject: viewed, good clip but near-duplicate of accepted fowlea clip (same uploader, same species)",
"uPvatDbKSG8":"reject: not selected (Gonyosoma already in old test set)","7ZWW49ZT5uE":"reject: same source as used banded krait milking","PKr3BCLhZwo":"reject: same source session as used banded krait milking",
"F2SU8ixnBss":"reject: same video as already-used Commons Laticauda clip",
"jfDvHnWcTJ0":"reject: monitor lizard, not selected (other negatives chosen)","tMdVIWQ8_Zw":"reject: not selected as negative","jXhezIDfO-E":"reject: not selected as negative (downloaded, not used)","ocxhQwJAEpk":"reject: not selected as negative (downloaded, not used)","X4No6dg7N5E":"reject: not viewed",
}
for r in acc.values():
    pass
accids={}
for c,r in acc.items():
    if "youtube.com" in r["source_url"]: accids[r["source_url"].split("v=")[1]]="ACCEPT (snake) as "+c
for c,r in neg.items():
    if "youtube.com" in r["source_url"]: accids[r["source_url"].split("v=")[1]]="ACCEPT (negative) as "+c
yrows=[]
for k,x in yt.items():
    if x.get("skip"): why="reject: "+x["skip"]+" filter (duration <5 s or >25 min) or metadata failure"; t=x.get("title") or ""
    else:
        t=x["title"]
        if k in accids: why=accids[k]
        elif k in spec: why=spec[k]
        elif not (x["license"] or "").startswith("Creative"): why="reject: page licence is not CC BY"
        else: why="reject: not selected: title/description did not state an in-list species clearly, or compilation/news/educational/non-snake (metadata only, not viewed)"
    yrows.append(["YouTube (CC filter)",t[:150],"https://www.youtube.com/watch?v="+k,why+" | licence="+str(x.get("license"))])
rows+=yrows
with open(G+"search/candidates.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["source","title","url","decision_and_reason"]); w.writerows(rows)
print(nc,ni,len(yrows),len(rows))
