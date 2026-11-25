import csv, json, re
G="results/m5/g5/"
cut=json.load(open(G+"search/cut_result.json"))
m1=json.load(open(G+"search/meta1.json")); m2=json.load(open(G+"search/meta2.json"))
y1=json.load(open(G+"search/yt_cands.json")); y2=json.load(open(G+"search/yt_cands2.json"))
yt={**{k:v for k,v in y1.items() if k!="_q"},**{k:v for k,v in y2.items() if k!="_q"}}
CCBY="CC BY (YouTube Creative Commons Attribution licence)"
def cm(title): 
    v=m1.get(title) or m2.get(title); return v
HK="File:King Cobra (Ophiophagus hannah) hunting in the afternoon in Singapore's Sungei Buloh Wetland Reserve (May 27, 2023).ogg"
def cu(t): return "https://commons.wikimedia.org/wiki/"+t.replace(" ","_")
def yu(i): return "https://www.youtube.com/watch?v="+i
S=[ # clip, species, kind, key, species_source, note
("ophiophagus_hannah_singapore_hunting","Ophiophagus hannah","c",HK,"stated","Wild king cobra moving on forest floor, snake small in frame but visible from ~5 s; same session/uploader as the 'crossing a path' clip which was not used. Start cut at 12 s of original."),
("ptyas_mucosa_india_drinking","Ptyas mucosa","c","File:Rat snake drinking water.webm","inferred","Page says only 'Ratsnake drinking water' (Colubrinae); species assumed Ptyas mucosa from appearance and Indian context. Doubtful ID."),
("bungarus_fasciatus_taiwan_night","Bungarus fasciatus","y","lV1z-1y2e6s","stated","Night footage with torch, snake visible throughout."),
("bungarus_fasciatus_kerala_pond","Bungarus fasciatus","y","QoGcWIgpsww","stated","Title/tags say banded krait (common name); Malayalam channel watermark; stick/pipe handling at the end."),
("ahaetulla_prasina_handled","Ahaetulla prasina","y","IUanO6AcvS8","stated","Snake handled by a person (hand and face visible part of the time)."),
("elaphe_taeniura_taiwan_tree","Elaphe taeniura","y","_qpCyYlcs00","stated","Subspecies friesi (Taiwan); dark forest footage, snake partly occluded by leaves."),
("chrysopelea_ornata_thailand_gecko","Chrysopelea ornata","y","JHdFB4kzZB0","stated","Vertical video with black side bars; title card removed by starting at 8 s; snake coiled on tree trunk."),
("cerberus_rynchops_eating_fish","Cerberus rynchops","y","iuuZxF5pBpg","stated","Dog-faced water snake eating fish, shallow water at night; hand-held."),
("oreocryptophis_porphyraceus_box","Oreocryptophis porphyraceus","y","rHaBYRTxEoo","stated","Captive snake in plastic box (subspecies coxi)."),
("indotyphlops_braminus_home","Indotyphlops braminus","y","bZ6ODBE53DI","stated","Tiny worm-like snake on a floor, text overlays (common name and size) present."),
("indotyphlops_braminus_almeria","Indotyphlops braminus","y","sagWY6JJ3R0","stated","Binomial and locality are burned into the frame as a caption (species name visible in pixels)."),
("laticauda_colubrina_bali","Laticauda colubrina","y","qHJYZ3MakxQ","stated","Dark night dive footage, snake small and thin but visible."),
("daboia_russelii_closeup","Daboia russelii","y","zwxn2JxntUw","stated","Title says Russell's Viper (common name); royalty-free close-up of a coiled snake."),
("craspedocephalus_malabaricus_branch","Craspedocephalus malabaricus","y","nxZI3XC1XIo","stated","Caption in frame reads 'Malabar pit-viper (Trimeresurus malabaricus)' = older name of Craspedocephalus malabaricus."),
("dendrelaphis_tristis_rock","Dendrelaphis tristis","y","SACCN2dinV4","inferred","Title says 'Common Bronzeback Tree Snake' only; India location assumed; thin snake on rock."),
("coelognathus_helena_floor","Coelognathus helena","y","Yx2lzpY9gcY","stated","Binomial burned into first seconds; outro card trimmed (16 s kept); worker tools near snake."),
("python_molurus_galibore","Python molurus","y","8uw4wpoA5J8","inferred","Title says 'Rock Python', Galibore (Karnataka); snake digesting a large meal, hand-held shaky footage."),
("naja_kaouthia_baby_container","Naja kaouthia","y","nVKT4Z3hoFA","stated","Title says Baby Monocled Cobra (common name); juvenile hooded in a clear container."),
("naja_kaouthia_captive_feeding","Naja kaouthia","y","6_lBJvP764w","stated","Captive coiled cobra in dim light, low quality; same uploader as other Naja kaouthia clips but different footage."),
("naja_kaouthia_bangladesh_hood","Naja kaouthia","y","X0nZn8sFPFo","stated","Title says Monocled Cobra (Bengali and English); hooded snake in the open then snake under a rock; channel logo watermark."),
("fowlea_piscator_rock","Fowlea piscator","y","VEEsXjK9Ni8","stated","Title 'Checkered keelback' (common name of Fowlea piscator); basking on rock."),
("lycodon_aulicus_baby","Lycodon aulicus","y","aGA4ANO5d-w","stated","Baby snake, small in frame, with channel watermark."),
]
with open(G+"new_clips.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow("clip,species,source_url,licence,author,species_source,source,original_seconds,cut_start_s,kept_seconds,notes".split(","))
    for name,sp,k,key,ss,note in S:
        r=cut[name]
        if k=="c":
            v=cm(key); w.writerow([name+".mp4",sp,cu(key),v["lic"],v["artist"],ss,"Wikimedia Commons",r["orig"],r["start"] if name!="ophiophagus_hannah_singapore_hunting" else 12,r["kept"] if name!="ophiophagus_hannah_singapore_hunting" else 26.3,note])
        else:
            x=yt[key]; w.writerow([name+".mp4",sp,yu(key),CCBY,x["uploader"],ss,"YouTube (page licence field)",r["orig"],r["start"],r["kept"],note])
N=[
("negative_eels_hawaii","two eels in a tidepool (eel, no snake)","c","File:2 eels - Anguilliformes in a tidepool near Kona, Hawaii get into a fight.ogv","Snake-like body; hard negative."),
("negative_slug_austria","a keelback slug Limax cinereoniger (slug, no snake)","c","File:Limax.webm","Soft-bodied, no snake."),
("negative_glass_lizard","a European legless lizard Pseudopus apodus (no snake)","y","vnTVtuqM8SQ","Legless lizard, looks like a snake; caption in first seconds trimmed; wildlife logo in corner."),
("negative_slowworm","a slowworm Anguis (legless lizard, no snake)","y","EEBWWyjrZr4","Legless lizard on leaf litter; hard negative."),
("negative_bengal_monitor","a Bengal monitor lizard walking (no snake)","y","5yoXZSawtXo","Large lizard, Yala Sri Lanka."),
]
with open(G+"new_negatives.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow("clip,content,source_url,licence,author,source,cut_start_s,kept_seconds,notes".split(","))
    for name,content,k,key,note in N:
        r=cut[name]
        if k=="c":
            v=cm(key); w.writerow([name+".mp4",content,cu(key),v["lic"],v["artist"],"Wikimedia Commons",r["start"],r["kept"],note])
        else:
            x=yt[key]; w.writerow([name+".mp4",content,yu(key),CCBY,x["uploader"],"YouTube (page licence field)",r["start"],r["kept"],note])
print("ok")
