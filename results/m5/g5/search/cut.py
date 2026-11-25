import subprocess, json, csv, os
G="results/m5/g5/"
HK="King Cobra (Ophiophagus hannah) hunting in the afternoon in Singapore's Sungei Buloh Wetland Reserve (May 27, 2023).ogg"
# name, source path, start, max_len
snk=[
("ophiophagus_hannah_singapore_hunting","raw/"+HK,5,30),
("ptyas_mucosa_india_drinking","raw/Rat snake drinking water.webm",0,30),
("bungarus_fasciatus_taiwan_night","raw_yt/lV1z-1y2e6s.mp4",0,30),
("bungarus_fasciatus_kerala_pond","raw_yt/QoGcWIgpsww.mp4",0,30),
("ahaetulla_prasina_handled","raw_yt/IUanO6AcvS8.mp4",0,30),
("elaphe_taeniura_taiwan_tree","raw_yt/_qpCyYlcs00.mp4",0,30),
("chrysopelea_ornata_thailand_gecko","raw_yt/JHdFB4kzZB0.mp4",8,30),
("cerberus_rynchops_eating_fish","raw_yt/iuuZxF5pBpg.mp4",0,30),
("oreocryptophis_porphyraceus_box","raw_yt/rHaBYRTxEoo.mp4",0,30),
("indotyphlops_braminus_home","raw_yt/bZ6ODBE53DI.mp4",5,30),
("indotyphlops_braminus_almeria","raw_yt/sagWY6JJ3R0.mp4",0,30),
("laticauda_colubrina_bali","raw_yt/qHJYZ3MakxQ.mp4",0,30),
("daboia_russelii_closeup","raw_yt/zwxn2JxntUw.mp4",0,30),
("craspedocephalus_malabaricus_branch","raw_yt/nxZI3XC1XIo.mp4",0,30),
("dendrelaphis_tristis_rock","raw_yt/SACCN2dinV4.mp4",0,30),
("coelognathus_helena_floor","raw_yt/Yx2lzpY9gcY.mp4",0,16),
("python_molurus_galibore","raw_yt/8uw4wpoA5J8.mp4",0,30),
("naja_kaouthia_baby_container","raw_yt/nVKT4Z3hoFA.mp4",0,30),
("naja_kaouthia_captive_feeding","raw_yt/6_lBJvP764w.mp4",0,30),
("naja_kaouthia_bangladesh_hood","raw_yt/X0nZn8sFPFo.mp4",0,30),
("fowlea_piscator_rock","raw_yt/VEEsXjK9Ni8.mp4",0,30),
("lycodon_aulicus_baby","raw_yt/aGA4ANO5d-w.mp4",0,30),
]
neg=[
("negative_eels_hawaii","raw/2 eels - Anguilliformes in a tidepool near Kona, Hawaii get into a fight.ogv",0,30),
("negative_slug_austria","raw/Limax.webm",0,30),
("negative_glass_lizard","raw_yt/vnTVtuqM8SQ.mp4",6,30),
("negative_slowworm","raw_yt/EEBWWyjrZr4.mp4",0,30),
("negative_bengal_monitor","raw_yt/5yoXZSawtXo.mp4",0,30),
]
res={}
for name,src,ss,t in snk+neg:
    out=f"new_clips/{name}.mp4"
    dur=float(subprocess.check_output(["ffprobe","-v","error","-show_entries","format=duration","-of","csv=p=0",G+src]).decode())
    keep=min(t,dur-ss)
    r=subprocess.run(["ffmpeg","-v","error","-y","-ss",str(ss),"-t",str(keep),"-i",G+src,"-an","-vf","scale='min(960,iw)':-2","-c:v","libx264","-crf","24","-preset","fast","-pix_fmt","yuv420p",G+out],capture_output=True,text=True)
    kd=float(subprocess.check_output(["ffprobe","-v","error","-show_entries","format=duration","-of","csv=p=0",G+out]).decode())
    res[name]={"src":src,"orig":round(dur,1),"start":ss,"kept":round(kd,1)}
    print(name,res[name],r.stderr[:100],flush=True)
json.dump(res,open(G+"search/cut_result.json","w"),indent=1)
