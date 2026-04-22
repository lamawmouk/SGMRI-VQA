  
  #!/usr/bin/env python3
"""
create_full_json_dataset_knee.py

Generates a JSON dataset for KNEE MRI volumes with VARIABLE slice counts.

For each volume:
  • Includes ALL slices found on disk
  • Slice numbering is taken directly from filename (1-based, possibly sparse)
  • Attach slice-level labels (from CSV)
  • Attach bounding boxes (from CSV)
  • final_diagnosis = union of all labels for that volume

Output structure:

{
  "volume_id": {
     "volume_id": "...",
     "final_diagnosis": [...],
     "slices": [
        {
           "slice": <actual slice index>,
           "image_path": "...",
           "label": [...],
           "bounding_boxes": [...]
        }
     ]
  }
}
"""

import os
import csv
import json
from collections import defaultdict

# --------------------------------------------------------
# CONFIG
# --------------------------------------------------------
BASE = "/storage/ice-shared/ae8803che/lmkh3/"
CSV_PATH = "/home/hice1/lmoukheiber3/SDR/fastmri-plus/Annotations/knee.csv"
OUTPUT_JSON = "/home/hice1/lmoukheiber3/SDR/fastmri-plus/Annotations/knee_val_medtr_volumes.json"

SPLIT = "val"                 # train / val / test
RAW_SUFFIX = "_labeled_raw"
# --------------------------------------------------------


# --------------------------------------------------------
# Detect filename column in CSV
# --------------------------------------------------------
with open(CSV_PATH, newline="") as f:
    reader = csv.DictReader(f)
    headers = reader.fieldnames

possible_cols = ["file", "file_bfile", "fname", "filename"]
volume_col = next((c for c in possible_cols if c in headers), None)

if volume_col is None:
    raise ValueError(f"Could not find filename column in CSV. Columns = {headers}")

print(f"Detected filename column: {volume_col}")


# --------------------------------------------------------
# LOAD CSV → slice labels + bounding boxes
# --------------------------------------------------------
annotations = defaultdict(lambda: defaultdict(list))   # volume → slice → boxes
slice_labels = defaultdict(lambda: defaultdict(set))   # volume → slice → labels
volume_labels = defaultdict(set)                      # volume → union labels

with open(CSV_PATH, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        vol = row.get(volume_col)
        if not vol or not row.get("slice"):
            continue

        slice_idx = int(row["slice"])   # matches filename index
        label = row.get("label", "").strip()

        if label:
            slice_labels[vol][slice_idx].add(label)
            volume_labels[vol].add(label)

        if all(row.get(k) for k in ["x", "y", "width", "height"]):
            try:
                annotations[vol][slice_idx].append({
                    "x": int(row["x"]),
                    "y": int(row["y"]),
                    "width": int(row["width"]),
                    "height": int(row["height"]),
                    "label": label
                })
            except ValueError:
                pass


# --------------------------------------------------------
# Build JSON entry for one volume
# --------------------------------------------------------
def build_volume_entry(volume_id, volume_dir):
    slice_entries = []

    pngs = []
    for f in os.listdir(volume_dir):
        if f.endswith(".png") and "_slice_" in f:
            try:
                slice_idx = int(f.split("_slice_")[1].split(".")[0])
                pngs.append((slice_idx, f))
            except ValueError:
                continue

    pngs.sort(key=lambda x: x[0])

    for slice_idx, png in pngs:
        image_path = os.path.join(volume_dir, png)

        slice_entries.append({
            "slice": slice_idx,
            "image_path": image_path,
            "label": sorted(slice_labels[volume_id].get(slice_idx, [])),
            "bounding_boxes": annotations[volume_id].get(slice_idx, [])
        })

    return {
        "volume_id": volume_id,
        "final_diagnosis": sorted(volume_labels.get(volume_id, [])),
        "slices": slice_entries
    }


# --------------------------------------------------------
# BUILD DATASET JSON
# --------------------------------------------------------
giant_json = {}

raw_root = os.path.join(BASE, f"knee_{SPLIT}{RAW_SUFFIX}")
if not os.path.isdir(raw_root):
    raise RuntimeError(f"Missing directory: {raw_root}")

print(f"\n=== Processing KNEE {SPLIT.upper()} volumes ===")

for volume_id in sorted(os.listdir(raw_root)):
    volume_dir = os.path.join(raw_root, volume_id)
    if not os.path.isdir(volume_dir):
        continue

    print(f"  Adding volume: {volume_id}")
    giant_json[volume_id] = build_volume_entry(volume_id, volume_dir)


# --------------------------------------------------------
# SAVE JSON
# --------------------------------------------------------
with open(OUTPUT_JSON, "w") as jf:
    json.dump(giant_json, jf, indent=2)

print(f"\nDONE — saved JSON dataset to:\n{OUTPUT_JSON}")

# --------------------------------------------------------
# PRINT SLICE COUNT STATISTICS
# --------------------------------------------------------
# PRINT SLICE COUNT STATISTICS
slice_count_stats = defaultdict(list)

for volume_id, volume_data in giant_json.items():
    slice_count = len(volume_data.get("slices", []))
    slice_count_stats[slice_count].append(volume_id)

# Print the statistics
for count, volumes in sorted(slice_count_stats.items()):
    print(f"  {count} slices:")
    for vol in volumes:
        print(f"   {vol}/")
        
""" 
    28 slices:
   file1001465/
   file1002078/
   file1002376/
   file1002479/

  29 slices:
   file1000568/
   file1000685/
   file1001627/

  30 slices:
   file1000053/
   file1000129/
   file1000167/
   file1000185/
   file1000210/
   file1000237/
   file1000250/
   file1000257/
   file1000298/
   file1000311/
   file1000418/
   file1000566/
   file1000587/
   file1000601/
   file1000602/
   file1000658/
   file1000661/
   file1000715/
   file1000873/
   file1000933/
   file1000973/
   file1000988/
   file1000994/
   file1001036/
   file1001099/
   file1001131/
   file1001173/
   file1001256/
   file1001348/
   file1001403/
   file1001495/
   file1001503/
   file1001544/
   file1001550/
   file1001698/
   file1001811/
   file1001854/
   file1001858/
   file1001909/
   file1001919/
   file1001944/
   file1001960/
   file1001963/
   file1001965/
   file1001975/
   file1001979/
   file1002022/
   file1002042/
   file1002055/
   file1002112/
   file1002239/
   file1002241/
   file1002244/
   file1002246/
   file1002256/
   file1002335/
   file1002469/
   file1002491/
   file1002497/
   file1002521/
   file1002534/
   file1002549/

  31 slices:
   file1000483/
   file1001699/
   file1001969/

  32 slices:
   file1000012/
   file1000032/
   file1000075/
   file1000081/
   file1000259/
   file1000275/
   file1000324/
   file1000347/
   file1000420/
   file1000459/
   file1000489/
   file1000493/
   file1000590/
   file1000653/
   file1000721/
   file1000723/
   file1000814/
   file1000827/
   file1000898/
   file1000919/
   file1000941/
   file1000963/
   file1000964/
   file1000971/
   file1000995/
   file1001071/
   file1001101/
   file1001214/
   file1001216/
   file1001287/
   file1001296/
   file1001320/
   file1001323/
   file1001436/
   file1001475/
   file1001519/
   file1001545/
   file1001690/
   file1001783/
   file1001870/
   file1001948/
   file1001966/
   file1002266/
   file1002300/
   file1002353/
   file1002458/
   file1002474/
   file1002563/

  33 slices:
   file1000029/
   file1000039/
   file1000060/
   file1000069/
   file1000094/
   file1000098/
   file1000109/
   file1000123/
   file1000138/
   file1000142/
   file1000143/
   file1000161/
   file1000342/
   file1000369/
   file1000382/
   file1000384/
   file1000443/
   file1000461/
   file1000485/
   file1000490/
   file1000509/
   file1000510/
   file1000583/
   file1000586/
   file1000597/
   file1000617/
   file1000619/
   file1000627/
   file1000640/
   file1000663/
   file1000728/
   file1000753/
   file1000777/
   file1000846/
   file1000848/
   file1000875/
   file1000876/
   file1000884/
   file1000954/
   file1000969/
   file1000983/
   file1001007/
   file1001008/
   file1001013/
   file1001027/
   file1001079/
   file1001087/
   file1001092/
   file1001121/
   file1001169/
   file1001193/
   file1001203/
   file1001206/
   file1001209/
   file1001300/
   file1001309/
   file1001310/
   file1001342/
   file1001373/
   file1001378/
   file1001401/
   file1001421/
   file1001445/
   file1001452/
   file1001468/
   file1001496/
   file1001521/
   file1001523/
   file1001580/
   file1001584/
   file1001587/
   file1001619/
   file1001632/
   file1001654/
   file1001677/
   file1001733/
   file1001775/
   file1001800/
   file1001836/
   file1001857/
   file1001864/
   file1001867/
   file1001875/
   file1001906/
   file1001953/
   file1001982/
   file1002086/
   file1002115/
   file1002153/
   file1002169/
   file1002175/
   file1002203/
   file1002231/
   file1002245/
   file1002298/
   file1002307/
   file1002315/
   file1002354/
   file1002369/
   file1002381/
   file1002395/
   file1002494/
   file1002543/

  34 slices:
   file1000023/
   file1000027/
   file1000154/
   file1000205/
   file1000222/
   file1000236/
   file1000248/
   file1000265/
   file1000293/
   file1000331/
   file1000378/
   file1000426/
   file1000427/
   file1000452/
   file1000458/
   file1000495/
   file1000539/
   file1000639/
   file1000734/
   file1000795/
   file1000808/
   file1000834/
   file1000896/
   file1000959/
   file1000968/
   file1001001/
   file1001023/
   file1001051/
   file1001094/
   file1001222/
   file1001236/
   file1001306/
   file1001334/
   file1001350/
   file1001369/
   file1001392/
   file1001393/
   file1001659/
   file1001666/
   file1001713/
   file1001773/
   file1001795/
   file1001801/
   file1001810/
   file1001849/
   file1001855/
   file1001892/
   file1001912/
   file1002010/
   file1002012/
   file1002023/
   file1002095/
   file1002163/
   file1002229/
   file1002238/
   file1002247/
   file1002367/
   file1002386/
   file1002396/
   file1002407/
   file1002422/
   file1002432/
   file1002461/
   file1002495/
   file1002509/
   file1002542/

  35 slices:
   file1000005/
   file1000021/
   file1000061/
   file1000084/
   file1000086/
   file1000088/
   file1000117/
   file1000131/
   file1000172/
   file1000224/
   file1000233/
   file1000242/
   file1000256/
   file1000287/
   file1000290/
   file1000313/
   file1000338/
   file1000376/
   file1000393/
   file1000425/
   file1000428/
   file1000448/
   file1000475/
   file1000486/
   file1000492/
   file1000501/
   file1000529/
   file1000547/
   file1000588/
   file1000592/
   file1000637/
   file1000695/
   file1000709/
   file1000740/
   file1000824/
   file1000829/
   file1000836/
   file1000859/
   file1000863/
   file1000952/
   file1000985/
   file1000997/
   file1001048/
   file1001061/
   file1001098/
   file1001139/
   file1001150/
   file1001154/
   file1001239/
   file1001257/
   file1001294/
   file1001355/
   file1001358/
   file1001426/
   file1001479/
   file1001500/
   file1001520/
   file1001526/
   file1001540/
   file1001571/
   file1001602/
   file1001609/
   file1001622/
   file1001642/
   file1001661/
   file1001695/
   file1001710/
   file1001746/
   file1001837/
   file1001860/
   file1001872/
   file1001878/
   file1001901/
   file1001915/
   file1001920/
   file1001935/
   file1001941/
   file1001991/
   file1002001/
   file1002027/
   file1002049/
   file1002054/
   file1002056/
   file1002074/
   file1002076/
   file1002090/
   file1002099/
   file1002132/
   file1002143/
   file1002186/
   file1002190/
   file1002227/
   file1002309/
   file1002363/
   file1002371/
   file1002397/
   file1002399/
   file1002416/
   file1002431/
   file1002481/
   file1002532/
   file1002544/

  36 slices:
   file1000001/
   file1000010/
   file1000015/
   file1000064/
   file1000065/
   file1000070/
   file1000097/
   file1000159/
   file1000176/
   file1000199/
   file1000221/
   file1000261/
   file1000296/
   file1000315/
   file1000363/
   file1000371/
   file1000372/
   file1000381/
   file1000401/
   file1000402/
   file1000424/
   file1000521/
   file1000535/
   file1000563/
   file1000610/
   file1000638/
   file1000708/
   file1000742/
   file1000743/
   file1000801/
   file1000856/
   file1000880/
   file1000936/
   file1000943/
   file1000967/
   file1000970/
   file1001003/
   file1001006/
   file1001026/
   file1001120/
   file1001134/
   file1001149/
   file1001175/
   file1001176/
   file1001186/
   file1001194/
   file1001235/
   file1001286/
   file1001379/
   file1001383/
   file1001390/
   file1001395/
   file1001415/
   file1001425/
   file1001432/
   file1001449/
   file1001494/
   file1001541/
   file1001546/
   file1001582/
   file1001599/
   file1001600/
   file1001636/
   file1001685/
   file1001688/
   file1001706/
   file1001708/
   file1001711/
   file1001847/
   file1001861/
   file1001902/
   file1001998/
   file1002004/
   file1002030/
   file1002038/
   file1002043/
   file1002045/
   file1002064/
   file1002071/
   file1002073/
   file1002089/
   file1002118/
   file1002123/
   file1002139/
   file1002189/
   file1002225/
   file1002263/
   file1002283/
   file1002324/
   file1002338/
   file1002356/
   file1002359/
   file1002378/
   file1002383/
   file1002423/
   file1002437/
   file1002440/
   file1002441/
   file1002444/
   file1002462/
   file1002471/
   file1002475/
   file1002499/
   file1002505/
   file1002545/
   file1002554/
   file1002569/

  37 slices:
   file1000066/
   file1000127/
   file1000141/
   file1000173/
   file1000179/
   file1000181/
   file1000193/
   file1000200/
   file1000208/
   file1000231/
   file1000244/
   file1000340/
   file1000357/
   file1000499/
   file1000561/
   file1000615/
   file1000632/
   file1000662/
   file1000693/
   file1000750/
   file1000778/
   file1000785/
   file1000821/
   file1000833/
   file1000922/
   file1000927/
   file1000955/
   file1001022/
   file1001029/
   file1001062/
   file1001067/
   file1001109/
   file1001141/
   file1001228/
   file1001245/
   file1001266/
   file1001327/
   file1001372/
   file1001382/
   file1001410/
   file1001478/
   file1001588/
   file1001605/
   file1001628/
   file1001723/
   file1001737/
   file1001745/
   file1001806/
   file1001823/
   file1001910/
   file1001945/
   file1001956/
   file1001973/
   file1002014/
   file1002151/
   file1002194/
   file1002201/
   file1002286/
   file1002303/
   file1002357/
   file1002366/
   file1002368/
   file1002516/
   file1002531/

  38 slices:
   file1000002/
   file1000040/
   file1000048/
   file1000058/
   file1000059/
   file1000204/
   file1000216/
   file1000284/
   file1000330/
   file1000351/
   file1000390/
   file1000467/
   file1000481/
   file1000518/
   file1000559/
   file1000584/
   file1000600/
   file1000626/
   file1000874/
   file1000882/
   file1000931/
   file1000947/
   file1001082/
   file1001108/
   file1001161/
   file1001185/
   file1001200/
   file1001217/
   file1001220/
   file1001246/
   file1001261/
   file1001299/
   file1001304/
   file1001319/
   file1001328/
   file1001333/
   file1001360/
   file1001385/
   file1001402/
   file1001413/
   file1001459/
   file1001504/
   file1001560/
   file1001565/
   file1001615/
   file1001623/
   file1001656/
   file1001675/
   file1001692/
   file1001701/
   file1001721/
   file1001741/
   file1001758/
   file1001777/
   file1001863/
   file1001873/
   file1001888/
   file1001967/
   file1001992/
   file1002044/
   file1002048/
   file1002117/
   file1002154/
   file1002161/
   file1002228/
   file1002251/
   file1002269/
   file1002273/
   file1002293/
   file1002296/
   file1002342/
   file1002352/
   file1002358/
   file1002388/
   file1002449/
   file1002456/
   file1002478/
   file1002492/
   file1002523/
   file1002541/
   file1002566/

  39 slices:
   file1000090/
   file1000120/
   file1000148/
   file1000177/
   file1000306/
   file1000307/
   file1000326/
   file1000383/
   file1000412/
   file1000479/
   file1000550/
   file1000633/
   file1000738/
   file1000770/
   file1000790/
   file1000806/
   file1000851/
   file1000883/
   file1000929/
   file1001014/
   file1001116/
   file1001130/
   file1001196/
   file1001277/
   file1001629/
   file1001645/
   file1001770/
   file1001949/
   file1001950/
   file1001986/
   file1001989/
   file1002006/
   file1002034/
   file1002041/
   file1002066/
   file1002113/
   file1002130/
   file1002243/
   file1002264/
   file1002287/
   file1002326/
   file1002361/
   file1002435/
   file1002439/
   file1002514/

  40 slices:
   file1000101/
   file1000149/
   file1000211/
   file1000220/
   file1000246/
   file1000403/
   file1000416/
   file1000449/
   file1000482/
   file1000508/
   file1000573/
   file1000605/
   file1000691/
   file1000762/
   file1000794/
   file1000807/
   file1000850/
   file1000888/
   file1001005/
   file1001042/
   file1001053/
   file1001056/
   file1001102/
   file1001110/
   file1001145/
   file1001229/
   file1001283/
   file1001388/
   file1001474/
   file1001498/
   file1001575/
   file1001613/
   file1001630/
   file1001673/
   file1001728/
   file1001748/
   file1001814/
   file1001830/
   file1001856/
   file1001931/
   file1001942/
   file1001985/
   file1001993/
   file1002037/
   file1002087/
   file1002094/
   file1002098/
   file1002146/
   file1002334/
   file1002385/
   file1002409/
   file1002473/
   file1002518/
   file1002520/
   file1002568/

  41 slices:
   file1000043/
   file1000213/
   file1000339/
   file1000554/
   file1000736/
   file1000784/
   file1000830/
   file1001595/
   file1002096/
   file1002455/

  42 slices:
   file1000057/
   file1000100/
   file1000125/
   file1000198/
   file1000249/
   file1000262/
   file1000434/
   file1000558/
   file1000569/
   file1000582/
   file1000611/
   file1000643/
   file1000819/
   file1001033/
   file1001037/
   file1001106/
   file1001114/
   file1001259/
   file1001491/
   file1001561/
   file1001572/
   file1001616/
   file1001671/
   file1001739/
   file1001750/
   file1002046/
   file1002069/
   file1002093/
   file1002195/
   file1002262/
   file1002325/
   file1002333/
   file1002459/

  43 slices:
   file1000207/
   file1000285/
   file1000760/
   file1001242/
   file1001638/
   file1001672/
   file1001752/

  44 slices:
   file1000115/
   file1000301/
   file1000956/
   file1001205/
   file1001210/
   file1001455/
   file1001999/
   file1002097/
   file1002193/
   file1002232/

  45 slices:
   file1000003/
   file1000085/
   file1000252/
   file1000612/
   file1001155/
   file1001422/
   file1001883/
   file1001922/
   file1002024/
   file1002332/
   file1002379/
   file1002567/

  46 slices:
   file1002000/

  48 slices:
   file1000271/
   file1001273/

  50 slices:
   file1000962/
   

val:  30 slices:
   file1000026/
   file1000107/
   file1000344/
   file1000555/
   file1000758/
   file1000885/
   file1000925/
   file1001059/
   file1001159/
   file1001480/
   file1001557/
   file1001598/
   file1002526/
  31 slices:
   file1000073/
   file1001499/
  32 slices:
   file1000178/
   file1000254/
   file1000314/
   file1000748/
   file1001148/
   file1001825/
   file1001977/
   file1001984/
  33 slices:
   file1000243/
   file1000280/
   file1000283/
   file1000291/
   file1000325/
   file1000477/
   file1000858/
   file1000899/
   file1001143/
   file1001440/
   file1001585/
   file1001715/
   file1001726/
   file1002159/
   file1002252/
  34 slices:
   file1000017/
   file1000292/
   file1000323/
   file1000328/
   file1000432/
   file1000871/
   file1001096/
   file1001188/
   file1001450/
   file1001995/
   file1001997/
   file1002412/
   file1002538/
  35 slices:
   file1000000/
   file1000071/
   file1000206/
   file1000229/
   file1000263/
   file1000267/
   file1000273/
   file1000356/
   file1000537/
   file1000942/
   file1001077/
   file1001104/
   file1001144/
   file1001202/
   file1001365/
   file1001506/
   file1001533/
   file1001668/
   file1001689/
   file1001793/
   file1001798/
   file1001916/
   file1001955/
   file1002002/
   file1002155/
  36 slices:
   file1000190/
   file1000196/
   file1000247/
   file1000264/
   file1000308/
   file1000464/
   file1000552/
   file1000593/
   file1000647/
   file1001122/
   file1001331/
   file1001703/
   file1001959/
   file1002214/
   file1002417/
  37 slices:
   file1000926/
   file1001057/
   file1001119/
   file1001184/
   file1001338/
   file1001497/
   file1001655/
   file1001862/
   file1002067/
   file1002187/
   file1002436/
  38 slices:
   file1000007/
   file1000041/
   file1000052/
   file1000182/
   file1000528/
   file1000591/
   file1000625/
   file1000735/
   file1000831/
   file1000842/
   file1001064/
   file1001090/
   file1001170/
   file1001298/
   file1001968/
   file1002007/
   file1002145/
   file1002377/
  39 slices:
   file1000635/
   file1002389/
  40 slices:
   file1000031/
   file1000201/
   file1000476/
   file1000538/
   file1000631/
   file1000660/
   file1000769/
   file1000976/
   file1000990/
   file1001126/
   file1001140/
   file1001289/
   file1001651/
   file1001687/
   file1001759/
   file1001843/
   file1001851/
   file1002340/
   file1002382/
  41 slices:
   file1000759/
   file1001458/
   file1001566/
  42 slices:
   file1000277/
   file1000389/
   file1001344/
   file1001429/
   file1001643/
   file1002274/
   file1002515/
   file1002546/
  45 slices:
   file1001938/
  46 slices:
   file1002351/
   file1002451/
   """
   