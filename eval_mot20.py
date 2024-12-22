import os
from eval.eval import eval


dataset_path = "data/MOT20/train"
out_path = "output/mot20"
exp_name = "val"

seqmap = os.path.join(out_path,exp_name, "val_seqmap.txt")

# 生成val_seqmap.txt文件
with open(seqmap,"w") as f:
    f.write("name\n")
    f.write("MOT20-01\n")
    f.write("MOT20-02\n")
    f.write("MOT20-03\n")
    f.write("MOT20-05\n")

HOTA,IDF1,MOTA,AssA = eval(dataset_path,out_path, seqmap, exp_name,1,False)