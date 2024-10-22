import ujson 
import os 

# 处理 max_new_token = 8 的情况
max_new_token = 8
mnt_to_smtid_to_factor = {
    8: {"smtid_4": 0.5/0.75}
}

print("max_new_token: ", max_new_token)
print("factor mapping: ")
smitd_to_factor =  mnt_to_smtid_to_factor[max_new_token]
for smtid, factor in smitd_to_factor.items():
    print("smtid: ", smtid, "factor: ", factor)

root_dir = "./experiments-full-t5seq-aq/t5seq_aq_encoder_seq2seq_1/"
source_example_path = os.path.join(root_dir, f"sub_smtid_train_decay2/qid_smtids_scores_{max_new_token}.train.json")
out_dir = os.path.join(root_dir, "lng_knp_sub_smtid_train_decay2")

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

target_example_path = os.path.join(out_dir, f"lng_knp_qid_smtids_scores_{max_new_token}.train.json")
with open(source_example_path) as fin:
    with open(target_example_path, "w") as fout:
        for line in fin:
            example = ujson.loads(line)

            for smtid, factor in smitd_to_factor.items():
                example[f"{smtid}_scores"] = [x*factor for x in example["scores"]]

            fout.write(ujson.dumps(example) + "\n")
