# import ujson
# import os
# import numpy as np
#
# # max_new_token = 16
# max_new_token = 8
# docid_to_smtid_path = "./experiments-full-t5seq-aq/t5_docid_gen_encoder_new2/aq_smtid/docid_to_smtid.json"
# teacher_score_path = "./experiments-full-t5seq-aq/t5_docid_gen_encoder_new2/out/pubmed/qid_smtid_docids_teacher_score.train.json"
# out_dir = f"./experiments-full-t5seq-aq/t5_docid_gen_encoder_new2/sub_smtid_{max_new_token}_out/"
#
# if not os.path.exists(out_dir):
#     os.makedirs(out_dir, exist_ok=True)
#
# with open(docid_to_smtid_path) as fin:
#     docid_to_smtids = ujson.load(fin)
#
# docid_to_smtid = {}
# for docid, smtids in docid_to_smtids.items():
#     assert len(smtids) in [5, 9, 17, 33], len(smtids)
#     smtid = "_".join([str(x) for x in smtids[1:1 + max_new_token]])
#     docid_to_smtid[docid] = smtid
# print("size of docid_to_smtid = {}".format(len(docid_to_smtid)))
#
# qid_to_smtid_to_docids = {}
# with open(teacher_score_path) as fin:
#     for line in fin:
#         example = ujson.loads(line)
#         qid = example["qid"]
#         docids = example["docids"]
#
#         if qid not in qid_to_smtid_to_docids:
#             qid_to_smtid_to_docids[qid] = {}
#
#         for docid in docids:
#             smtid = docid_to_smtid[docid]
#             if smtid not in qid_to_smtid_to_docids[qid]:
#                 qid_to_smtid_to_docids[qid][smtid] = [docid]
#             else:
#                 qid_to_smtid_to_docids[qid][smtid] += [docid]
#
# docid_lengths = []
# for qid in qid_to_smtid_to_docids:
#     for smtid in qid_to_smtid_to_docids[qid]:
#         docid_lengths.append(len(qid_to_smtid_to_docids[qid][smtid]))
# print("distribution of docid_lengths: ", np.quantile(docid_lengths, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]))
#
# if not os.path.exists(out_dir):
#     os.mkdir(out_dir)
#
# with open(os.path.join(out_dir, "qid_smtid_docids.train.json"), "w") as fout:
#     ujson.dump(qid_to_smtid_to_docids, fout)

import ujson
import os
import numpy as np

# 仅处理 max_new_token = 8 的情况
max_new_token = 8
docid_to_smtid_path = "./data3/experiments-full-t5seq-aq/t5_docid_gen_encoder_0_1/aq_smtid/docid_to_smtid.json"
teacher_score_path = "./data3/experiments-full-t5seq-aq/t5_docid_gen_encoder_0_1/out/pubmed/bm25_tran_socre_for_title_cleaned.json"
out_dir = f"./data3/experiments-full-t5seq-aq/t5_docid_gen_encoder_0_1/sub_smtid_{max_new_token}_out/"

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

with open(docid_to_smtid_path) as fin:
    docid_to_smtids = ujson.load(fin)

docid_to_smtid = {}
for docid, smtids in docid_to_smtids.items():
    assert len(smtids) in [5, 9, 17, 33], len(smtids)
    smtid = "_".join([str(x) for x in smtids[1:1 + max_new_token]])
    docid_to_smtid[docid] = smtid
print("size of docid_to_smtid = {}".format(len(docid_to_smtid)))

qid_to_smtid_to_docids = {}
with open(teacher_score_path) as fin:
    for line in fin:
        example = ujson.loads(line)
        qid = example["qid"]
        docids = example["docids"]

        if qid not in qid_to_smtid_to_docids:
            qid_to_smtid_to_docids[qid] = {}

        for docid in docids:
            smtid = docid_to_smtid.get(docid)
            if smtid:
                if smtid not in qid_to_smtid_to_docids[qid]:
                    qid_to_smtid_to_docids[qid][smtid] = [docid]
                else:
                    qid_to_smtid_to_docids[qid][smtid].append(docid)

docid_lengths = []
for qid in qid_to_smtid_to_docids:
    for smtid in qid_to_smtid_to_docids[qid]:
        docid_lengths.append(len(qid_to_smtid_to_docids[qid][smtid]))
print("distribution of docid_lengths: ", np.quantile(docid_lengths, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]))

with open(os.path.join(out_dir, "qid_smtid_docids.train.json"), "w") as fout:
    ujson.dump(qid_to_smtid_to_docids, fout)
