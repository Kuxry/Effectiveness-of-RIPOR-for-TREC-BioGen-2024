import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# 假设模型路径
model_dir = "./experiments-full-t5seq-aq/t5seq_aq_encoder_seq2seq_new1/checkpoint"

# 加载模型和分词器
tokenizer = T5Tokenizer.from_pretrained(model_dir)
model = T5ForConditionalGeneration.from_pretrained(model_dir)

# 测试的查询
query = "Double balloon enteroscopy: is it efficacious and safe in a community setting?？"
query_id = "23831910"

# 模拟的文档内容（假设已经是编码后的形式）
document_ids = ["1523516", "1523517", "1523518"]
document_contents = [
    "Twenty-two patients with neurologic deficit due to delayed posttraumatic vertebral collapse after osteoporotic compression fractures of the thoracolumbar spine underwent anterior decompression and reconstruction with bioactive Apatite-Wollastonite containing glass ceramic vertebral prosthesis and Kaneda instrumentation. Eighteen patients previously had minor trauma that resulted in a mild vertebral compression fracture without any neurologic involvement and were either conservatively treated or not treated at all. Four had no history of back injury. The preoperative neurologic status was incomplete paralysis in all patients....",
    "Segmental pedicle screw instrumentation in adult lumbar scoliosis allows better curve correction and restoration of lumbar lordosis. In a retrospective study, to assess the value of this fixation, 9 patients treated with the AO Internal Fixator and 18 with Cotrel-Dubousset instrumentation were reviewed. Mean age at surgery was 60 years (range, 40-88), and curves were measured between 22 degrees and 82 degrees. ...",
    "Twenty-four patients undergoing anterior and posterior spinal fusion were preoperatively assessed and at weekly intervals during their hospitalization for the following: serum albumin, transferrin, total lymphocyte count, skin anergy and anthropometric measurements. Eleven patients underwent staged anterior and posterior spinal reconstruction and 13 patients underwent similar procedures in a combined fashion. The groups were similar in age, sex, diagnosis, and number of levels fused. Results of these tests were kept as well as operative time, blood loss, transfusion requirements, wound healing problems, infection, length of hospital stay, patient satisfaction, and total cost. All 24 patients had normal preoperative nutritional parameters. Seven patients in the staged group and ten in the combined group became malnourished after surgery."
]

# 将文档内容编码为token id序列
encoded_documents = [tokenizer.encode(doc, return_tensors="pt") for doc in document_contents]

# 查询和文档的输入拼接
# 假设模型要求将查询和文档一起输入，形成类似于 `query: <query_text> document: <document_text>` 的格式
inputs = []
for doc_id, encoded_doc in zip(document_ids, encoded_documents):
    input_text = f"query: {query} document: {doc_id}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    inputs.append(input_ids)

# 转换为模型输入的形式
inputs = torch.cat(inputs, dim=0)  # 组合成批量输入

# 使用模型进行推理
with torch.no_grad():
    outputs = model.generate(inputs, max_length=50, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)


# 解码模型输出
decoded_outputs = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

# 输出结果
for doc_id, output in zip(document_ids, decoded_outputs):
    print(f"Document ID: {doc_id}, Predicted Output: {output}")
