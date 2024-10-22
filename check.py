import ujson

input_file = 'bm25_tran_socre_for_title.json'  # 替换为你的实际文件路径
output_file = 'bm25_tran_socre_for_title_cleaned.json'

with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    for i, line in enumerate(infile, start=1):
        try:
            # 尝试加载每一行 JSON 数据
            data = ujson.loads(line)

            # 检查是否存在 'scores' 字段，且长度至少为 100，同时确保 'scores' 和 'docids' 长度一致
            if 'scores' in data and 'docids' in data:
                if len(data['scores']) >= 100 and len(data['scores']) == len(data['docids']):
                    # 保存合法数据到输出文件
                    ujson.dump(data, outfile)
                    outfile.write('\n')
                else:
                    # 输出具体缺陷信息
                    if len(data['scores']) < 100:
                        print(f"Insufficient scores in line {i}: {len(data['scores'])} scores found")
                    elif len(data['scores']) != len(data['docids']):
                        print(f"Mismatch in lengths for docids and scores in line {i}: {len(data['docids'])} docids, {len(data['scores'])} scores")
            else:
                if 'scores' not in data:
                    print(f"Missing 'scores' in line {i}")
                if 'docids' not in data:
                    print(f"Missing 'docids' in line {i}")

        except ujson.JSONDecodeError as e:
            # 跳过无效的 JSON 行
            print(f"Skipping invalid JSON in line {i}: {e}")

print(f"Cleaned data saved to {output_file}")
