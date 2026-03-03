import json

# 读取 JSONL
jsonl_file = "/mnt/jinbo/RLRM/previous_work/MemCoT/RRcot-copy/data/train/train.jsonl"
json_file = "/mnt/zhaorunsong/lx/RRcot/data/eval/distill.json"

data_list = []

with open(jsonl_file, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        # 构造新结构
        new_item = {
            "meta_data": item.get("question_list", []),
            "question": " ".join(item.get("question_list", [])),
            "answer": item.get("gt_output", "").split("\\boxed{")[-1].split("}")[0] if "\\boxed{" in item.get("gt_output", "") else "",
            "choices_list": [c.split(" ")[-1] for c in item.get("question_list", []) if c.startswith("$\\text{(")],
            "domain": "distill",
            "question_list": item.get("question_list", [])
        }
        data_list.append(new_item)

# 保存成 JSON
output = {"distill": data_list}
with open(json_file, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=4, ensure_ascii=False)
