from transformers import AutoTokenizer

# ===== 改成你自己的本地路径 =====
qwen_path = "/mnt/zhaorunsong/models/Qwen2.5-0.5B-Instruct" 
llama_path = "/mnt/zhaorunsong/models/meta-llama/Llama-3.2-1B-Instruct" 

# 加载 tokenizer
qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_path, trust_remote_code=True)
llama_tokenizer = AutoTokenizer.from_pretrained(llama_path)

# 测试文本（中英混合更容易看差异）
text = "Hello world! 你好，世界！"

print("=" * 60)
print("原始文本：", text)
print("=" * 60)

# ===== Qwen =====
print("\n🔵 Qwen tokenizer")
ids_qwen_no_special = qwen_tokenizer(text, add_special_tokens=False)["input_ids"]
ids_qwen_with_special = qwen_tokenizer(text, add_special_tokens=True)["input_ids"]

print("add_special_tokens=False 长度:", len(ids_qwen_no_special))
print(ids_qwen_no_special)
print("解码:", qwen_tokenizer.decode(ids_qwen_no_special))

print("\nadd_special_tokens=True 长度:", len(ids_qwen_with_special))
print(ids_qwen_with_special)
print("解码:", qwen_tokenizer.decode(ids_qwen_with_special))


# ===== Llama =====
print("\n🦙 Llama tokenizer")
ids_llama_no_special = llama_tokenizer(text)["input_ids"]
ids_llama_with_special = llama_tokenizer(text, add_special_tokens=True)["input_ids"]

print("add_special_tokens=False 长度:", len(ids_llama_no_special))
print(ids_llama_no_special)
print("解码:", llama_tokenizer.decode(ids_llama_no_special))

print("\nadd_special_tokens=True 长度:", len(ids_llama_with_special))
print(ids_llama_with_special)
print("解码:", llama_tokenizer.decode(ids_llama_with_special))


# ===== 差异对比 =====
print("\n" + "=" * 60)
print("长度对比：")
print("Qwen (no special): ", len(ids_qwen_no_special))
print("Llama(no special): ", len(ids_llama_no_special))
print("Qwen (with special): ", len(ids_qwen_with_special))
print("Llama(with special): ", len(ids_llama_with_special))
print("=" * 60)