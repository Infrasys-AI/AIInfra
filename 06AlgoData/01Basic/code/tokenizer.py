from transformers import AutoTokenizer

# 加载 Qwen 的 Tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B", trust_remote_code=True)

# 测试文本
text = "这是一个测试，看看分词器是否能正确处理中文。"

# 编码：将文本转换为 token IDs
encoded_ids = tokenizer.encode(text)
print("编码结果 (Token IDs):", encoded_ids)

# 解码：将 token IDs 转换回文本
decoded_text = tokenizer.decode(encoded_ids)
print("解码结果 (文本):", decoded_text)

decoded_text = tokenizer.decode([434])
print("解码结果 (文本):", decoded_text)

decoded_text = tokenizer.decode([1589])
print("解码结果 (文本):", decoded_text)

# 如果需要查看具体的 tokens（子词单元）
tokens = tokenizer.tokenize(text)
print("分词结果 (Tokens):", tokens)

# 编码（Tokenize）
text = "Hello, world! Transformers is great for NLP."
input_ids = tokenizer.encode(text, return_tensors="pt")
print(input_ids)

# 解码（Decode）
decoded_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
print(decoded_text)

# 测试 vocab-token 数量占比
vocab = tokenizer.get_vocab()

total_tokens = len(vocab)
chinese_tokens = 0
english_tokens = 0
non_chinese_or_english_tokens = 0

def is_chinese(text):
    return all('\u4e00' <= char <= '\u9fff' for char in text)

def is_english(text):
    return all(re.match(r"[a-zA-Z]", char) for char in text)

for token_id in range(total_tokens):
    token = tokenizer.decode([token_id])
    if is_chinese(token):
        chinese_tokens += 1
    elif is_english(token):
        english_tokens += 1
    else:
        non_chinese_or_english_tokens += 1

chinese_ratio = chinese_tokens / total_tokens * 100
english_ratio = english_tokens / total_tokens * 100
other_ratio = non_chinese_or_english_tokens / total_tokens * 100
print(f"词表总大小: {total_tokens}")
print(f"中文 token 数量: {chinese_tokens} 占比: {chinese_ratio:.2f}%")
print(f"英文 token 数量: {english_tokens} 占比: {english_ratio:.2f}%")
print(f"其他 token 数量: {non_chinese_or_english_tokens} 占比: {other_ratio:.2f}%")