from transformers import pipeline

# 使用情绪分析流水线
classifier = pipeline('sentiment-analysis')
res = classifier('')

print(res)
