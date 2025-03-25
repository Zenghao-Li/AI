#from transformers import pipeline
#text = """Dear Amazon, last week I ordered an Optimus Prime action figuretext =from your online store in Germany. Unfortunately, when I opened the packageI discovered to my horror that I had been sent an action figure of Megatroninstead! As a lifelong enemy of the Decepticons, I hope you can understand mydilemma. To resolve the issue, I demand an exchange of Megatron for theOptimus Prime figure I ordered, Enclosed are copies of my records concerningthis purchase. I expect to hear from you soon. Sincerely, Bumblebee."""
#1.5.1 文本分类
#classifier = pipeline("text-classification")
import pandas as pd
#outputs_1_5_1 = classifier(text)
#out_1_5_1 = pd.DataFrame(outputs_1_5_1)
#print(out_1_5_1)
#1.5.2 命名实体识别
#ner_tagger = pipeline("ner",aggregation_strategy="simple")
#outputs_1_5_2 = ner_tagger(text)
#out_1_5_2 = pd.DataFrame(outputs_1_5_2)
#print(out_1_5_2)
#1.5.3 问答
#reader = pipeline("question-answering")
#question = "What does the customer want?"
#outputs_1_5_3 = reader(question = question,context = text)
#out_1_5_3 = pd.DataFrame([outputs_1_5_3])
#print(out_1_5_3)
#1.5.4 文本摘要
#summarizer = pipeline("summarization")
#outputs_1_5_4 = summarizer(text,max_length = 60,clean_up_tokenization_spaces=True)
#print(outputs_1_5_4[0]['summary_text'])
#1.5.5 翻译
#translator = pipeline("translation_en_to_zh",model="Helsinki-NLP/opus-mt-en-zh")
#outputs_1_5_5 = translator(text,min_length=100)
#print(outputs_1_5_5[0]['translation_text'])
#1.5.6 文本生成
#generator = pipeline("text-generation")
#response = "Dear Bumblebee, I am sorry to hear that your order was mixed up."
#prompt = text + "\n\nCustomer service response:\n" + response
#outputs_1_5_6 = generator(prompt,max_length = 200)
#print(outputs_1_5_6[0]['generated_text'])
#2.1.1 HuggingFace Dataset库
from datasets import load_dataset
emotions = load_dataset("emotion")
#train_ds = emotions["train"]
#train_ds
#print(train_ds.features)
#print(train_ds[:5])
#2.1.2 从Datasets到DataFrame
#import pandas as pd
emotions.set_format(type = "pandas")
#df = emotions["train"][:]
#print(df.head(5))
#def label_int2str(row):
#    return emotions["train"].features["label"].int2str(row)
#df["label_name"] = df["label"].apply(label_int2str)
#print(df.head(5))
#2.1.3 查看类分布
import matplotlib.pyplot as plt
#df["label_name"].value_counts(ascending = True).plot.barh()
#plt.title("Frequency of Classes")
#plt.show()
#2.1.4 推文长度
#df["Words Per Tweet"] = df["text"].str.split().apply(len)
#df.boxplot("Words Per Tweet",by = "label_name", grid = False, showfliers = False, color = "black")
#plt.suptitle("")
#plt.xlabel("")
#plt.show()
#emotions.reset_format()
#2.2.1 字符词元化
#text = "Tokenizing text is a core task of NLP."
#tokenized_text = list(text)
#print(tokenized_text)
#token2idx = {ch:idx for idx, ch in enumerate(sorted(set(tokenized_text)))}
#print(token2idx)
#input_ids = [token2idx[token] for token in tokenized_text]
#print(input_ids)
#categorical_df = pd.DataFrame(
#    {"Name":["Bumblebee","Optimus Prime","Megatron"],"label ID":[0,1,2]})
#pdout = pd.get_dummies(categorical_df["Name"])
#print(pdout)
#import torch
#import torch.nn.functional as F
#input_ids = torch.tensor(input_ids)
#one_hot_encodings = F.one_hot(input_ids,num_classes=len(token2idx))
#one_hot_encodings.shape
#torch.Size([38,20])
#print(f"Token:{tokenized_text[0]}")
#print(f"Tensor index:{input_ids[0]}")
#print(f"One-hot:{one_hot_encodings[0]}")
#2.2.2 单词词元化
#tokenized_text = text.split()
#print(tokenized_text)
#2.2.3 子词词元化
from transformers import AutoTokenizer
model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
#encoded_text = tokenizer(text)
#print(encoded_text)
#tokens = tokenizer.convert_ids_to_tokens(encoded_text.input_ids)
#print(tokens)
#print(tokenizer.convert_tokens_to_string(tokens))
#2.2.4 整个数据集的词元化
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)
#print(tokenize(emotions["train"][:2]))
emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)
#print(emotions_encoded["train"].column_names)
#2.3.1 使用Transformer作为特征提取器
from transformers import AutoModel
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained(model_ckpt).to(device)
text = "this is a test"
inputs = tokenizer(text,return_tensors="pt")
print(f"Input tensor shape: {inputs['input_ids'].size()}")
inputs = {k:v.to(device) for k,v in inputs.items()}
with torch.no_grad():
    outputs_2_3_1 = model(**inputs)
print(outputs_2_3_1)
def extract_hidden_states(batch):
    #把模型输入在GPU上
    inputs = {k:v.to(device) for k,v in batch.items()
              if k in tokenizer.model_input_names}
    #扩展最后一项隐藏状态
    with torch.no_grad():
        last_hidden_state = model(**inputs).last_hidden_state
    return {"hidden_state":last_hidden_state[:,0].cpu().numpy}
emotions_encoded.set_format("torch",columns=["input_ids","attention_mask","label"])
emotions_hidden = emotions_encoded.map(extract_hidden_states,batched=True)
emotions_hidden["train"].column_names
#创建特征矩阵
import numpy as np
x_train = np.array(emotions_hidden["train"]["hidden_state"])
x_valid = np.array(emotions_hidden["validation"]["hidden_state"])
y_train = np.array(emotions_hidden["train"]["label"])
y_valid = np.array(emotions_hidden["validation"]["label"])
x_train.shape,x_valid.shape
labels = emotions["trains"].features["label"].names
#可视化训练集
from sklearn.linear_model import LogisticRegression
lr_clf = LogisticRegression(max_iter=3000)
lr_clf.fit(x_train,y_train)
lr_clf.score(x_valid,y_valid)
from sklearn.dummy import DummyClassifier
dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(x_train,y_train)
dummy_clf.score(x_valid,y_valid)
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
def plot_confusion_matrix(y_preds,y_true,labels):
    cm = confusion_matrix(y_true,y_preds,normalize="true")
    fig, ax = plt.subplots(figsize=(6,6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
    plt.title("Normalized confusion matrix")
    plt.show()
y_preds = lr_clf.predict(x_valid)
plot_confusion_matrix(y_preds, y_valid, labels)