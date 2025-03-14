import json
from agent import analyze_news
import pandas as pd
import random
import concurrent.futures

dp = "data" #modify this to the directory of Fake.csv and True.csv

df = pd.read_csv(f"{dp}/True.csv")

ids = random.sample(list(range(len(df))), k=1000)

samples = []
for id in ids:
    samples.append({
        "id": id,
        "title": df.iloc[id]["title"],
        "text": df.iloc[id]["text"],
        "subject": df.iloc[id]["subject"],
        "date": df.iloc[id]["date"],  
    })
    
results = []

#Using thread pool to accelerate execution
with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
    
    future_to_sample = {
        executor.submit(analyze_news, sample["title"], sample["text"], sample["date"], 'deepseek'): sample
        for sample in samples
    }
    
    for future in concurrent.futures.as_completed(future_to_sample):
        sample = future_to_sample[future]
        reply = future.result()
        sample["label"] = reply["label"]
        sample["reason"] = reply["reason"]
        results.append(sample)

with open(f"{dp}/true_deepseek.json", 'w') as file:
    json.dump(results, file, indent=4)


df = pd.read_csv(f"{dp}/Fake.csv")

ids = random.sample(list(range(len(df))), k=1000)

samples = []
for id in ids:
    samples.append({
        "id": id,
        "title": df.iloc[id]["title"],
        "text": df.iloc[id]["text"],
        "subject": df.iloc[id]["subject"],
        "date": df.iloc[id]["date"],  
    })
    
results = []

with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
    
    future_to_sample = {
        executor.submit(analyze_news, sample["title"], sample["text"], sample["date"], 'deepseek'): sample
        for sample in samples
    }
    
    for future in concurrent.futures.as_completed(future_to_sample):
        sample = future_to_sample[future]
        reply = future.result()
        sample["label"] = reply["label"]
        sample["reason"] = reply["reason"]
        results.append(sample)

with open(f"{dp}/fake_deepseek.json", 'w') as file:
    json.dump(results, file, indent=4)


df = pd.read_csv(f"{dp}/True.csv")

ids = random.sample(list(range(len(df))), k=1000)

samples = []
for id in ids:
    samples.append({
        "id": id,
        "title": df.iloc[id]["title"],
        "text": df.iloc[id]["text"],
        "subject": df.iloc[id]["subject"],
        "date": df.iloc[id]["date"],  
    })
    
results = []

with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
    
    future_to_sample = {
        executor.submit(analyze_news, sample["title"], sample["text"], sample["date"], 'llama'): sample
        for sample in samples
    }
    
    for future in concurrent.futures.as_completed(future_to_sample):
        sample = future_to_sample[future]
        reply = future.result()
        sample["label"] = reply["label"]
        sample["reason"] = reply["reason"]
        results.append(sample)

with open(f"{dp}/true_llama.json", 'w') as file:
    json.dump(results, file, indent=4)


df = pd.read_csv(f"{dp}/Fake.csv")

ids = random.sample(list(range(len(df))), k=1000)

samples = []
for id in ids:
    samples.append({
        "id": id,
        "title": df.iloc[id]["title"],
        "text": df.iloc[id]["text"],
        "subject": df.iloc[id]["subject"],
        "date": df.iloc[id]["date"],  
    })
    
results = []

with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
    
    future_to_sample = {
        executor.submit(analyze_news, sample["title"], sample["text"], sample["date"], 'llama'): sample
        for sample in samples
    }
    
    for future in concurrent.futures.as_completed(future_to_sample):
        sample = future_to_sample[future]
        reply = future.result()
        sample["label"] = reply["label"]
        sample["reason"] = reply["reason"]
        results.append(sample)

with open(f"{dp}/fake_llama.json", 'w') as file:
    json.dump(results, file, indent=4)


