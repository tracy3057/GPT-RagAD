import pandas as pd
from datasets import load_dataset
import pandas as pd
import json
from datetime import datetime
import heapq
# import spacy
from collections import defaultdict 
import pickle
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
import os
# from transformers import *
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
# import spacy
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
import random
from sklearn.metrics import ndcg_score
from sentence_transformers import SentenceTransformer

openai_api_key = ""

def symptoms():
    dataset = load_dataset("celikmus/mayo_clinic_symptoms_and_diseases_v1",split="train")
    all_symptoms = pd.DataFrame()
    symptom_list = []
    disease_name_list = []
    for item in dataset:
        symptom_list.append(item['text'])
        disease_name_list.append(item['label'])
    all_symptoms['disease_name'] = disease_name_list
    all_symptoms['symptoms'] = symptom_list
    all_symptoms.to_pickle("./processed_data/Symptom2Disease/mayo_symptoms.pkl")



def raw_data_extract():
    dataset = load_dataset("gretelai/symptom_to_diagnosis")
    symptom_diagnose_df = pd.DataFrame()
    disease_list = []
    symptom_list = []
    for item in dataset['train']:
        disease_list.append(item['output_text'])
        symptom_list.append(item['input_text'])
    for item in dataset['test']:
        disease_list.append(item['output_text'])
        symptom_list.append(item['input_text'])
    symptom_diagnose_df['diagnosis'] = disease_list
    symptom_diagnose_df['symptoms'] = symptom_list
    print(symptom_diagnose_df)
    symptom_diagnose_df.to_pickle("./processed_data/Symptom2Disease/symptom_to_diagnosis.pkl")

def llm_construct(openai_api_key):
    # llm = Ollama(model="mixtral:8x7b-instruct-v0.1-q5_K_M", openai_api_key=openai_api_key)
    # llm = Ollama(model="mixtral:8x7b-instruct-v0.1-q5_K_M")
    llm = ChatOpenAI(model="gpt-4", openai_api_key=openai_api_key)
    # llm = ChatOllama(model="llama3")
    return llm

def prompt_construct(template):
    prompt = PromptTemplate(template=template, input_variables=["info_data"])
    return prompt

def event_info_generation(model, prompt):
    llm_chain = LLMChain(prompt=prompt, llm=model)
    return llm_chain

def prompt_generation(patient_symptom):

    template = """
    Please summarize the symptoms of the patient based on the description {patient_symptom}. Use a word or a phrase to summarize each symptom and separate each symptom by a comma, no explanation needed.
    """

    return template


def prompt_generation_mayo(disease, symptoms):

    template = """
    Please summarize the symptoms of {disease} based on the following content. Use a word or a phrase to summarize each symptom and separate the symptoms by comma, no explanation needed. 
    The content is: {symptoms}
    """

    return template


def user_symptom_generate():
    data = pd.read_pickle("./processed_data/conversation.pkl".format("merged_data"))
    print(set(data.disease_result))

    f = open("./processed_data/patient_symptom.txt".format("merged_data"), "w")
    for idx, row in data.iterrows():
        prompt = ChatPromptTemplate.from_template(prompt_generation(row['basic_info']))
        llm = llm_construct(openai_api_key)
        llm_chain = event_info_generation(llm, prompt)
        res = llm_chain.run(patient_symptom=row['basic_info'])
        f.write("index: " + str(idx))
        f.write("\n")
        f.write("disease_name: " + row["disease_result"])
        f.write("\n")
        f.write("raw_text: " + row["basic_info"])
        f.write("\n")
        f.write("gpt_results: " + res)
        f.write("\n")
        f.write("=====================================")
        f.write("\n")

    f.close()

def mayo_symptom_generate():
    data = pd.read_pickle("./processed_data/Symptom2Disease/mayo_symptoms.pkl")

    f = open("./processed_data/Symptom2Disease/mayo_symptoms.txt", "w")
    for idx, row in data.iterrows():
        prompt = ChatPromptTemplate.from_template(prompt_generation_mayo(row['disease_name'], row['symptoms']))
        llm = llm_construct(openai_api_key)
        llm_chain = event_info_generation(llm, prompt)
        res = llm_chain.run(disease=row['disease_name'], symptoms=row['symptoms'])
        f.write("index: " + str(idx))
        f.write("\n")
        f.write("disease_name: " + row["disease_name"])
        f.write("\n")
        f.write("gpt_results: " + res)
        f.write("\n")
        f.write("=====================================")
        f.write("\n")

    f.close()


def disease_graph_gen():
    disease_info = pd.read_pickle("./processed_data/Symptom2Disease/disease_info.pkl")
    print(disease_info)
    node_id_map = {}
    node_idx = 0
    for idx, row in disease_info.iterrows():
        if row['disease_names'] not in node_id_map.keys():
            node_id_map[row['disease_names']] = node_idx
            node_idx += 1
    for idx, row in disease_info.iterrows():
        for symptom in row['symptoms']:
            if symptom not in node_id_map.keys():
                node_id_map[symptom] = node_idx
                node_idx += 1
    print(node_id_map)
    node_names = list(node_id_map.keys())
    disease_node_df = pd.DataFrame()
    disease_node_df['node_name'] = list(node_id_map.keys())
    disease_node_df['node_id'] = list(node_id_map.values())
    # disease_node_df['embeddings']
    model = SentenceTransformer("all-MiniLM-L6-v2")
    sentence_embeddings = model.encode(node_names)
    disease_node_df['embeddings'] = sentence_embeddings.tolist()
    feature = torch.Tensor(list(disease_node_df['embeddings'].values))
    torch.save(feature, "./processed_data/Symptom2Disease/features.pt")
    print(feature.shape)
    adj = defaultdict(list) 
    symptom_stats = {}
    for idx, row in disease_info.iterrows():
        adj[node_id_map[row['disease_names']]] = []
        for symptom in row['symptoms']:
            adj[node_id_map[row['disease_names']]].append(node_id_map[symptom])
            if symptom not in symptom_stats:
                symptom_stats[symptom] = 1
            else:
                symptom_stats[symptom] += 1
    print(symptom_stats)
    with open("./processed_data/Symptom2Disease/symptom_stats.json","w") as f:
        json.dump(symptom_stats, f)
    pickle.dump(adj,open("./processed_data/Symptom2Disease/adj.pkl","wb"), protocol=pickle.HIGHEST_PROTOCOL)
    with open("./processed_data/Symptom2Disease/node_id_map.json","w") as f:
        json.dump(node_id_map,f)

def mayo_symptom_process():
    with open("./processed_data/Symptom2Disease/mayo_symptoms.txt", "r") as f:
        patient_symptom = f.read()
    patient_symptoms = patient_symptom.split("\n=====================================")
    patient_symptoms = [i for i in patient_symptoms if i != '\n']
    user_symptom_df = pd.DataFrame()
    patient_diagnosis_list = []
    patient_rawtext_list = []
    patient_symptom_list = []
    for item in patient_symptoms:
        try:
            disease_name = re.search(r'disease_name: (.*?)\ngpt_results: ', item).group(1)
        # if disease_name.lower() not in disease_name_list_all:
        #     continue
            symptoms = re.search(r'gpt_results: (.*?)$', item).group(1).lower()
            symptoms_items = symptoms.split(', ')
        except:
            continue
        patient_diagnosis_list.append(disease_name)
        patient_symptom_list.append(symptoms_items)
    user_symptom_df['disease_name'] = patient_diagnosis_list
    user_symptom_df['symptoms'] = patient_symptom_list

    print(user_symptom_df)
    user_symptom_df.to_pickle("./processed_data/{}/mayo_conversation.pkl".format("Symptom2Disease"))


def mayo_disease_graph_gen():
    disease_info = pd.read_pickle("./processed_data/Symptom2Disease/mayo_conversation.pkl")
    print(disease_info)
    node_id_map = {}
    node_idx = 0
    for idx, row in disease_info.iterrows():
        if row['disease_name'] not in node_id_map.keys():
            node_id_map[row['disease_name']] = node_idx
            node_idx += 1
    for idx, row in disease_info.iterrows():
        for symptom in row['symptoms']:
            if symptom not in node_id_map.keys():
                node_id_map[symptom] = node_idx
                node_idx += 1
    print(node_id_map)
    node_names = list(node_id_map.keys())
    disease_node_df = pd.DataFrame()
    disease_node_df['node_name'] = list(node_id_map.keys())
    disease_node_df['node_id'] = list(node_id_map.values())
    # disease_node_df['embeddings']
    model = SentenceTransformer("all-MiniLM-L6-v2")
    sentence_embeddings = model.encode(node_names)
    disease_node_df['embeddings'] = sentence_embeddings.tolist()
    feature = torch.Tensor(list(disease_node_df['embeddings'].values))
    torch.save(feature, "./processed_data/Symptom2Disease/mayo_features.pt")
    print(feature.shape)
    adj = defaultdict(list) 
    symptom_stats = {}
    for idx, row in disease_info.iterrows():
        adj[node_id_map[row['disease_name']]] = []
        for symptom in row['symptoms']:
            adj[node_id_map[row['disease_name']]].append(node_id_map[symptom])
            if symptom not in symptom_stats:
                symptom_stats[symptom] = 1
            else:
                symptom_stats[symptom] += 1
    print(symptom_stats)
    with open("./processed_data/Symptom2Disease/mayo_symptom_stats.json","w") as f:
        json.dump(symptom_stats, f)
    pickle.dump(adj,open("./processed_data/Symptom2Disease/mayo_adj.pkl","wb"), protocol=pickle.HIGHEST_PROTOCOL)
    with open("./processed_data/Symptom2Disease/mayo_node_id_map.json","w") as f:
        json.dump(node_id_map,f)



def user_symptom_process(theme_name):
    with open("./processed_data/merged_data/patient_symptom.txt", "r") as f:
        patient_symptom = f.read()
    patient_symptoms = patient_symptom.split("\n=====================================")
    patient_symptoms = [i for i in patient_symptoms if i != '\n']
    user_symptom_df = pd.DataFrame()
    patient_diagnosis_list = []
    patient_rawtext_list = []
    patient_symptom_list = []
    for item in patient_symptoms:
        disease_name = re.search(r'disease_name: (.*?)\nraw_text: ', item).group(1)
        # if disease_name.lower() not in disease_name_list_all:
        #     continue
        raw_text = re.search(r'raw_text: (.*?)\ngpt_results: ', item).group(1)
        symptoms = re.search(r'gpt_results: (.*?)$', item).group(1).lower()
        symptoms_items = symptoms.split(', ')
        patient_diagnosis_list.append(disease_name)
        patient_rawtext_list.append(raw_text)
        patient_symptom_list.append(symptoms_items)
    user_symptom_df['disease_name'] = patient_diagnosis_list
    user_symptom_df['raw_text'] = patient_rawtext_list
    user_symptom_df['symptoms'] = patient_symptom_list

    print(user_symptom_df)
    user_symptom_df.to_pickle("./processed_data/{}/conversation.pkl".format(theme_name))


def candidate_disease_gen():
    embedding = np.load("./processed_data/Symptom2Disease/mayo_node_embeddings_disease.npy")
    with open("./processed_data/Symptom2Disease/mayo_adj.pkl","rb") as f:
        mayo_adj = pickle.load(f)
    print(mayo_adj)
    symptom_stats = json.load(open("./processed_data/Symptom2Disease/mayo_symptom_stats.json"))
    with open("./processed_data/Symptom2Disease/mayo_node_id_map.json","r") as f:
        node_id_map = json.load(f)
    print(node_id_map)
    theme_name = "Symptom2Disease"
    conversation = pd.read_pickle("./processed_data/{}/conversation.pkl".format(theme_name))
    # conversation = conversation.loc[~conversation['disease_name'].isin(['fungal infection', 'allergy', 'drug reaction'])]
    disease_info = pd.read_pickle("./processed_data/{}/mayo_conversation.pkl".format(theme_name))
    disease_list= list(disease_info.disease_name.values)
    disease_embedding = {}
    symptom_embedding = {}

    pred_res_df = pd.DataFrame()
    symptom_list = []
    disease_label_list = []
    pred_disease_list = []
    confident_score_list = []

    for key, value in node_id_map.items():
        if value <= 938:
            cur_disease_embeddings = []
            for cur_symptom in mayo_adj[value]:
                cur_disease_embeddings.append(embedding[cur_symptom])
            disease_embedding[key] = np.asarray(cur_disease_embeddings).mean(axis=0)
        else:
            symptom_embedding[key] = embedding[value]
    # print(symptom_embedding)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    sentence_embeddings = model.encode(list(symptom_embedding.keys()))
    symptom_text_embedding = sentence_embeddings.tolist()
    symptom_text_embedding_dict = dict(zip(list(symptom_embedding.keys()),symptom_text_embedding))
    print(conversation.columns)
    all_cnt = 0
    hit_cnt = 0
    for idx, row in conversation.iterrows():
        patient_embedding = []
        # symptom_idx = 1
        for symptom in row['symptoms']:
            # if symptom not in symptom_embedding.keys():
            cur_embedding = model.encode(symptom)
            res = cosine_similarity(cur_embedding.reshape(1, -1),list(symptom_text_embedding_dict.values()))[0].tolist()
            res_ranked = heapq.nlargest(len(res), res)
            res_ranked_idx = [res.index(i) for i in res_ranked]
            selected_symptom = [list(symptom_embedding.keys())[i] for i in res_ranked_idx]
            # if max(res) <0.8:
            #     continue
            cur_symptom_embedding = []
            for symptom_idx in range(len(selected_symptom)):
                if res_ranked[symptom_idx] <0.8:
                    if symptom_idx == 0:
                        similar_symptom = list(symptom_embedding.keys())[res_ranked_idx[symptom_idx]]
                        cur_symptom_embedding.append([res_ranked[symptom_idx] * i for i in symptom_embedding[similar_symptom]])
                    break
                similar_symptom = list(symptom_embedding.keys())[res_ranked_idx[symptom_idx]]
                cur_symptom_embedding.append([res_ranked[symptom_idx] * i for i in symptom_embedding[similar_symptom]])
            cur_symptom_embedding = np.asarray(cur_symptom_embedding).mean(axis=0)
            patient_embedding.append(cur_symptom_embedding)
        cur_conversation_embedding = np.asarray(patient_embedding).mean(axis=0)
        try:
            pred_disease_res = cosine_similarity(cur_conversation_embedding.reshape(1, -1),list(disease_embedding.values()))[0].tolist()
            all_cnt += 1
        except:
            continue
        res2 = [pred_disease_res.index(i) for i in heapq.nlargest(len(pred_disease_res), pred_disease_res)]
        selected_pred = [list(disease_embedding.keys())[i] for i in res2]
        # print(row['disease_name'].lower() in disease_embedding.keys())
        if not row['disease_name'].lower() in disease_embedding.keys():
            continue
        if row['disease_name'] in selected_pred:
            hit_cnt += 1
        print(row['raw_text'])
        symptom_list.append(row['raw_text'])
        print(selected_pred[0:30])

        # print(heapq.nlargest(3, pred_disease_res))
        print(row['disease_name'])
        disease_label_list.append(row['disease_name'])
        pred_disease_list.append(selected_pred)
        confident_score_list.append(heapq.nlargest(len(pred_disease_res), pred_disease_res))
    print(hit_cnt/all_cnt)
    pred_res_df['symptoms'] = symptom_list
    pred_res_df['disease_name'] = disease_label_list
    pred_res_df['pred_disease'] = pred_disease_list
    pred_res_df['confident_score'] = confident_score_list
    pred_res_df.to_pickle("./processed_data/merged_data/pred_disease.pkl")
    
# disease_graph_gen()
candidate_disease_gen()
