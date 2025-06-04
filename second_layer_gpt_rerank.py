import pandas as pd
import json
from datetime import datetime
import spacy
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
import os
from transformers import *
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import spacy
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
import random
from sklearn.metrics import ndcg_score
openai_api_key = ""
# 

def llm_construct(openai_api_key):
    # llm = Ollama(model="mixtral:8x7b-instruct-v0.1-q5_K_M", openai_api_key=openai_api_key)
    # llm = Ollama(model="mixtral:8x7b-instruct-v0.1-q5_K_M")
    # gpt-3.5-turbo-1106
    # llm = ChatOpenAI(model="gpt-3.5-turbo-1106", openai_api_key=openai_api_key)
    llm = ChatOpenAI(model="gpt-4", openai_api_key=openai_api_key)
    # llm = Ollama(model="mixtral:8x7b-instruct-v0.1-q5_K_M")
    # llm = ChatOllama(model="llama3")
    return llm
    

def prompt_construct(template):
    prompt = PromptTemplate(template=template, input_variables=["info_data"])
    return prompt

def event_info_generation(model, prompt):
    llm_chain = LLMChain(prompt=prompt, llm=model)
    return llm_chain


def run_LLM():
    data = pd.read_pickle("./processed_data/Symptom2Disease/pred_disease.pkl")

    f = open("./processed_data/Symptom2Disease/gpt4_pred_disease_80.txt", "a+")
    for idx, row in data.iterrows():
        candidate_disease = ', '.join(row['pred_disease'][0:80])
        prompt = ChatPromptTemplate.from_template(prompt_generation(patient_symptom=row['symptoms'],candidate_disease=candidate_disease))
        llm = llm_construct(openai_api_key)
        llm_chain = event_info_generation(llm, prompt)
        res = llm_chain.run(patient_symptom=row['symptoms'],candidate_disease=candidate_disease)
        f.write("symptoms: " + row["symptoms"])
        f.write("\n")
        f.write("real_disease: " + row["disease_name"])
        f.write("\n")
        f.write("gpt_results: " + res)
        f.write("\n")
        f.write("=====================================")
        f.write("\n")

    f.close()



def prompt_generation(patient_symptom, candidate_disease):

    template = """
    Assume you are a doctor and you need predict the patient's potential disease. I will provide you with the patient's self-described symptoms and the possible diseases.
    Patient's symptoms: {patient_symptom}
    Candidate diseases: {candidate_disease}
    Please to the following tasks:
    1. Re-rank the candidate diseases based on the possibility that the patient might catch. Match the exact disease names. Add the order before the disease name and separete them commas. Include all candidate diseases.
    2. To be more confident about the disease diagnosis, what other information do you want to ask for the patients?

    Answer the question by filling in the <answer> below, match the exact format described above.

    Re-ranked diseases: <answer>
    Other information to request: <answer>

    """

    return template




def prompt_disease_summarize(disease_raw):

    template = """
    Below are the potential diseases information of a patient: {disease_raw}, please summarize the diseases with specific disease names, separate each disease with a comma.
    
    Based on the above information, please summarize and reformat the information by filling in the <answer> below, following the format proposed in the demostration, no explanation needed:

    diseases: <answer>
    
    """

    return template

# run_LLM_disease()
run_LLM()