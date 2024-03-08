import streamlit as st
import openai
import base64, re, os
from streamlit_tags import st_tags
from pdfminer.high_level import extract_text
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer, util
import torch, pandas
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()
openai.api_key = os.environ.get('OPENAI_API_KEY')

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
def get_similarity_text(jd, resume_embedding):
    '''finding similarity between two texts'''
    # calculating cosine similarity
    jd_embedding = model.encode(jd, convert_to_tensor=True)

    cos_scores = util.cos_sim(jd_embedding, resume_embedding)[0]
    top_results = torch.topk(cos_scores, k=1)
    score = top_results[0].numpy()[0]

    return score
# Load English tokenizer, tagger, parser, NER, and word vectors
# nlp = spacy.load("en_core_web_sm")


def pdf_reader(pdf_file):

    text = extract_text(pdf_file).lower()
    skill = text.split("skills")[1]
    os.remove(pdf_file)
#     get keywords
    keywords = " ".join(re.findall(r'[a-zA-Z]\w+',skill.lower()))

    token_text = word_tokenize(keywords)
    stop_words = stopwords.words('english')
    clean_text = []
    for i in token_text:
        if i not in stop_words:
            clean_text.append(i)
    clean_text = " ".join(clean_text)
    
    pattern = re.compile(r'[^a-zA-Z0-9\s]')
    clean_text = re.sub(pattern, '', clean_text).replace("\n", "")
    
        # Define a regular expression pattern to match numbers
    pattern2 = r'\d+'

    # Remove numbers from the text using regex substitution
    text_without_numbers = re.sub(pattern2, '', clean_text)
    
    return text_without_numbers

def show_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    # pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf">'
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def chat_completion(text,jd):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
            "role": "system",
            "content": """
                Act Like a skilled or very experience ATS(Application Tracking System)
                with a deep understanding of Data science, web development. Your task is to evaluate the resume based on the given job description.
                Assign the percentage Matching based on Jd and the missing keywords with high accuracy

                I want the response in dictionary format having the structure
                ("JDMatch":int,"MissingKeywords":[],"Profile Summary":"")
                """
            },
            {
            "role": "user",
            "content": f"resume:{text}\n\njob description:{jd}"
            }
        ],
        temperature=0,
        max_tokens=512,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        #stop=["\n\n"]
    )
    return response["choices"][0]["message"]["content"]

load_data = pandas.read_csv("newdata.csv")
# Streamlit app
def main():
    
    st.title("\t\tJob Matchermaker")
    st.write("Upload a PDF resume to extract summary and get your job match .")
    
    # File upload
    uploaded_file = st.file_uploader("Choose a PDF file", type=".pdf")
    
    if uploaded_file is not None:
        # Read PDF file
        
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())

        
        with st.spinner("Extracting text from PDF..."):
            show_pdf(uploaded_file.name)

            resume_data = pdf_reader(uploaded_file.name)
                
        with st.spinner("Generating embeddings of extracted text..."):
            resume_embedding = model.encode(resume_data, convert_to_tensor=True)   
        
        with st.spinner("Similarity checking..."):
            l = []
            for desc in load_data["Description"]:
                l.append(get_similarity_text(desc,resume_embedding))

        
        with st.spinner("Working on the Analysis...."):
            st.header("**Job recommendation based on Resume**")
            final = sorted(list(enumerate(l)), reverse=True, key=lambda x: x[1])[0:5]

            openai_results = []
            # jobs = []
            job1 = load_data.iloc[final[0][0]]
            job2 = load_data.iloc[final[1][0]]
            job3 = load_data.iloc[final[2][0]]
            job4 = load_data.iloc[final[3][0]]
            # job5 = load_data.iloc[final[4][0]]
            for i,j in final[:4]:
                openai_results.append(chat_completion(resume_data,load_data.iloc[i].Description))
                    
                
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(job1['JobTitles'])
            st.markdown(f"**Company name : {job1['Company_Name']}**")
            st.markdown(f"*Stipend : {job1['Stipend']}*")
            st.markdown(f"[**APPLY**]({job1['Links']})")
            st.subheader("Analysis of Job 1")
            st.caption(openai_results[0])

            st.markdown("----------------")

            st.subheader(job3['JobTitles'])
            st.markdown(f"**Company name : {job3['Company_Name']}**")
            st.markdown(f"*Stipend : {job3['Stipend']}*")
            st.markdown(f"[**APPLY**]({job3['Links']})")
            st.subheader("Analysis of Job 3")
            st.caption(openai_results[2])


        with col2:
            st.subheader(job2['JobTitles'])
            st.markdown(f"**Company name : {job2['Company_Name']}**")
            st.markdown(f"*Stipend : {job2['Stipend']}*")
            st.markdown(f"[**APPLY**]({job2['Links']})")
            st.subheader("Analysis of Job 2")
            st.caption(openai_results[1])

            st.markdown("----------------")

            st.subheader(job4['JobTitles'])
            st.markdown(f"**Company name : {job4['Company_Name']}**")
            st.markdown(f"*Stipend : {job4['Stipend']}*")
            st.markdown(f"[**APPLY**]({job4['Links']})")
            st.subheader("Analysis of Job 4")
            st.caption(openai_results[3])

                
if __name__ == "__main__":
    main()
