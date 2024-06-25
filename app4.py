import pickle
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer
import nltk

nltk.download('punkt')

# Load datasets
job_data = pd.read_csv("hf://datasets/jacob-hugging-face/job-descriptions/training_data.csv")
courses_list = pickle.load(open('courses.pkl', 'rb'))

# Rename columns immediately after loading
courses_list.rename(columns={
    'Course Rating': 'rating',
    'University': 'university',
    'Course URL': 'course_url',
    'Course Description': 'Course Description 2'
}, inplace=True)

def clean_text(text):
    """ Apply multiple replacements for text formatting. """
    replacements = {
        ':': '',    # Remove colons
        '_': '',    # Remove underscores
        '(': '',    # Remove opening parenthesis
        ')': ''     # Remove closing parenthesis
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

# Clean the text data
job_data['job_description'] = job_data['job_description'].apply(clean_text)
job_data['model_response'] = job_data['model_response'].apply(clean_text)

# Create tags column and clean text
ps = PorterStemmer()
cv = CountVectorizer(max_features=5000, stop_words='english')
job_data['tags'] = job_data['job_description'].apply(lambda x: " ".join([ps.stem(word) for word in nltk.word_tokenize(x.lower())]))
courses_list['tags'] = courses_list['Course Description 2'].apply(lambda x: " ".join([ps.stem(word) for word in nltk.word_tokenize(x.lower())]))

job_vectors = cv.fit_transform(job_data['tags'])
course_vectors = cv.transform(courses_list['tags'])
similarity = cosine_similarity(job_vectors, course_vectors)

def recommend(job_title):
    """ Recommend courses based on job title. """
    try:
        job_index = job_data[job_data['position_title'] == job_title].index[0]
        distances = similarity[job_index]
        course_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:7]
        recommended_courses = [
            (
                courses_list.iloc[i[0]]['course_name'],
                courses_list.iloc[i[0]]['university'],
                courses_list.iloc[i[0]]['rating'],
                courses_list.iloc[i[0]]['Course Description2'],
                courses_list.iloc[i[0]]['course_url'],
                i[1]
            )
            for i in course_list
        ]
        return recommended_courses
    except IndexError:
        st.error("Job title not found in the dataset.")
        return []

# Streamlit UI with layout
st.markdown("# Job Description Course Recommendation System")
st.markdown("## Find similar courses based on job descriptions")

# Implementing searchable select box
selected_job = st.selectbox(
    "Type or select a job you like:",
    options=job_data['position_title'],
    format_func=lambda x: x if x else "Select a position..."  # This will show this text by default
)

# Display job description in an expander
if selected_job:
    with st.expander("Show/Hide Job Description"):
        st.write(job_data[job_data['position_title'] == selected_job]['job_description'].iloc[0])

# Show Recommended Courses Button
if st.button('Show Recommended Courses') and selected_job:
    recommended_courses = recommend(selected_job)
    if recommended_courses:
        for name, uni, rate, desc, url, score in recommended_courses:
            st.markdown(f"""
                <div style="border:1px solid #ccc; border-radius:5px; padding:10px; margin:10px 0;">
                    <h4>{name} (Rating: {rate}/5)</h4>
                    <h5>{uni}</h5>
                    <p>{desc}</p>
                    <a href="{url}" target="_blank">Go to course</a>
                    <p>Similarity score: {score:.2f}</p>
                </div>
            """, unsafe_allow_html=True)
