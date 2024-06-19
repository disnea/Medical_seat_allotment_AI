import streamlit as st
import pandas as pd
import spacy

# Load Excel data from a predefined path
@st.cache_data
def load_data(file_path):
    df = pd.read_excel(file_path)
    return df

# Function to extract relevant information from the query
def extract_college_course_category_and_rank(query, nlp):
    doc = nlp(query)
    college = None
    course = None
    category = None
    rank = None
    for ent in doc.ents:
        if ent.label_ == "ORG":
            college = ent.text
        elif ent.label_ in ["COURSE", "GPE", "NORP"]:  # Adding more labels for courses and categories
            course = ent.text
        elif ent.label_ == "CATEGORY":
            category = ent.text
        elif ent.label_ == "CARDINAL":  # Assuming rank is labeled as CARDINAL by spaCy
            rank = int(ent.text)
    return college, course, category, rank

# Function to filter data based on the extracted college name, course, category, and rank
def filter_data(df, college_name=None, course_name=None, category=None, rank=None):
    filtered_df = df
    if college_name:
        filtered_df = filtered_df[filtered_df["Allotted Institute"].str.contains(college_name, case=False, na=False)]
    if course_name:
        filtered_df = filtered_df[filtered_df["Course Alloted"].str.contains(course_name, case=False, na=False)]
    if category:
        filtered_df = filtered_df[filtered_df["Candidate Category"].str.contains(category, case=False, na=False)]
    if rank is not None:
        filtered_df = filtered_df[filtered_df["Rank"] <= rank]
    return filtered_df

# Streamlit App
def main():
    st.title("Medical College Allotment AI Assistant")

    # Load data from predefined local path
    file_path = "D:\\projects\\python ML projects\\Medical-LLm\\New folder\\MCI_1st_LIST_ALLOTMENT ALLINDIA.xlsx"
    data = load_data(file_path)
    
    st.write("## Data Preview")
    st.dataframe(data.head())

    st.write("### How to Ask Your Question")
    st.markdown("""
    Here are some tips on how to phrase your question:
    - Include the name of the college (e.g., 'Vardhaman Mahavir Medical College').
    - Optionally, include the course name (e.g., 'MBBS').
    - Specify the category if applicable (e.g., 'general', 'SC', 'ST', 'OBC', 'EWS').
    - Mention your rank if you want to know what colleges and courses you can expect (e.g., 'My rank is 10000').
    Examples:
    - What rank should I get for Vardhaman Mahavir Medical College?
    - What is the cutoff for general category at Vardhaman Mahavir Medical College?
    - What is the cutoff for SC category at Maulana Azad Medical College?
    - Which rank is needed for MBBS in Kasturba Medical College?
    - What rank is required for EWS category at King George's Medical University?
    - My rank is 10000, what colleges can I expect and what courses can I expect?
    """)

    query = st.text_input("Ask a question based on the data above")

    if query:
        nlp = spacy.load("en_core_web_sm")
        college_name, course_name, category, rank = extract_college_course_category_and_rank(query, nlp)

        results = filter_data(data, college_name, course_name, category, rank)
        
        if not results.empty:
            if rank is not None:
                st.write(f"With a rank of {rank}, you can expect the following colleges and courses:")
            else:
                st.write(f"To get into {college_name}" + (f" for the {course_name} course" if course_name else "") + (f" under {category} category" if category else "") + ", you should aim for a rank within the following range:")
            st.write("## Detailed Results")
            st.dataframe(results)
        else:
            st.write(f"No results found with the given criteria.")
    else:
        st.write("Enter a query to get the answer.")

if __name__ == "__main__":
    main()
