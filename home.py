import streamlit as st
import admin as ad
import pandas as pd
import ATE_for_csv as pfc
import ate_sc as sc
import ATE as at
from streamlit_option_menu import option_menu

st.title("Aspect Based Sentiment Analysis")

upload, approaches, chart, export, admin = st.tabs(["upload", "approaches", "chart", "export", "admin"])

with upload:
   st.header("upload")
   file1 = st.file_uploader("upload your file : ",type=["csv","xlsx","txt"])
   if file1 is not None:
        # Check the file type
        file_extension = file1.name.split('.')[-1]
        if file_extension.lower() in 'csv':
            df = pd.read_csv(file1)
            st.dataframe(df)
        elif file_extension.lower() in 'xlsx':
            df = pd.read_excel(file1)
            st.dataframe(df)
            

        elif file_extension.lower() in 'txt':
            text_1 = file1.read()
            st.text(text_1)
   

with approaches:
   st.header("approaches")
   approaches = st.selectbox("choose the approaches",["Aspect Term Extraction","Aspect Term Extraction for csv","Opinion Term Extraction","Aspect-Level Sentiment Classification",
                               "Aspect-Oriented Opinion Extraction","Aspect Term Extraction and Sentiment Classification",
                               "Pair Extraction","Triplet Extraction"])
   
   if st.button("submit"):
       if approaches == "Aspect Term Extraction and Sentiment Classification":
           ttt = st.text_input("Enter the review column name: ")
           if ttt:  # Check if the text_input is not empty
               st.dataframe(sc.call(df, ttt))
       elif approaches == "Aspect Term Extraction for csv":
          st.dataframe(pfc.ATE_for_csv(df))
       



with chart:
   st.header("this is chart page")
   
    
    
    
    
    
 
            
        
        


    
   







