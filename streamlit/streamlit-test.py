import streamlit as st
st.title("Streamlit Sample Webapp")

name = st.text_input("Enter your name")

if st.button("say hello"):
    if name:
        st.success(f"hello {name}, Welcome to my page")
    else:
        st.warning("please enter your name")