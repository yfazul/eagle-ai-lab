import streamlit as st

# App title
st.title("🧮 Simple Calculator")

# Input fields
st.write("Enter two numbers below:")
num1 = st.number_input("First number", value=0.0)
num2 = st.number_input("Second number", value=0.0)

# Operation selection
operation = st.selectbox(
    "Choose an operation:",
    ("Addition (+)", "Subtraction (-)", "Multiplication (×)", "Division (÷)")
)

# Perform calculation when button clicked
if st.button("Calculate"):
    if operation == "Addition (+)":
        result = num1 + num2
        st.success(f"Result: {num1} + {num2} = {result}")
    elif operation == "Subtraction (-)":
        result = num1 - num2
        st.success(f"Result: {num1} - {num2} = {result}")
    elif operation == "Multiplication (×)":
        result = num1 * num2
        st.success(f"Result: {num1} × {num2} = {result}")
    elif operation == "Division (÷)":
        if num2 == 0:
            st.error("Error: Cannot divide by zero!")
        else:
            result = num1 / num2
            st.success(f"Result: {num1} ÷ {num2} = {result}")
