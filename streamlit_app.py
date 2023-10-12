import streamlit as st
import numpy as np
import statsmodels.api as sm

def perform_simple_linear_regression(y, x):
    if len(y) != len(x):
        st.error("# of elements in dependent + independent variables must be the same.")
        return

    x = sm.add_constant(x)  # Adds a constant term to the predictor
    model = sm.OLS(y, x)
    results = model.fit()

    st.subheader("Regression Results:")
    st.text(results.summary())

def main():
    st.title("Simple Linear Regression App")

    # Streamlit widgets for user input
    y_input = st.text_input("Dependent variable separated by commas:")
    x_input = st.text_input("Independent variable separated by commas:")
    submit_button = st.button("Run Linear Regression")

    if submit_button:
        y = np.array(list(map(float, y_input.split(","))))
        x = np.array(list(map(float, x_input.split(","))))
        perform_simple_linear_regression(y, x)

if __name__ == "__main__":
    main()
