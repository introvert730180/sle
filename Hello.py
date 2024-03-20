import streamlit as st
from joblib import load
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# Load the saved model using the built-in load_model function
modelCounter = load_model('model2.h5')
modelCross = load_model('model.h5')

# Load the scaler object from the file
scaler = load('scaler.joblib')
scaler3 = load('scaler3.joblib')

# Custom CSS for styling
custom_css = """
    <style>
        /* Add custom CSS styles here */
        .gradient-background {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(to bottom, #bdc3c7, #2c3e50);
        }
        .content {
            padding: 20px;
        }
        .stButton>button {
            background-color: #FF5733;
            color: white;
            font-weight: bold;
            font-size: 16px;
            padding: 10px 20px;
            border-radius: 5px;
            transition: background-color 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #FF7F50;
        }
        .stTextInput>div>div>input {
            background-color: #FFF0E0;
            color: #333;
            font-size: 16px;
            border-radius: 5px;
        }
        .stTextInput>div>label {
            font-size: 16px;
            color: #333;
        }
        .stText>div>div>div>label {
            font-size: 16px;
            color: #333;
        }
        .stText>div>div>div>span>span {
            font-size: 16px;
            color: #333;
        }
    </style>
"""

# Add custom CSS styles
st.markdown(custom_css, unsafe_allow_html=True)

def main():
    # Adding a gradient background
    st.markdown('<div class="gradient-background"></div>', unsafe_allow_html=True)

    # Main content
    st.markdown('<div class="content">', unsafe_allow_html=True)
    st.title("Comparison of counter and cross current SLE on the basis of solvent and its constituent :--")

    # Input boxes with default values and hints
    input3, input4 = st.columns(2)
    with input3:
        solvent_input = st.number_input("Solvent", value=None, step=None, format="%.2f", help="Please fill the solution amount")
    with input4:
        solute_input = st.number_input("Solute conc.", value=None, step=None, format="%.4f", help="Please fill the amount of solute in feed")

    # Button to trigger action
    if st.button("Calculate"):
        if solvent_input is not None and solute_input is not None:
            # Show text fields if all input boxes have numeric values
            a = scaler.fit_transform([[solvent_input]])
            b = scaler3.fit_transform([[solvent_input]])
            new_test_case = pd.DataFrame([[0.0, 0.195, a, solute_input]],
                                         columns=['feature1', 'feature2', 'feature3', 'feature4'])
            new_test_case1 = pd.DataFrame([[0.0, 0.195, b, solute_input]],
                                          columns=['feature1', 'feature2', 'feature3', 'feature4'])

            # Make predictions using the model
            predictions_cross = modelCross.predict(new_test_case)
            predictions_counter = modelCounter.predict(new_test_case1)

            # Display predictions in text fields
            output1, output2 = st.columns(2)
            with output1:
                st.write("Text Field 1:")
                st.text_input("% Overall Cross removal", str(predictions_cross[0][0]))
            with output2:
                st.write("Text Field 2:")
                st.text_input("% Overall Counter removal", str(predictions_counter[0][0]))

        else:
            # Display error message if any input box is empty
            st.error("All input boxes must have numeric values or zero.")

    st.markdown('</div>', unsafe_allow_html=True)  # Closing the content div

if __name__ == "__main__":
    main()
