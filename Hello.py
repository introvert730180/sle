import streamlit as st
from joblib import load
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from keras.models import Sequential
from tensorflow.keras.models import load_model

# Load the saved model using the built-in load_model function
modelCounter = load_model('model2.h5')
modelCross = load_model('model.h5')


# Load the scaler object from the file
scaler = load('scaler.joblib')
scaler3 = load('scaler3.joblib')



def main():
    st.title("Comparison of counter and cross current SLE on the basis of solvent and its constituent :--")

    # Input boxes with default values and hints
    input3 = st.number_input("Solvent", value=None, step=None, format="%.2f", help="Please fill the solution amount")
    input4 = st.number_input("Solute conc.", value=None, step=None, format="%.4f",
                             help="Please fill the amount of solute in feed")

    # Button to trigger action
    if st.button("Calculate"):
        if input3 is not None and input4 is not None:
            # Show text fields if all input boxes have numeric values
            a = scaler.fit_transform([[input3]])
            b = scaler3.fit_transform([[input3]])
            new_test_case = pd.DataFrame([[0.0, 0.195, a, input4]],
                                         columns=['feature1', 'feature2', 'feature3', 'feature4'])
            new_test_case1 = pd.DataFrame([[0.0, 0.195, b, input4]],
                                         columns=['feature1', 'feature2', 'feature3', 'feature4'])

            # Make predictions using the model
            predictions_cross = modelCross.predict(new_test_case)
            predictions_counter = modelCounter.predict(new_test_case1)

            # Create empty text fields
            text_field_cross = st.empty()
            text_field_counter = st.empty()

            # Display predictions in text fields
            text_field_cross.write("Text Field 1:")
            text_field_cross.text_input("% Overall Cross removal", str(predictions_cross[0][0]))
            text_field_counter.write("Text Field 2:")
            text_field_counter.text_input("% Overall Counter removal", str(predictions_counter[0][0]))

        else:
            # Display error message if any input box is empty
            st.error("All input boxes must have numeric values or zero.")




if __name__ == "__main__":
    main()
