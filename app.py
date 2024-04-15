import streamlit as st
import pandas as pd
import os
from Data_Analyser_Agent import analyse_data
from Data_Visualisation_Agent import visualise_data
from Ml_model_builder import model_build
from ML_End_to_End import ml_end_to_end
from interpreter import code_interpret
import base64

def get_image_data(img_path):
    with open(img_path, "rb") as f:
        data = f.read()
        encoded = base64.b64encode(data)
    data = "data:image/png;base64," + encoded.decode("utf-8")
    return data


def show_code_button(code_file_path):
    if st.button("View Code"):
        with open(code_file_path, "r") as f:
            code_content = f.read()
        st.code(code_content)


def main():
    _, img_col, _ = st.columns([1, 3, 1])
    img_col.image("static/ml-devin-Logo.png")

    st.info(
        """
        **Welcome to ML Devin** 🚀
        
        **ML Devin** open source AI tool which helps you build ML model  automating your Machine learning
        task with autogen and  open source e2b interpretor 

       
        """
    )

    with st.sidebar:
        _, img_col, _ = st.columns([1, 3, 1])
        img_col.image("static/ml-devin-Logo.png")
    st.markdown("--- ")
    st.header("🛠️ Tools")
    col1, col2 = st.columns(2)

    video_image_1 = get_image_data("static/ml-devin-Logo.png")
    st.title("Plug your Data let AI build model")


    st.subheader("Upload CSV File")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)


        st.write("Uploaded CSV data:")
        st.write(df)

        save_button = st.button("Save CSV to Local")
        if save_button:
            save_path = os.path.join("coding", "uploaded_data.csv")
            df.to_csv(save_path, index=False)
            st.success(f"CSV file saved to {save_path}")

        st.subheader("Choose an action:")
        option = st.radio("", ("Data Analysis", "Data Visualization", "Model Building", "Complete ML task"))

        if option == "Data Analysis":
            with st.spinner("Analysing Data..."):

                analyse_data()
                show_code_button("coding/data_analyses.py")

                with open("coding/data_analyses.py", "r") as f:
                    code_content = f.read()

            bt=st.button("Execute the code")

            if bt:
                result=code_interpret(code_content)

                st.write("The Code Execution Result")
                st.code(result)




        elif option == "Data Visualization":
            with st.spinner("Visualising Data..."):
                visualise_data()
                show_code_button("coding/data_visualisation.py")

                with open("coding/data_visualisation.py", "r") as f:
                    code_content = f.read()

            bt = st.button("Execute the code")

            if bt:
                result = code_interpret(code_content)

                st.write("The Code Execution Result")
                st.code(result)



        elif option == "Model Building":
            with st.spinner("Building Model..."):
                model_build()
                show_code_button("coding/model_building.py")

                with open("coding/model_building.py", "r") as f:
                    code_content = f.read()

            bt = st.button("Execute the code")

            if bt:
                result = code_interpret(code_content)

                st.write("The Code Execution Result")
                st.code(result)



        elif option == "Complete ML task":
            with st.spinner("Building a End to End ML task..."):
                ml_end_to_end()
                show_code_button("coding/Ml_end_to_end_.py")

                with open("coding/Ml_end_to_end_.py", "r") as f:
                    code_content = f.read()

            bt = st.button("Execute the code")

            if bt:
                result = code_interpret(code_content)

                st.write("The Code Execution Result")
                st.code(result)





if __name__ == "__main__":
    main()
