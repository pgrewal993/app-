# Import necessary libraries
import streamlit as st
import os
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from gradientai import Gradient

# Load the dataset with a specified encoding
data = pd.read_csv('Final_Clean_Data.csv', encoding='latin1')

# Page 1: Dashboard
def dashboard():
    st.image('images.png', use_column_width=True)

    st.subheader("💡 Abstract: ")

    inspiration = '''
    The Edmonton Food Drive project intends to use machine learning to optimise drop-off and pick-up operations in order to revolutionise food donation management in Edmonton, Alberta. Current challenges in coordination and route planning necessitate automation for timely collections and reduced logistical complexities. Creating a machine learning model to determine the best drop-off sites, automating pick-up routes, and enhancing stakeholder communication are among the goals. The strategy include developing route planning algorithms, putting in place a centralised system for stakeholder cooperation, and evaluating historical data to determine the best sites. Lessons learnt include enhanced efficiency, improved resource utilization insights, and the value of real-time data in refining donation processes.
    '''

    st.write(inspiration)

    st.subheader("👨🏻‍💻 What our Project Does?")

    what_it_does = '''
    We started a thorough data collecting phase utilising Google Forms for our food drive project, which was launched on September 23rd. Manual collection and Excel tools were also used for data collection. This project was made possible by the cooperative efforts of student organisations and Edmonton city volunteers. After that, in order to improve the quality and consistency of our dataset, we carefully cleaned the data, eliminating duplicates and standardizing formats to enhance the quality and reliability of our dataset.
    Our machine learning goal is to solve a regression issue by predicting the number of donation bags that will be collected. This helps with the efficient resource utilization and allocation of volunteers. We created a user-friendly dashboard with tools for Exploratory Data Analysis (EDA), ML Modelling, Neighbourhood Mapping, Data Collection,  data visualizationas well as a chatbot for communicating with stakeholders. Among them the ML Modelling part helps  to predict number of donation bags in each stake, the EDA part offers insightful visualisations. A link to an updated Google Form is provided for volunteers in the Data Collection section, and the Neighbourhood Mapping option creates maps based on user inputs. The Chatbot enables easy communication among volunteers.
    '''

    st.write(what_it_does)




# Page 2: Exploratory Data Analysis (EDA)
def exploratory_data_analysis():
    st.title("Exploratory Data Analysis")


    # Visualize the distribution of numerical features using Plotly
    fig = px.histogram(data, x='# of Adult Volunteers', nbins=20, labels={'# of Adult Volunteers': 'Adult Volunteers'})
    st.plotly_chart(fig)

    fig = px.histogram(data, x='# of Youth Volunteers', nbins=20, labels={'# of Youth Volunteers': 'Youth Volunteers'})
    st.plotly_chart(fig)

    fig = px.histogram(data, x='Donation Bags Collected', nbins=20, labels={'Donation Bags Collected': 'Donation Bags Collected'})
    st.plotly_chart(fig)

    fig = px.histogram(data, x='Time to Complete (min)', nbins=20, labels={'Time to Complete (min)': 'Time to Complete'})
    st.plotly_chart(fig)

# Page 3: Machine Learning Modeling
def machine_learning_modeling():
    st.title("Machine Learning Modeling")
    st.write("Enter the details to predict donation bags:")

    # Input fields for user to enter data
    completed_routes = st.slider("Completed More Than One Route", 0, 1, 0)
    routes_completed = st.slider("Routes Completed", 1, 10, 5)
    time_spent = st.slider("Time Spent (minutes)", 10, 300, 60)
    adult_volunteers = st.slider("Number of Adult Volunteers", 1, 50, 10)
    doors_in_route = st.slider("Number of Doors in Route", 10, 500, 100)
    youth_volunteers = st.slider("Number of Youth Volunteers", 1, 50, 10)

    # Predict button
    if st.button("Predict"):
        # Load the trained model
        model = joblib.load('knn_regressor_model_fd.pkl')

        # Prepare input data for prediction
        input_data = [[completed_routes, routes_completed, time_spent, adult_volunteers, doors_in_route, youth_volunteers]]

        # Make prediction
        prediction = np.round(model.predict(input_data))

        # Display the prediction
        st.success(f"Predicted Donation Bags: {prediction[0]}")

        # You can add additional information or actions based on the prediction if needed
# Page 4: Neighbourhood Mapping
# Read geospatial data
geodata = pd.read_csv("ADDRESS ONLY Property_Assessment_Data__Current_Calendar_Year_ - Property_Assessment_Data__Current_Calendar_Year_.csv")

def neighbourhood_mapping():
    st.title("Neighbourhood Mapping")

    # Get user input for neighborhood
    user_neighbourhood = st.text_input("Enter the neighborhood:")

    # Check if user provided input
    if user_neighbourhood:
        # Filter the dataset based on the user input
        filtered_data = geodata[geodata['Neighbourhood'] == user_neighbourhood]

        # Check if the filtered data is empty, if so, return a message indicating no data found
        if filtered_data.empty:
            st.write("No data found for the specified neighborhood.")
        else:
            # Create the map using the filtered data
            fig = px.scatter_mapbox(filtered_data,
                                    lat='Latitude',
                                    lon='Longitude',
                                    hover_name='Neighbourhood',
                                    zoom=12)

            # Update map layout to use OpenStreetMap style
            fig.update_layout(mapbox_style='open-street-map')

            # Show the map
            st.plotly_chart(fig)
    else:
        st.write("Please enter a neighborhood to generate the map.")


# Page 5: Data Collection
def data_collection():
    st.title("Data Collection")
    st.write("Please fill out the Google form to contribute to our Food Drive!")
    google_form_url = "https://forms.gle/NQX4z9WwqhJaeUFV8"#YOUR_GOOGLE_FORM_URL_HERE
    st.markdown(f"[Fill out the form]({google_form_url})")

# Page 6: Chatbot
def chatbot():
    st.title("Interactive Food Drive Assistant")
    st.write("Ask a question about the Food Drive!")

    with Gradient() as gradient:
        base_model = gradient.get_base_model(base_model_slug="nous-hermes2")
        new_model_adapter = base_model.create_model_adapter(name="interactive_food_drive_model")

        user_input = st.text_input("Ask your question:")
        if user_input and user_input.lower() not in ['quit', 'exit']:
            sample_query = f"### Instruction: {user_input} \n\n### Response:"
            st.markdown(f"Asking: {sample_query}")

            # before fine-tuning
            completion = new_model_adapter.complete(query=sample_query, max_generated_token_count=100).generated_output
            print(f"Generated (before fine-tune): {completion}")

            completion = new_model_adapter.complete(query=sample_query, max_generated_token_count=100).generated_output
            st.markdown(f"Generated: {completion}")

        # Delete the model adapter after generating the response
        new_model_adapter.delete()

# Main App Logic
def main():
    st.sidebar.title("Food Drive App")
    app_page = st.sidebar.radio("Select a Page", ["Dashboard", "EDA", "ML Modeling", "Neighbourhood Mapping", "Data Collection", "Chatbot"])

    if app_page == "Dashboard":
        dashboard()
    elif app_page == "EDA":
        exploratory_data_analysis()
    elif app_page == "ML Modeling":
        machine_learning_modeling()
    elif app_page == "Neighbourhood Mapping":
        neighbourhood_mapping()
    elif app_page == "Data Collection":
        data_collection()
    elif app_page == "Chatbot":
        chatbot()

if __name__ == "__main__":
    main()
