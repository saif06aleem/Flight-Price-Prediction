from flask import Flask, render_template,request,session,flash
import sqlite3 as sql
import os
import pandas as pd
import random
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import plotly.express as px
import plotly.io as pio


app = Flask(__name__)


# Function to generate a random flight name
import random

def generate_flight_name():
    airlines = ['Air India', 'Spice Jet', 'Emirates', 'Vistara', 'Indigo', 'Qatar Airways']
    return random.choice(airlines)

@app.route('/')
def home():
   return render_template('index.html')

@app.route('/home1')
def home1():
   return render_template('home1.html')

@app.route('/enternew')
def new_user():
   return render_template('signup.html')

@app.route('/addrec',methods = ['POST', 'GET'])
def addrec():
    if request.method == 'POST':
        try:
            nm = request.form['Name']
            phonno = request.form['MobileNumber']
            email = request.form['email']
            unm = request.form['Username']
            passwd = request.form['password']
            with sql.connect("flight.db") as con:
                cur = con.cursor()
                cur.execute("INSERT INTO agriuser(name,phono,email,username,password)VALUES(?, ?, ?, ?,?)",(nm,phonno,email,unm,passwd))
                con.commit()
                msg = "Record successfully added"
        except:
            con.rollback()
            msg = "error in insert operation"

        finally:
            return render_template("result.html", msg=msg)
            con.close()

@app.route('/userlogin')
def user_login():
   return render_template("login.html")

@app.route('/logindetails',methods = ['POST', 'GET'])
def logindetails():
    if request.method=='POST':
            usrname=request.form['username']
            passwd = request.form['password']

            with sql.connect("flight.db") as con:
                cur = con.cursor()
                cur.execute("SELECT username,password FROM agriuser where username=? ",(usrname,))
                account = cur.fetchall()

                for row in account:
                    database_user = row[0]
                    database_password = row[1]
                    if database_user == usrname and database_password==passwd:
                        session['logged_in'] = True
                        return render_template('home1.html')
                    else:
                        flash("Invalid user credentials")
                        return render_template('login.html')

@app.route('/predictinfo')
def predictin():
   return render_template('info.html')

@app.route('/predictinfo1')
def predictin1():
   return render_template('info1.html')


@app.route('/predict',methods = ['POST', 'GET'])
def predcrop():
    if request.method == 'POST':
        import pandas as pd

        from sklearn.preprocessing import LabelEncoder
        from sklearn.tree import DecisionTreeClassifier

        # Load the dataset
        df = pd.read_csv("main_123.csv")

        # Convert categorical variables to numerical using LabelEncoder
        for col in df.columns:
            if df[col].dtype == 'object':
                l_en = LabelEncoder()
                df[col] = l_en.fit_transform(df[col])

        # Define features and target variable
        x = df[['Seat comfort', 'Inflight entertainment', 'Ease of Online booking', 'Online support', 'Gender',
                'Customer Type', 'Type of Travel', 'Class', 'On-board service', 'Leg room service', 'Online boarding']]
        y = df["satisfaction"]

        # Split the data into training and testing sets
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)

        # Train the decision tree model
        model = DecisionTreeClassifier(max_depth=17)
        model.fit(X_train, y_train)

        # Function to predict satisfaction based on user input
        def predict_satisfaction():
            # Prompt the user to enter values for each feature
            
            seat_comfort = request.form['comment1']
            inflight_entertainment = request.form['comment2']
            ease_of_online_booking = request.form['comment3']
            online_support = request.form['comment4']

            customer_type = request.form['comment6']


            on_board_service = request.form['comment9']
            leg_room_service = request.form['comment10']
            online_boarding = request.form['comment11']

            # Convert gender to numerical value
            gender = request.form['comment5']
            gender_numeric = 1 if gender == 'Female' else 0  # Convert 'Female' to 1, 'Male' to 0

            # Convert type_of_travel to numerical value
            type_of_travel = request.form['comment7']
            type_of_travel_numeric = 1 if type_of_travel == 'Business' else 0  # Convert 'Business' to 1, 'Personal' to 0

            # Convert class_type to numerical value
            class_type = request.form['comment8']
            class_type_numeric = 1 if class_type == 'Business' else 0  # Convert 'Business' to 1, 'Economy' to 0

            # Create a DataFrame with user input
            user_input = pd.DataFrame([[seat_comfort, inflight_entertainment, ease_of_online_booking, online_support, gender_numeric,
                                        customer_type, type_of_travel_numeric, class_type_numeric, on_board_service, leg_room_service,
                                        online_boarding]],
                                    columns=['Seat comfort', 'Inflight entertainment', 'Ease of Online booking',
                                            'Online support', 'Gender', 'Customer Type', 'Type of Travel', 'Class',
                                            'On-board service', 'Leg room service', 'Online boarding'])
            
            # Predict satisfaction
            satisfaction = model.predict(user_input)

            
            # Print the predicted satisfaction
            print("Predicted satisfaction:", satisfaction[0])
            satisfy = satisfaction[0]
            return satisfy

        # Call the function to predict satisfaction based on user input
        satisfy = predict_satisfaction()
        if satisfy == 0:
            print('Customer Not Satisfied')
            satisfy1 = 'Customer Not Satisfied'
        elif satisfy == 1:
            print('Customer Satisfied')
            satisfy1 = 'Customer Satisfied'

        # Generate a random flight name
        flight_name = generate_flight_name()

        return render_template('resultpred.html', prediction=satisfy1, flight_name=flight_name)


@app.route('/predict1', methods=['POST', 'GET'])
def predcrop1():
    if request.method == 'POST':
        import pandas as pd
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import LabelEncoder

        # Load the data
        data = pd.read_csv('Clean_Dataset.csv')
        df = pd.DataFrame(data)

        # Encode categorical variables
        label_encoders = {}
        for column in ['airline', 'source_city', 'departure_time', 'stops', 'arrival_time', 'destination_city', 'class']:
            label_encoders[column] = LabelEncoder()
            df[column] = label_encoders[column].fit_transform(df[column])

        # Train the model
        X = df.drop(columns=['price'])
        y = df['price']
        model = LinearRegression()
        model.fit(X, y)

        # Function to get user input and predict price
        def predict_price(airline, source_city, departure_time, stops, arrival_time, destination_city, flight_class):
            input_data = {
                'airline': [airline],

                'source_city': [source_city],
                'departure_time': [departure_time],
                'stops': [stops],
                'arrival_time': [arrival_time],
                'destination_city': [destination_city],
                'class': [flight_class]

            }

            input_df = pd.DataFrame(input_data)
            print("input_data", input_df)
            for column in label_encoders:
                print("aaaaaaa", column)
                input_df[column] = label_encoders[column].transform(input_df[column])
            predicted_price = model.predict(input_df)
            predicted_price = predicted_price[0] + 2000

            return predicted_price

        # Function to prompt user for destination input
        # Function to get destination city
        airline = request.form['comment']
        source_city = request.form['comment1']
        departure_time = request.form['comment2']
        stops = request.form['comment3']
        arrival_time = request.form['comment4']


        flight_class = request.form['comment6']

        def get_destination(source_city):
            destination = request.form['comment5']
            if destination != source_city:
                return destination
            else:
                return None

        destination_city = get_destination(source_city)
        if destination_city is None:
            return render_template('result.html',
                                   msg="Destination city cannot be the same as source city. Please enter a different destination.")

        # Get user inputs

        # Predict price
        predicted_price = predict_price(airline, source_city, departure_time, stops, arrival_time, destination_city,
                                        flight_class)
        print("Predicted Price:", predicted_price)
    return render_template('resultpred1.html', prediction1=int(predicted_price))


from flask import render_template

# Define the available chart types and colors
chart_types = ['pie', 'bar', 'line']
chart_colors = ['blue', 'green', 'red']

@app.route('/chart')
def chart():
    # Load the dataset
    df = pd.read_csv('main_123.csv')

    # Count the occurrences of each satisfaction level
    satisfaction_counts = df['satisfaction'].value_counts()

    # Create a Plotly pie chart
    fig = px.pie(satisfaction_counts, values=satisfaction_counts.values, names=satisfaction_counts.index,
                 title='Distribution of Satisfaction Levels', color_discrete_sequence=px.colors.qualitative.Set1)

    # Customize hover info
    fig.update_traces(hoverinfo='label+percent+value')

    # Update layout for a more professional look
    fig.update_layout(
        title_font=dict(size=24, family="Arial"),
        font=dict(size=14, family="Arial"),
        legend=dict(title='', orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=80, b=20),
        autosize=True
    )

    # Convert the figure to HTML
    plot_html = pio.to_html(fig, full_html=False)

    return render_template('chart.html', plot_html=plot_html, chart_types=chart_types, chart_colors=chart_colors)


@app.route("/logout")
def logout():
    session['logged_in'] = False
    return render_template('login.html')

if __name__ == '__main__':
    app.secret_key = os.urandom(12)
    app.run(debug=True)

