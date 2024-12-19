import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats
import plotly.express as px
import seaborn as sns 
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split
from sklearn.metrics  import f1_score,accuracy_score, precision_score, recall_score, classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import os
import joblib





        



def main():
    st.title("Chess Analysis")
    menu = ["Home"]
    
    choice = st.sidebar.selectbox("Menu", menu)
    if (choice == "Home"):
        st.subheader("Welcome to Chess Analysis Web Application")
        st.text("Upload your data to begin")
        upload()



def upload():
    dataframe = ""
    st.text('Please upload a single CSV file')

    uploaded_file = st.file_uploader('Upload file', accept_multiple_files=False, type=['csv'])

    if (uploaded_file is not None):
        dataframe = pd.read_csv(uploaded_file)
        full_expand = st.expander('View Full Data')
        full_expand.write(dataframe)
        action(dataframe)


        
def action(dataframe):
    submenu = ["Choose an option","View Data Statistics", "Perform Exploratory Data Analysis","Models Analysis and Metrics", "Predict Winner"]
    selected_option = st.selectbox("Select Task", submenu)
    
    if (selected_option == "Choose an option"):
        st.warning("Please select an appropriate option.")
    
    if (selected_option == submenu[1]):
        view_data(dataframe)
    
    elif (selected_option == submenu[2]):
        eda(dataframe)
                    
    elif (selected_option == submenu[3]):
        modelAnalysis(dataframe)
    
    elif (selected_option == submenu[4]):
        predict(dataframe)

    
    
   

def view_data(dataframe):
    st.header("Statistical View of Uploaded Data")
    st.text(f"Total number of rows in the data: {len(dataframe)}")
    st.text("Datatype associated with each column are as follows:")
    st.write(dataframe.dtypes)
    st.text("Statistics of data:")
    st.write(dataframe.describe())
    st.markdown("<h3 style='color: green; font-size: 20px'> To view first 10 rows of data, click below.</h3>", unsafe_allow_html=True)
    head_expand = st.expander("View")
    head_expand.write(dataframe.head(10))
    st.markdown("<h3 style='color: green; font-size: 20px'> To view last 10 rows of data, click below.</h3>", unsafe_allow_html=True)
    tail_expand = st.expander("View")
    tail_expand.write(dataframe.tail(10))
    

def clean_data(dataframe):
    
       # st.markdown("<h3 style='color: red;'>WARNING : Data cleaning in progress...</h3>", unsafe_allow_html=True)
        
        #Formatting column names.
        #st.write("Formatting  column names...")
        dataframe.columns = dataframe.columns.str.upper()
        dataframe.columns = dataframe.columns.str.replace('_', ' ')
        
        #Converting feature "CREATED AT" and "LAST MOVE AT" to datetime format.
       # st.write("Converting feature CREATED AT and LAST MOVE AT to datetime format.")
        dataframe['CREATED AT'] = pd.to_datetime(dataframe['CREATED AT'], unit='ms')
        dataframe['LAST MOVE AT'] = pd.to_datetime(dataframe['LAST MOVE AT'], unit='ms')    
        
        #Adding a new feature "MATCH DURATION" to store the duration of game.
        dataframe['MATCH DURATION'] = (dataframe['LAST MOVE AT'] - dataframe['CREATED AT']) / pd.Timedelta(minutes=1)
        
        #Selecting the features of interest from the list of features. 
        game_data = dataframe[["RATED", "TURNS", "VICTORY STATUS", "WINNER", "INCREMENT CODE", "WHITE ID","WHITE RATING", "BLACK ID","BLACK RATING", "MOVES", "OPENING NAME", "OPENING PLY", "MATCH DURATION"]]
        #st.text("Most Releveant Features Selected are as follows:")
        #st.write(dataframe.dtypes)
        
        #Changing the format of Increment Code.
        game_data[['INCREMENT CODE', 'INCREMENT CODE EXTENSION']] = game_data['INCREMENT CODE'].str.split('+', expand = True)
        game_data.drop(columns=['INCREMENT CODE EXTENSION'], inplace=True)
        
        #Changing the datatype of increment code to Integer.   
        game_data['INCREMENT CODE'] = game_data['INCREMENT CODE'].astype(int)   
        
        #Checking for null values in the dataset
        pd.isnull(game_data).sum()
    
        #Checking for duplicate values in the dataset and removing them.
        game_data.duplicated().sum()
        game_data = game_data.drop_duplicates()
        #st.write(dataframe.describe())
        #st.write("Number of rows remaining in the dataset after removing duplicate values:")
        #st.write(len(game_data))
        
        #Dropping games that have shorter than 2 moves.
        game_data = game_data.loc[game_data.TURNS > 3]
        
        #Converting moves to list for easy access
        game_data['MOVES'] = game_data['MOVES'].values.tolist()
        
        #Extracting the main category of opening names and saving it in a new column for better analysis.
        game_data['OPENING NAME'] = game_data['OPENING NAME'].str.strip()
        opening_category = game_data['OPENING NAME'].map(lambda x : x.split("|")[0].split(":")[0])
        game_data['OPENING CATEGORY'] = opening_category

        #Creating a new column to store the mean ratings of two players.
        game_data['MEAN RATING'] = (game_data['WHITE RATING'] + game_data['BLACK RATING'])/2
        #st.write(game_data['MEAN RATING'].describe()['min':'max'])
    
        #Dividing the game level into 4 categories based on the player's ratings : Beginner, Intermediate, Advanced and Expert.
        rating = (2500-800)/4
        levels = ['Beginner', 'Intermediate', 'Advanced', 'Expert']
        #for i in range(4):
        #    st.write(levels[i] + ' : ' + str(int(800+rating*i)) + ' to ' + str(int(800+rating*(i+1))))


        #Introducing a new feature "GAME LEVEL" and storing this information.
        def assign_level(mean_rating):
            for value in range(4):
                if mean_rating >= (800+rating*value) and mean_rating < (800+rating*(value+1)):
                    return levels[value]

        game_data['GAME LEVEL'] = game_data['MEAN RATING'].apply(assign_level)
        #st.write(game_data['GAME LEVEL'].value_counts())
        
        #Selecting only columns that have match duration greater than 0.
        game_data = game_data[game_data["MATCH DURATION"] > 0]

        #Segregating the game into four variants based on the increment time : Bullet, Blitz, Rapid and Classical.
        def assign_game_type(increment_code):
            if increment_code < 3:
                return 'Bullet'
            elif increment_code >=3 and increment_code <=10:
                return 'Blitz'
            elif increment_code >10 and increment_code <=60:
                return 'Rapid'
            else:
                return 'Classical'

        game_data['GAME VARIANT'] = game_data['INCREMENT CODE'].apply(assign_game_type)
        
        #Removing Openings That have been used just once.
        opening_name_count = game_data['OPENING NAME'].value_counts()
        least_used_opening_name = opening_name_count[opening_name_count<20]
        least_used_opening_names = least_used_opening_name.index.tolist()
        game_data = game_data[~game_data['OPENING NAME'].isin(least_used_opening_names)]
        
        #st.write(game_data)
        return game_data




def eda(dataframe):
    #upload()
    st.header('Welcome to Exploratory Data Analysis')
    game_data = clean_data(dataframe)
    
    st.markdown("<h3 style='color: green; font-size: 20px'> Please choose from options below:</h3>", unsafe_allow_html=True)
    if(st.checkbox('View Data Distribution')):
        st.write(game_data.describe())
        
    if(st.checkbox('View Histogram For Numerical Features')):
        plot_columns = game_data.drop('MATCH DURATION', axis=1)
        fig, ax = plt.subplots(figsize=(14,14))
        plot_columns.hist(ax=ax)
        st.pyplot(fig)

        
    # if(st.checkbox('View Histogram to Visualize Winner Against Each Opening Name')):
    #     fig, ax = plt.subplots(figsize=(80, 80))
    #     game_data.hist(column="WINNER", by="OPENING CATEGORY", ax=ax)
    #     st.pyplot(fig)

        
    if(st.checkbox('View Distribution of Openings')):
        st.write(game_data['OPENING NAME'].value_counts().describe())
        
    
    if(st.checkbox("View Most Frequently Used Openings")):
        st.write(game_data['OPENING NAME'].value_counts().head(10))
    
    if(st.checkbox("View Least Frequently Used Openings")):
        st.write(game_data['OPENING NAME'].value_counts().tail(10))
    
    if(st.checkbox("View Victory Status Distribution")):
        status = game_data['VICTORY STATUS'].value_counts()
        fig = px.pie(status, values=status.values, names=status.index, color_discrete_sequence=px.colors.qualitative.Light24, width = 400, height = 400)
        fig.update_layout(
            title={
                'text': "Victory Status Distribution",
                'y':0.90,
                'x':0.50,
                'xanchor': 'center',
                'yanchor': 'top'},
        )
        st.plotly_chart(fig)

        
    if(st.checkbox("View Victory Status by Color of Chess Piece")):
        fig = px.histogram(game_data, x='WINNER', color='VICTORY STATUS', title='Victory Status by Color of Chess Piece', color_discrete_sequence=px.colors.qualitative.Light24, width = 400, height = 400)
        fig.update_layout(
            yaxis={'title': 'NUMBER OF GAMES'}
        )
        st.plotly_chart(fig)

    
    
    #Plotting winner distribution by opening name
    if(st.checkbox("View Winner Distribution by Opening Name")):
        fig = px.histogram(game_data, x='OPENING NAME', color='WINNER', title='Winner by Opening Name', color_discrete_sequence=px.colors.qualitative.Light24, width = 1000, height = 1000)
        fig.update_layout(
            yaxis={'title': 'NUMBER OF GAMES'}
        )
        st.plotly_chart(fig)
    
    
    #Plotting opening name used by each type of winner
    if(st.checkbox("View Opening Name Used by Winner")):
        fig = px.histogram(game_data, x='WINNER', color='OPENING NAME', title='Opening Names used by each Winner', color_discrete_sequence=px.colors.qualitative.Light24, width = 800, height = 800)
        fig.update_layout(
            yaxis={'title': 'NUMBER OF GAMES'}    
        )
        st.plotly_chart(fig)

    #Plotting winner distribution by the variant of the game.
    if(st.checkbox("View Winner Distribution by Game Variant")):
        fig = px.histogram(game_data, x='WINNER', color='GAME VARIANT', title='Winner Distribution by Game Variant', color_discrete_sequence=px.colors.qualitative.Light24, width = 400, height = 400)
        fig.update_layout(
            yaxis={'title': 'NUMBER OF GAMES'}    
        )
        st.plotly_chart(fig)
        
    #Plotting winner distribution by the level of difficulty of the game.
    if(st.checkbox("View Winner Distribution by Game Level")):
        fig = px.histogram(game_data, x='WINNER', color='GAME LEVEL', title='Winner Distribution by Level of Difficulty', color_discrete_sequence=px.colors.qualitative.Light24, width = 400, height = 400)
        fig.update_layout(
            yaxis={'title': 'NUMBER OF GAMES'}    
        )
        st.plotly_chart(fig)
        
    #Plotting the average length of the game against each level of difficulty.
    if(st.checkbox("View Average Length of Game Against Each Difficulty Level")):
        level = game_data.groupby(game_data['GAME LEVEL']).mean()['TURNS'].sort_values().round()
        st.write(level)
        fig = px.histogram(level, x=level.index, y=level.values, histfunc='avg', labels={'y':'moves'}, color = level.values, color_discrete_sequence = px.colors.qualitative.Light24, width = 400, height = 400)
        fig.update_layout(bargap=0.1)
        fig.update_layout(
             title={
                 'text': "Average length of a game",
                  'y':0.95,
                  'x':0.50,
                  'xanchor': 'center',
                  'yanchor': 'top'},               
            yaxis={'title': 'AVERAGE NUMBER OF MOVES'}
            )
        st.plotly_chart(fig)
        
    #Top moves used in the game
    if(st.checkbox("View Top 5 Most Used Moves in the Game.")):
        st.write(game_data['MOVES'].value_counts().head(5))
        
    #First Moves
    if(st.checkbox("View Distribution of First Moves")):
        
        def fetch_moves(moves_list):
            return moves_list[:2]
  
        top_opening_move = game_data['MOVES'].apply(fetch_moves)
        st.write(top_opening_move.value_counts())
        
        
    #Scatter Plot showing Mean Rating vs Turns corresponding to the difficulty level of the game.
    if(st.checkbox("View Mean Rating vs Turns")):
        fig = px.scatter(game_data, x='TURNS', y='MEAN RATING', color='GAME LEVEL', color_discrete_sequence = px.colors.qualitative.Light24, width = 400, height = 400)
        fig.update_layout(
                title={
                    'text': "Mean Rating vs Turns",
                    'y':0.95,
                    'x':0.50,
                    'xanchor': 'center',
                    'yanchor': 'top'},  
        )
        st.plotly_chart(fig)

          

        
def modelAnalysis(dataframe):
    st.header('Welcome to Model Analysis')
    game_data = clean_data(dataframe)
    
   

    #Dropping game data that results in draw.
    draw = game_data.loc[game_data['WINNER'] == 'draw']
    game_data.drop(draw.index, inplace = True)
    
    features = ["RATED", "TURNS", "INCREMENT CODE","WHITE RATING","BLACK RATING", "OPENING CATEGORY", "MEAN RATING", "GAME LEVEL", "GAME VARIANT"]

    #Encoding Categorical Variables
    label_encoder = preprocessing.LabelEncoder() 
    encoded_labels = label_encoder.fit_transform(game_data['WINNER'])
    decoded_values = {encoded: label for encoded, label in zip(encoded_labels, game_data['WINNER'])}
    game_data['WINNER'] = label_encoder.fit_transform(game_data['WINNER'])
    #game_data['VICTORY STATUS'] = label_encoder.fit_transform(game_data['VICTORY STATUS']) 
    #game_data['OPENING NAME'] = label_encoder.fit_transform(game_data['OPENING NAME']) 
    game_data['OPENING CATEGORY'] = label_encoder.fit_transform(game_data['OPENING CATEGORY']) 
    game_data['GAME LEVEL'] = label_encoder.fit_transform(game_data['GAME LEVEL']) 
    game_data['GAME VARIANT'] = label_encoder.fit_transform(game_data['GAME VARIANT']) 
    #game_data['MOVES'] = label_encoder.fit_transform(game_data['MOVES']) 
    #st.write("Data after cleaning and encoding")
    #st.write(game_data[features])   
    
    # #Splitting the data intro train data and test data
    # X_train, X_test, y_train, y_test = train_test_split(game_data[features], game_data['WINNER'], test_size = 0.4, stratify = game_data['WINNER'])
    
    # X_train = preprocessing.normalize(X_train)
    # X_test = preprocessing.normalize(X_test)

    # X_train = preprocessing.scale(X_train)
    # X_test = preprocessing.scale(X_test)

    #Handler Functions for Evaluating Metrices
    #Calculates Accuracy, Precision, F1Score, Recall and generated Classification Report
    def calculate_metrics(actual, predicted):
        actual_decoded = list(map(lambda x: decoded_values[x], actual))
        predicted_decoded = list(map(lambda x: decoded_values[x], predicted))
        accuracy = accuracy_score(actual_decoded, predicted_decoded)
        labels = ['Black', 'White']  # Replace with your actual labels

        precision = precision_score(actual_decoded, predicted_decoded , average=None)
        f1Score = f1_score(actual, predicted)
        report = classification_report(y_true=actual_decoded, y_pred=predicted_decoded, output_dict=True)
        recall = recall_score(actual, predicted)

        st.write("Accuracy of the Model =", accuracy)
        st.write("Precision Score : ", precision)
        st.write("F1 Score =", f1Score)
        st.write("Recall =", recall)
        st.write("\nClassification Report:")
        st.write(pd.DataFrame(report).transpose())
    
        display_confusion_matrix(actual_decoded, predicted_decoded)
        
    #Creates the Confusion matrix
    def display_confusion_matrix(actual, predicted):
        conf_mat = confusion_matrix(actual, predicted)
        fig, ax = plt.subplots(figsize=(5, 5))
        matrix = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
        matrix.plot(ax=ax, cmap=plt.cm.Blues)
        plt.title('Confusion matrix')
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        st.pyplot(fig)
    
    #Training Loop  
    def train_model(x, y, model):
         model.fit(x,y)
         
    
    #Testing Loop
    def test_model(x, y, model):
        predicted = model.predict(x)
        actual = y
        calculate_metrics(actual, predicted)
        draw_AUC_Curve(actual, predicted)
    
    
    #Draws the ROC Curve
    def draw_AUC_Curve(actual, predicted):
        false_positve_rate, true_positive_rate, threshold = roc_curve(actual, predicted)
        roc_auc = auc(false_positve_rate, true_positive_rate)
        fig = plt.figure(figsize=(5, 5))
        plt.plot(false_positve_rate, true_positive_rate, color='orange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='green', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic (ROC) curve')
        plt.legend(loc="lower right")
        st.pyplot(fig)
    
    def split_data():    
    #Splitting the data intro train data and test data
        training_data_percentage = st.slider("Please vary the slider to select percentage of data to be used for training", 1, 99)
        slider_placeholder = st.empty()
        try:
            if(training_data_percentage):
                selected_train_percentage =  training_data_percentage/100
                X_train, X_test, y_train, y_test = train_test_split(game_data[features], game_data['WINNER'], train_size = selected_train_percentage, stratify = game_data['WINNER'])
    
                X_train = preprocessing.normalize(X_train)
                X_test = preprocessing.normalize(X_test)

                X_train = preprocessing.scale(X_train)
                X_test = preprocessing.scale(X_test)
                
                return X_train, X_test, y_train, y_test
                
            else:
                slider_placeholder.empty()
                
        except:
            st.error("Please choose a range below 100")

    
    X_train, X_test, y_train, y_test = split_data()
    
    models = ['Select a model','Naive Bayes', 'K-Nearest Neighbor', 'Logistic Regression', 'Decision Tree', 'Random Forest', 'Neural Network - Multi Layer Perceptron']
    model = st.selectbox("Model", models)
    
    if (model == "Select a model"):
        st.warning("Please select an appropriate model from the list.")
    
    
            
    
    #Naive Bayes     
    if(model == 'Naive Bayes'):
        model_gnb = GaussianNB()
        model_bc = BaggingClassifier(base_estimator = model_gnb, n_estimators=10)
        train_model(X_train, y_train, model_bc)
        test_model(X_test, y_test, model_bc)
        
     
    #K-Nearest Neighbor
    if(model == 'K-Nearest Neighbor'):
            param = {'n_neighbors': [5, 15, 20],
              'weights': ['uniform', 'distance'],
              'p': [1, 2]}
            #split_data()
            model_gs = GridSearchCV(KNeighborsClassifier(), param, cv=5, scoring='accuracy')
            train_model(X_train, y_train, model_gs)
            test_model(X_test, y_test, model_gs)
            
    
    #Logistic Regression
    if(model == 'Logistic Regression'):
        #split_data()
        model_lr = LogisticRegression()
        train_model(X_train, y_train, model_lr)
        test_model(X_test, y_test, model_lr)
        
    
    #Decision Tree
    if(model == 'Decision Tree'):
        param = {'criterion': ['gini', 'entropy'], 'max_depth': [
                 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20, 30, 40, 50, 70, 90, 120, 150]}
        #split_data()
        model_dtg = GridSearchCV(DecisionTreeClassifier(), param, cv=10, return_train_score=True)

        train_model(X_train, y_train, model_dtg)
        test_model(X_test, y_test, model_dtg)
        
    
    #Random Forest
    if(model == 'Random Forest'):
        #split_data()
        model_rf = RandomForestClassifier()
        train_model(X_train, y_train, model_rf)
        test_model(X_test, y_test, model_rf)
    
    
    #Neural Network - Multi Layer Perceptron
    if(model == 'Neural Network - Multi Layer Perceptron'):
        #split_data()
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)
        nn_model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 5), random_state=42, max_iter = 2000)
        train_model(X_train, y_train, nn_model)
        test_model(X_test, y_test, nn_model)
    

def predict(dataframe):
    st.header('Welcome to Model Prediction')
    game_data = clean_data(dataframe)
    
   
    def assign_level(mean_rating):
        rating = (2500-800)/4
        levels = ['Beginner', 'Intermediate', 'Advanced', 'Expert']
        for value in range(4):
            if mean_rating >= (800+rating*value) and mean_rating < (800+rating*(value+1)):
                return levels[value]
        
    def assign_game_type(increment_code):
        if increment_code < 3:
            return 'Bullet'
        elif increment_code >=3 and increment_code <=10:
            return 'Blitz'
        elif increment_code >10 and increment_code <=60:
            return 'Rapid'
        else:
            return 'Classical'
        
    #Dropping game data that results in draw.
    draw = game_data.loc[game_data['WINNER'] == 'draw']
    game_data.drop(draw.index, inplace = True)
    
    features = ["RATED", "TURNS", "INCREMENT CODE","WHITE RATING","BLACK RATING", "OPENING CATEGORY", "MEAN RATING", "GAME LEVEL", "GAME VARIANT"]

    test_input = dict()
    st.markdown('## Chess Prediction Inputs')
    # Take input for each feature
    test_input['RATED'] = eval(st.radio("Is Player Rated?", ('True', 'False')))
    test_input['TURNS'] = st.number_input('Turns')
    test_input['WHITE RATING'] = st.number_input('White Rating')
    test_input['BLACK RATING'] = st.number_input('Black Rating')
    test_input['OPENING CATEGORY'] = st.selectbox('Opening Category:', list(game_data['OPENING CATEGORY'].unique()))
    test_input['MEAN RATING'] = (test_input['WHITE RATING'] + test_input['BLACK RATING']) / 2
    test_input['INCREMENT CODE'] = st.number_input('Increment Code')
    test_input['GAME LEVEL'] = assign_level(test_input['MEAN RATING'])
    test_input['GAME VARIANT'] = assign_game_type(test_input['INCREMENT CODE'])

    # Append the test input to the game_data dataframe
    test_input_df = pd.DataFrame(test_input, columns=game_data.columns, index = [0])
    game_data = pd.concat([game_data, test_input_df], ignore_index=True)
    #Encoding Categorical Variables
    label_encoder = preprocessing.LabelEncoder() 
    # Iterate over the unique values in the original column
    encoded_labels = label_encoder.fit_transform(game_data['WINNER'])
    decoded_values = {label: encoded for label, encoded in zip(game_data['WINNER'], encoded_labels)}
    game_data['WINNER'] = label_encoder.fit_transform(game_data['WINNER'])

    game_data['OPENING CATEGORY'] = label_encoder.fit_transform(game_data['OPENING CATEGORY']) 
    game_data['GAME LEVEL'] = label_encoder.fit_transform(game_data['GAME LEVEL']) 
    game_data['GAME VARIANT'] = label_encoder.fit_transform(game_data['GAME VARIANT']) 

    #Splitting the data intro train data and test data
    X_test = game_data.iloc[-1][features]
    game_data = game_data.iloc[:-1]
    X_train, y_train = game_data[features], game_data['WINNER']
    X_train = preprocessing.normalize(X_train)
    X_train = preprocessing.scale(X_train)
    
    models = ['Naive Bayes', 'K-Nearest Neighbor', 'Logistic Regression', 'Decision Tree', 'Random Forest', 'Neural Network - Multi Layer Perceptron']
    model = st.selectbox("Model", models)
    
    def show_prediction(model, X_test):
        pred = model.predict(X_test)
        st.header('Chess Prediction Output')
        for label, encoded_label in decoded_values.items():
            if encoded_label == pred:
                st.write("The predicted winner is : ",label.capitalize())
                st.image(f"./data/{label}.png", width=50)
    
    if model == 'Naive Bayes':
        model_nb = GaussianNB()
        model_nb.fit(X_train, y_train)
        X_test = X_test.values.reshape(1, -1)
        if st.button('Predict'):
                show_prediction(model_nb, X_test)
        # st.markdown('## Chess Prediction Output')
        # for label, encoded_label in decoded_values.items():
        #     if encoded_label == pred:
        #         st.markdown(label)
     

    #K-Nearest Neighbor
    if(model == 'K-Nearest Neighbor'):
            param = {'n_neighbors': [5, 15, 20],
              'weights': ['uniform', 'distance'],
              'p': [1, 2]}
            
            model_gs = GridSearchCV(KNeighborsClassifier(), param, cv=5, scoring='accuracy')
            model_gs.fit(X_train, y_train)
            X_test = X_test.values.reshape(1, -1)
            if st.button('Predict'):
                show_prediction(model_gs, X_test)
            # st.markdown('## Chess Prediction Output')
            # for label, encoded_label in decoded_values.items():
            #     if encoded_label == pred:
            #         st.markdown(label)
            
    
    
    #Logistic Regression
    if(model == 'Logistic Regression'):
        model_lr = LogisticRegression()
        model_lr.fit(X_train, y_train)
        X_test = X_test.values.reshape(1, -1)
        if st.button('Predict'):
                show_prediction(model_lr, X_test)
                
                
    #Decision Tree
    if(model == 'Decision Tree'):
        param = {'criterion': ['gini', 'entropy'], 'max_depth': [
                 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20, 30, 40, 50, 70, 90, 120, 150]}

        model_dtg = GridSearchCV(DecisionTreeClassifier(), param, cv=10, return_train_score=True)
        model_dtg.fit(X_train, y_train)
        X_test = X_test.values.reshape(1, -1)
        if st.button('Predict'):
                show_prediction(model_dtg, X_test)
                
    
    
    #Random Forest
    if(model == 'Random Forest'):
        model_rf = RandomForestClassifier()
        model_rf.fit(X_train, y_train)
        X_test = X_test.values.reshape(1, -1)
        if st.button('Predict'):
                show_prediction(model_rf, X_test)
                
                
    #Neural Network - Multi Layer Perceptron
    if(model == 'Neural Network - Multi Layer Perceptron'):  
        scaler = StandardScaler()
        nn_model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 5), random_state=42, max_iter = 2000)
        nn_model.fit(X_train, y_train)
        X_test = X_test.values.reshape(1, -1)
        if st.button('Predict'):
                show_prediction(nn_model, X_test)
                

if __name__ == '__main__':
    main()

                
                    
            
    
        
    
