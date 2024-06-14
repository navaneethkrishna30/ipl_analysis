#!/usr/bin/env python
# coding: utf-8

# In[ ]:

def get_ans(team1,team2,toss,venue):
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    # Load the dataset
    data = pd.read_csv(r"C:\Users\Hp\Desktop\projects\ipl_analysis\ipl_dataset\matches.csv")

    # Handle missing values (if any)
    data.dropna(subset=['winner', 'toss_winner', 'venue', 'team1', 'team2'], inplace=True)

    # Initialize the label encoders
    label_encoder_venue = LabelEncoder()
    label_encoder_team = LabelEncoder()

    # Fit the label encoder with all possible unique values from the dataset
    label_encoder_venue.fit(data['venue'])
    label_encoder_team.fit(pd.concat([data['team1'], data['team2'], data['toss_winner'], data['winner']]))

    # Encode the columns
    data['venue_encoded'] = label_encoder_venue.transform(data['venue'])
    data['team1_encoded'] = label_encoder_team.transform(data['team1'])
    data['team2_encoded'] = label_encoder_team.transform(data['team2'])
    data['toss_winner_encoded'] = label_encoder_team.transform(data['toss_winner'])
    data['winner_encoded'] = label_encoder_team.transform(data['winner'])

    # Select relevant features and target variable
    features = data[['team1_encoded', 'team2_encoded', 'toss_winner_encoded', 'venue_encoded']]
    target = data['winner_encoded']

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)


    # In[ ]:


    from sklearn.ensemble import RandomForestClassifier

    # Initialize the Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the model
    rf_model.fit(X_train, y_train)


    # In[ ]:


    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    # Make predictions on the test set
    y_pred = rf_model.predict(X_test)

    # Evaluate the model
    # accuracy = accuracy_score(y_test, y_pred)
    # report = classification_report(y_test, y_pred)
    # conf_matrix = confusion_matrix(y_test, y_pred)



    # In[ ]:


    # Function to add unseen labels to label encoders
    def add_unseen_label(encoder, new_label):
        new_classes = np.append(encoder.classes_, new_label)
        encoder.classes_ = new_classes

    # Example input
    input_team1 = team1
    input_team2 = team2
    input_toss_winner = toss
    input_venue = venue

    if input_team1 == input_team2:
            return "Invalid match setup: both teams are the same"

        # Check if the new inputs exist in the encoder classes, if not add them temporarily
    if input_venue not in label_encoder_venue.classes_:
        add_unseen_label(label_encoder_venue, input_venue)
    if input_team1 not in label_encoder_team.classes_:
        add_unseen_label(label_encoder_team, input_team1)
    if input_team2 not in label_encoder_team.classes_:
        add_unseen_label(label_encoder_team, input_team2)
    if input_toss_winner not in label_encoder_team.classes_:
        add_unseen_label(label_encoder_team, input_toss_winner)

    # Encode the input values
    input_team1_encoded = label_encoder_team.transform([input_team1])[0]
    input_team2_encoded = label_encoder_team.transform([input_team2])[0]
    input_toss_winner_encoded = label_encoder_team.transform([input_toss_winner])[0]
    input_venue_encoded = label_encoder_venue.transform([input_venue])[0]

    # Create the input feature array
    input_features = [[input_team1_encoded, input_team2_encoded, input_toss_winner_encoded, input_venue_encoded]]

    # Make the prediction
    predicted_winner_encoded = rf_model.predict(input_features)[0]
    predicted_winner = label_encoder_team.inverse_transform([predicted_winner_encoded])[0]

    return "Winner is {}".format(predicted_winner)


# In[ ]:




