#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

def get_ans(team1, team2, toss, venue):
    # Load the dataset
    data = pd.read_csv(r"ipl_dataset/matches.csv")
    
    # Handle missing values (drop rows with missing 'winner', 'team1', 'team2', etc.)
    data.dropna(subset=['winner', 'toss_winner', 'venue', 'team1', 'team2'], inplace=True)

    # Initialize LabelEncoders for categorical variables
    label_encoder_venue = LabelEncoder()
    label_encoder_team = LabelEncoder()
    label_encoder_decision = LabelEncoder()

    # Fit label encoders for the necessary columns
    label_encoder_venue.fit(data['venue'])
    label_encoder_team.fit(pd.concat([data['team1'], data['team2'], data['toss_winner'], data['winner']]))
    label_encoder_decision.fit(data['toss_decision'])

    # Encode the categorical columns
    data['venue_encoded'] = label_encoder_venue.transform(data['venue'])
    data['team1_encoded'] = label_encoder_team.transform(data['team1'])
    data['team2_encoded'] = label_encoder_team.transform(data['team2'])
    data['toss_winner_encoded'] = label_encoder_team.transform(data['toss_winner'])
    data['winner_encoded'] = label_encoder_team.transform(data['winner'])
    data['toss_decision_encoded'] = label_encoder_decision.transform(data['toss_decision'])

    # Feature engineering: Add a feature that indicates whether the toss winner is the match winner
    data['toss_win_equals_match_win'] = (data['toss_winner'] == data['winner']).astype(int)

    # Select features: Include toss decision and new feature
    features = data[['team1_encoded', 'team2_encoded', 'toss_winner_encoded', 'venue_encoded', 'toss_decision_encoded', 'toss_win_equals_match_win']]
    target = data['winner_encoded']

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Hyperparameter optimization for Random Forest
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10]
    }

    # Initialize RandomForestClassifier
    rf_model = RandomForestClassifier(random_state=42)

    # Perform GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Best parameters from GridSearchCV
    best_params = grid_search.best_params_

    # Train the model using the best parameters
    rf_model = RandomForestClassifier(**best_params, random_state=42)
    rf_model.fit(X_train, y_train)

    # Evaluate the model using cross-validation for better robustness
    cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5)

    # Evaluate on the test set
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)


    # Function to add unseen labels to label encoders
    def add_unseen_label(encoder, new_label):
        new_classes = np.append(encoder.classes_, new_label)
        encoder.classes_ = new_classes

    # Check if new inputs exist in the encoder, if not, add them
    if team1 == team2:
        return "Invalid match setup: both teams are the same"

    if venue not in label_encoder_venue.classes_:
        add_unseen_label(label_encoder_venue, venue)
    if team1 not in label_encoder_team.classes_:
        add_unseen_label(label_encoder_team, team1)
    if team2 not in label_encoder_team.classes_:
        add_unseen_label(label_encoder_team, team2)
    if toss not in label_encoder_team.classes_:
        add_unseen_label(label_encoder_team, toss)

    # Encode the input values
    input_team1_encoded = label_encoder_team.transform([team1])[0]
    input_team2_encoded = label_encoder_team.transform([team2])[0]
    input_toss_winner_encoded = label_encoder_team.transform([toss])[0]
    input_venue_encoded = label_encoder_venue.transform([venue])[0]

    # Create input features array (use default toss_decision = 0 for example)
    input_features = [[input_team1_encoded, input_team2_encoded, input_toss_winner_encoded, input_venue_encoded, 0, 0]]

    # Predict the match outcome
    predicted_winner_encoded = rf_model.predict(input_features)[0]
    predicted_winner = label_encoder_team.inverse_transform([predicted_winner_encoded])[0]

    # Print the values on the terminal
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("Classification Report:")
    print(report)
    print("Confusion Matrix:")
    print(conf_matrix)
    
    
    return f"Winner is {predicted_winner}"