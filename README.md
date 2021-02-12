## HINT3: Raising the bar for Intent Detection in the Wild

This repository contains datasets and code for the paper
 "HINT3: Raising the bar for Intent Detection in the Wild" 
 accepted at EMNLP-2020's
  [Insights workshop](https://insights-workshop.github.io/)
  
Preprint for the paper is available [here](https://arxiv.org/abs/2009.13833)

**Update Feb 2021: We noticed in our analysis of the results that
 there are few ground truth labels which are incorrect. Hence, we're releasing 
 a new version, v2 of the dataset, present inside dataset/v2 folder. All the
  results in the paper were obtained on the earlier version of the dataset 
  present inside dataset/v1, which should be used to exactly reproduce
   the results presented in the paper.**


### Dataset

- Train and Test sets for SOFMattress, Curekart and Powerplay11
 are available in `dataset` folder for both Full and Subset variations.
- You can also use `prepare_subset_of_data.ipynb` notebook to generate
 subset variations of full datasets.  All the entailment assets
  generated can be downloaded from [here](https://drive.google.com/drive/folders/1Un97REmtSbxmcNlDgg5qX0awxFoGz0n4?usp=sharing).


### EDA

We have done EDA analysis on the datasets which is accessible
 from the `data_exploration` folder.


### Test set predictions

Predictions from BERT and 4 NLU platforms on test sets used for
 analysis in the paper are present in `preds` folder. Feel free to
  do further analysis on these predictions if you want.

### Test set metrics

All the metrics from BERT and 4 NLU platforms on test sets
 are present in `results` folder for further analysis. Graphs plotted in
  the paper can be reproduced using `analysis/plot_metrics_graph.ipynb` 
  notebook 


### Reproducibility Instructions

The scripts to generate training data and predicting intents
 based on the testing data for all the 4 platforms and BERT 
 based classifier are inside `platforms` folder within
  their named directories.


#### Rasa

- The `training_data_conversion.ipynb` notebook is used to
 convert the training set into a JSON format that Rasa
  mandates in order to train its model. The generated JSON
   file is created inside the `data` directory

- In order to train a model for one particular bot, keep only
 that bot's JSON file inside the `data` directory

- Train the model using this command: `rasa train nlu`

- Once the model is trained, its tar.gz file will be stored
 inside the `models` directory based on the current timestamp

- In order to start the NLU server, run the following command:
 `rasa run --enable-api -m models/nlu-<timestamp>.tar.gz` where
  `nlu-<timestamp>.tar.gz` is the name of the model's file
   created in the previous step

- In order to generate a report against a testing set file,
 run the `generate_preds.ipynb` notebook after specifying the
  name of the bot. Generated predictions will be stored inside
   `preds` folder



#### Dialogflow
- The `training_data_conversion.ipynb` file is used to convert
 the training set into a bunch of JSON files that Dialogflow
  mandates in order to train its model. The generated JSON files
   are stored inside the `intents` directory

- Login to the Diaologflow dashboard using a Gmail account
 and visit `https://dialogflow.cloud.google.com`

- Dialogflow allows bulk upload of the training set by
 importing a zip file. The compressed folder has a predefined
  structure. In order to create this folder, create a copy of
   the `agent_template` directory and rename the folder as
    per your bot name. Then, copy all the JSON files created
     in step 1 and paste it inside the `intents` folder of your
      agent directory. Then, open the `agent.json` file and edit
       the `displayName` property to specify the name of the
        agent of your bot. An agent is analogous to an app or
         a bot. Once these changes are done, compress the agent
          directory into a zip file

- Create a new agent on the Dialogflow dashboard
 here: `https://dialogflow.cloud.google.com/?authuser=1#/newAgent`

- Delete `Default Fallback Intent` from the intents dashboard

- Edit the agent: `https://dialogflow.cloud.google.com/?authuser=1#/editAgent/mt11-agent-ugmx/` -> Export & Import -> Import from zip -> upload the agent zip file. This will allow us to bulk upload all intents along with their respective utterances

- Go to Edit agent -> ML settings. The default threshold value
 is 0.3. Change it to 0.05 and Train the model

- Copy the CURL request from the API playground. We can get
 the authentication token and the model's API endpoint from
  this CURL request

- The `generate_preds.ipynb` file will help generate predictions
 for the bot.


#### LUIS
- The `training_data_conversion.ipynb` file will generate
 a JSON file based on the training set's CSV file

- Login to `luis.ai`, go to `https://www.luis.ai/applications`
 and click on `New app for conversation` -> `Import as JSON`.
  Upload the JSON file generated in the first step

- Once all the intents are uploaded, click on the `Train` button
 to train the model. Once the model is trained, click on
  `Publish` followed by selecting `Production slot`

- Now, go to the `Manage` section of the app and copy the
 App ID. We will be using this App ID in the `generate_preds.ipynb` file to generate our prediction reports

- Go to the settings page of your account in order to get
 the `PREDICTION_KEY` and `PREDICTION_ENDPOINT` used in
  `generate_preds.ipynb` file

#### Haptik

- Access requests for signup on Haptik are processed via contact
 form at https://haptik.ai/contact-us/

- Once you get the access, you'll be able to create bots
 and run predictions using the scripts provided in
  `platforms/haptik`


#### BERT

- Results on BERT can be reproduced using scripts in the
 folder `platforms/bert`

- The folder also contains config for each of the models
 trained on Full and Subset variations of datasets