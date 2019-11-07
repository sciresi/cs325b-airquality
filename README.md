# CS 325B - Air Quality

## The Problem
The Environmental Protection Agency (EPA) employs monitoring stations throughout the United States that keep tabs on the air quality in the surrounding area. Concentrations of pollutants such as CO, NO<sub>2</sub>, and SO<sub>2</sub> are monitored daily by these stations. In particular, the pollutant PM<sub>2.5</sub> has recently become a focal point of the research community for its concerning relationship with mortality rates. Unfortunately, this network of EPA monitoring stations is sparse, leaving much of the United States without a nearby air quality monitoring station. This leaves civilians unaware of potentially high PM<sub>2.5</sub> concentrations in their areas of residence or work, and inhibits researchers from gaining a country-wide understanding of PM<sub>2.5</sub> and its fatal properties.

## The Solution
Our task is to "fill in the gaps" for this sparse EPA monitoring network. Specifically, can we use remote sensing data (such as satellite imagery and spectral measurements) and weather data to accurately predict PM<sub>2.5</sub> concentrations, regardless of the presence of an EPA station? This repository stores our data processing, model training, and evaluation pipeline for leveraging machine learning models to achieve this goal.

## Model Training
Our training pipeline uses `.yaml` files to keep track of model and training configurations (model hyperparameters, number of epochs to train, etc.). A template for such configurations can be found in `yamlTemplate.yaml` in the main folder. This file stores the possible configuration names as well as a comment for their purpose. It's recommended that you take the following steps for each experiment that you run:

1. Create a new folder for the next experiment you are going to run (for example, `CNNWeightDecayTuning`). It's recommended that you house such folders in a larger folder called `Experiments`.
2. Copy the `yamlTemplate.yaml` file into this new experiment folder, and give it the same name as your folder name.
3. Fill in the configuration file with the appropriate values based on the experiment you want to run. Feel free to delete the comments once you have filled in the values, as well as delete any optional configuration entries that you don't plan to use for this experiment.
4. To train, simply run `python {PATH_TO_REPO}/experiment.py --yaml-file={NAME_OF_FILE}.yaml` from your experiment folder.
5. Following training, results such as the trained model parameters, loss graph, metrics, and other important artifacts will be saved in the folder you specified in the `.yaml` file (highly recommended to be the same folder housing your `.yaml` file!).
