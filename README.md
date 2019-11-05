# CS 325B - Air Quality

## The Problem
The Environmental Protection Agency (EPA) employs monitoring stations throughout the United States that keep tabs on the air quality in the surrounding area. Concentrations of pollutants such as CO, NO<sub>2</sub>, and SO<sub>2</sub> are monitored daily by these stations. In particular, the pollutant PM<sub>2.5</sub> has recently become a focal point of the research community for its concerning relationship with mortality rates. Unfortunately, this network of EPA monitoring stations is sparse, leaving much of the United States without a nearby air quality monitoring station. This leaves civilians unaware of potentially high PM<sub>2.5</sub> concentrations in their areas of residence or work, and inhibits researchers from gaining a country-wide understanding of PM<sub>2.5</sub> and its fatal properties.

## The Solution
Our task is to "fill in the gaps" for this sparse EPA monitoring network. Specifically, can we use remote sensing data (such as satellite imagery and spectral measurements) and weather data to accurately predict PM<sub>2.5</sub> concentrations, regardless of the presence of an EPA station? This repository stores our data processing, model training, and evaluation pipeline for leveraging machine learning models to achieve this goal.
