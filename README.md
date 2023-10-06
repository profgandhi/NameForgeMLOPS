# Name Generator with PyTorch

## Overview

This Python project utilizes PyTorch to train a deep learning model for generating names. Whether you need creative character names for your story or want to come up with unique usernames, this name generator can help you create novel and interesting names. This is made with the understanding of Karapathy's Makemore series and build llama-from-scratch blog. Custom names dataset scraped for indian names.

## Features

- Generate names based on patterns learned from a dataset.
- Customizable model architecture and training parameters.
- Save and load trained models for future use.
- Simple command-line interface for generating names.


## Docker
 - Created Dockerfile with ubuntu base image
 - Downloaded python, pip and packages using RUN
 - Transfer files and models from local directory using COPY
 - Running inference_pipeline.py using CMD 
 - Build image with name polo123/namesforge and ran image successfully
 - Pushed to Docker Hub


## Unit tests
 - pytest test in tests folder
 - Added tests for tokenizer
 - Added test for build_dataset 





