

# Introduction

**Named Entity Recognition (NER)** is one of the important tasks in the natural language processing, which are developed with the aim of recognize named entities within different texts and classify to categories such as Person, Organization, Location, etc. .

One of the related challenges, is the develop of NLP algorithms such as NER for languages that do not have a rich and diverse dataset. Therefore, in the proposed method, multi-task learning technique has been used to overcome this issue and provide more accurate results about **NER**, **Part-of-speech(POS)** and **Ezafe** in persian language.


# Usage
Project running tips have been explained step-by-step that you can follow them to run the code and deploy trained model as web API using TensorFlow Serving.

## Step-1
in first step you must clone the project on your system as follow :

```git clone https://github.com/mnouri92/persian_ner.git```
> if you dont have the git on your system you can download and install that from [Git](https://git-scm.com/downloads)

## Step-2
The project is developed by **Python** programming language and using some important packages related to Machine-Learning programming such as TensorFlow. 
> if you haven't installed Python on your system you can download and install from [Python.org](https://www.python.org/downloads/)

Therefore after cloning the project, move to project's root directory and install requirements by :

``` python -m pip install -r requirements.txt```

## Step-3
Now is turn to train the models
> For train the models you need training data related to each of tasks and word embedding for Persian vocabularies that all of these will be downloaded automatically by running the train_mtl.sh file

For train the model all that you have to do is just specify the chosen model to train (mtl2, mtl3 or mtl4), main task and auxiliary task(s) and then run the following command in terminal : 
``` ./train_mtl.sh <chosen_type> <main_task_path> <auxiliary_task_path> <second_aux_path_if_type_is_mtl3> <third_aux_path_if_type_is_mtl4>```

> type can be chosen from :
> [ mtl2 , mtl3 , mtl4]
> 
> each of paths can be chosen from :
> [ files/mtl/ner_bijankhan , files/mtl/ner_armanperso , files/mtl/gen , files/mtl/pos ]

After that model training was ended you can evaluate the model by run the bellow code in terminal :
``` ./evaluate_mtl.sh <model_type> <trained_model_path> <input_data_path> <output_data_path> ```

## Step-4
After running the code for each of tasks , the trained model will be saved in **./model** sub-directory related to chosen task as main.
The trained model that saved on model sub-directory is in **checkpoints** format that you will need **SavedModel** format of that to deploy as web API using TensorFlow Serving module.
The **export_saved_model.sh** file has been provided for this reason that you can use that as follow to export the SavedModel format of checkpoints saved model.

``` ./export_saved_model.sh <type> <task_path> <model_version_number>```

> type is one of the mtl2 , mtl3 or mtl4
> task_path is the path of task that has been chosen as main task in training step. for example could be files/mtl/gen if gen task has been chosen as main task.

## Step-5
Now the SavedModel format of trained model is ready in **./saved_model/{version_number}** sub-directory of chosen model.

**TensorFlow Serving** is one of the useful module from TFX that is used to serve the SavedModel as a web API by implements the TF Server on the host system and running the web API services on it. you can use this module directly but simpler way is using TensorFlow/Serving image of **Docker** that let you to implement the TensorFlow server as an **Docker Container** on the host system.

> If you haven't installed Docker in your system you can download and install it from [Docker](https://www.docker.com/)

You can run the web API service by trained model as Docker Container using the fhe following command terminal :

``` docker run -p 8501:8501 -p 8500:8500 --name <Container_name> --mount type=bind,source=<path_of_SavedModel>,target=/models/<model_name> -e MODEL_NAME=<model_name> -t tensorflow/serving & ```

The above code will implement the TF Server as a Docker container on the host system and then run the web API service on TF Server. Finally map the port of container to the system's port (the 8501 port are used to communicate with the web API by RestFul standard and 8500 are used to communicate by gRPC standard).

After running the above code the implemented web API can be accessed via the **http://localhost:8501/v1/models/model_name:predict**

