## Project info 
TODO



## Data And Variable


**Data and variable source**:  [Kaggle](https://www.kaggle.com/debajyotipodder/co2-emission-by-vehicles)



## File Descriptions

* `notebook.ipynb`  Notebook contains data  preparation, EDA, feature importance, parameter tuning and  model selection.
* `train.py` Python file contains best model and saving model file. (<i>Exported script</i>)
* `predict.py` Python file contains model file and serving as Flask app.
* `predict_test.py` Python file contains one observation for probility and CHD risk result. **(local solution)**
* `cloud_predict.py ` Python file contains model for one observation and includes cloud endpoint. **(cloud solution)**
* `requirements.txt` Txt file contains all dependencies  for notebook.ipynb and predictions scripts. 


## Preparing Python Environments





```bash
git clone https://github.com/yusyel/ML-bookcamp-capstone.git
```


```bash
cd ML-bookcamp-capstone
```

> Activate python environments
```bash
pipenv shell
```
> In python environment installing python dependency:

```bash
pip install -r requirements.txt
```
## Preparing And Running Docker Image


> For building docker image:
```bash
docker build -t capstone .
```
> After building docker image you can run docker image with this command:

```bash
docker run -it --rm -p 9696:9696 capstone 
```

## Runing Predictions File

> While docker container running In your python shell:

```bash
python3 predict_test.py
```
*  TODO

> Cloud test. `cloud_predict.py` contains server endpoint.

```bash
python3 cloud_predict.py
```

* TODO
