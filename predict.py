# %%
from flask import Flask
import pickle
from flask import request
from flask import jsonify

import numpy as np

input_file = 'model.bin'

with open(input_file, 'rb') as f_in:
    (dv, rf) = pickle.load(f_in)
# %%

app = Flask('test')


@app.route('/predict', methods=['POST'])
def predict():


	x = request.get_json()
	X = dv.transform([x])
	y_pred = rf.predict(X)
	result ={
		'Carbon dioxide emission is': float(np.exp(y_pred))
		
	}
	return jsonify(result)



if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)

# %%

