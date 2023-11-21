# importing the libraries
import numpy as np
import pickle

from flask import Flask, request, render_template


# create a app object using the Flask class
app = Flask(__name__)

#Load the trained model. (Pickle file)
plin_model = pickle.load(open('model.pkl', 'rb'))
plin_model

#Define the route to be home. 
#The decorator below links the relative route of the URL to the function it is decorating.
#Here, home function is with '/', our root directory. 
#Running the app sends us to index.html.
#Note that render_template means it looks for the file in the templates folder. 

#use the route() decorator to tell Flask what URL should trigger our function.
@app.route('/')
def home():
  return render_template('index.html')

#Used the methods argument of the route() decorator to handle different HTTP methods.
#GET: A GET message is send, and the server returns data
#POST: Used to send HTML form data to the server.
#Add Post method to the decorator to allow for form submission. 
#Redirected to /predict page with the output
@app.route('/predict',methods=['POST'])
def predict():

    int_features = [float(x) for x in request.form.values()] #Converting string inputs to float.
    features = [np.array(int_features)]  #Converting to the form [[a, b]] for input to the model
    prediction = plin_model.predict(features)  # features Must be in the form [[a, b]]

    output = round(prediction[0], 2) # rounding to the 2nd decimal

    return render_template('index.html', prediction_text='Percent with heart disease is {}'.format(output))


if __name__ == "__main__":
    app.run(port=5000)