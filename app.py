from flask import Flask,render_template,request
import joblib
from helpers.dummies import *

app=Flask(__name__)

model=joblib.load('models/model.h5')
scaler=joblib.load('models/scaling.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['GET', 'POST'])
def predict():
    ##all_data=request.args
    if request.method == 'POST' :
        Iron_Feed=float(request.form['Iron_Feed'])
        Silica_Feed=float(request.form['Silica_Feed'])
        Starch_Flow=float(request.form['Starch_Flow'])
        Amina_Flow =float(request.form['Amina_Flow'])
        Ore_Pulp_Flow =float(request.form['Ore_Pulp_Flow'])
        Ore_Pulp_pH =float(request.form['Ore_Pulp_pH'])
        Ore_Pulp_Density =float(request.form['Ore_Pulp_Density'])
        Flotation_Column_01_Air_Flow =float(request.form['Flotation_Column_01_Air_Flow'])
        Flotation_Column_02_Air_Flow =float(request.form['Flotation_Column_02_Air_Flow'])
        Flotation_Column_03_Air_Flow =float(request.form['Flotation_Column_03_Air_Flow'])
        Flotation_Column_04_Air_Flow =float(request.form['Flotation_Column_04_Air_Flow'])
        Flotation_Column_05_Air_Flow =float(request.form['Flotation_Column_05_Air_Flow'])
        Flotation_Column_06_Air_Flow =float(request.form['Flotation_Column_06_Air_Flow'])
        Flotation_Column_07_Air_Flow =float(request.form['Flotation_Column_07_Air_Flow'])
        Flotation_Column_01_Level =float(request.form['Flotation_Column_01_Level'])
        Flotation_Column_02_Level =float(request.form['Flotation_Column_02_Level'])
        Flotation_Column_03_Level =float(request.form['Flotation_Column_03_Level'])
        Flotation_Column_04_Level =float(request.form['Flotation_Column_04_Level'])
        Flotation_Column_05_Level =float(request.form['Flotation_Column_05_Level'])
        Flotation_Column_06_Level =float(request.form['Flotation_Column_06_Level'])
        Flotation_Column_07_Level =float(request.form['Flotation_Column_07_Level'])
        Month=Month_Name_dummies[request.form['Month']] 
        Day=Day_Name_dummies[request.form['Day']]
        Hour=Hour_dummies[request.form['Hour']]
        data=[Iron_Feed,Silica_Feed,Starch_Flow,Amina_Flow,Ore_Pulp_Flow, Ore_Pulp_pH,Ore_Pulp_Density,Flotation_Column_01_Air_Flow,Flotation_Column_02_Air_Flow ,Flotation_Column_03_Air_Flow ,Flotation_Column_04_Air_Flow , Flotation_Column_05_Air_Flow ,Flotation_Column_06_Air_Flow ,Flotation_Column_07_Air_Flow ,Flotation_Column_01_Level ,Flotation_Column_02_Level , Flotation_Column_03_Level ,Flotation_Column_04_Level ,Flotation_Column_05_Level ,Flotation_Column_06_Level , Flotation_Column_07_Level ]+ Month+Day+Hour

        data_scaled=scaler.transform([data])
        pred=model.predict(data_scaled)[0]

        return render_template('prediction.html',profit=pred)
    # return "project"


if __name__=='__main__':
    app.run(debug = True)