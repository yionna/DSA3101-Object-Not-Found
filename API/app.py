from flask import Flask, request, render_template
import pandas as pd
import os

# Initialize the Flask application
app = Flask(__name__)

path = '../data/predictions'
files = [f for f in os.listdir(path) if 'csv' in f]
data = pd.read_csv(os.path.join(path, files[0]))
for file in files[1:]:
    temp =  pd.read_csv(os.path.join(path, file))
    data = data.merge(temp, on='CLIENTNUM', how='inner')


@app.route('/')
def index():   
    return render_template('index.html', clientnums=data['CLIENTNUM'])
 

@app.route('/customer_information', methods=['GET', 'POST'])
def retrieve_information():
    if request.method == 'POST':
        clientnum = request.form.get('clientnum')
        client_data = data[data['CLIENTNUM'] == int(clientnum)]
        information = {
            "CLIENTNUM": clientnum,
            "headers": list(data.columns),
            "client_data": client_data.to_dict(orient='records')[0]
        }
        return render_template('customer_information.html', information=information, clientnums=data['CLIENTNUM'])
    elif request.method == 'GET':
        return render_template('customer_information.html', clientnums=data['CLIENTNUM'])
    

@app.route('/display', methods=['GET'])
def display():
    return render_template('display.html')

demographic_df = pd.read_csv('demographic.csv')

@app.get("/get_campaign/{clientnum}")
async def get_campaign(clientnum: int):
    # Extract client information
    client_data = demographic_df[demographic_df['CLIENTNUM'] == clientnum]
    if client_data.empty:
        return {"error": "Client not found"}

    campaign = {
        "ChosenTiming": int(client_data['ChosenTiming'].values[0]),
        "ChosenFrequency": int(client_data['ChosenFrequency'].values[0]),
        "ChosenChannel": client_data['ChosenChannel'].values[0]
    }
    return campaign

if __name__ == '__main__':
    app.run(debug=True)