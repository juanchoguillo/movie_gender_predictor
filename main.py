from flask import Flask, request, jsonify, render_template
from flask_restx import Api, Resource, fields
import joblib
import numpy as np 
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


app = Flask(__name__)


# Definición API Flask
api = Api(
    app, 
    version='1.0', 
    title='Gender Movie Predictor',
    description='Gender Movie Predictor Api')

ns = api.namespace('predict', 
     description='Gender Movie Predictor')

# Definición argumentos o parámetros de la API
parser = api.parser()
parser.add_argument(
    'PLOT', 
    type=str, 
    required=True, 
    help='Plot Of the movie', 
    location='args')

resource_fields = api.model('Resource', {
    'result': fields.String,
})

# Definición de la clase para disponibilización
@ns.route('/')
class CarPredictionApi(Resource):

    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    
    def get(self):
        args = parser.parse_args()
        
        return {
         "result": predict_movie_gender(args['PLOT'])
        }, 200

def predict_movie_gender(plot):
    cols = ['p_Action', 'p_Adventure', 'p_Animation', 'p_Biography', 'p_Comedy', 'p_Crime', 'p_Documentary', 'p_Drama', 'p_Family',
        'p_Fantasy', 'p_Film-Noir', 'p_History', 'p_Horror', 'p_Music', 'p_Musical', 'p_Mystery', 'p_News', 'p_Romance',
        'p_Sci-Fi', 'p_Short', 'p_Sport', 'p_Thriller', 'p_War', 'p_Western']
    
    # Transformacion del plot 
    vect = joblib.load('vectorizer.joblib')
    plot_dtm =  vect.transform([plot])

    # cargar modelo 
    model_in = joblib.load('model_proyecto_2.joblib')

    # predicion del genero usando el modelo 
    pred_genre = model_in.predict_proba(plot_dtm)

    # Hacer la prediccion del genero segun la propbabilidad mas alta 
    predicted_genre = cols[np.argmax(pred_genre)]

    return predicted_genre


if __name__ == '__main__':
    # app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
    app.run(debug=True, host='54.145.223.240', port=5000)