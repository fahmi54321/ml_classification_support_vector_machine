from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# LOAD MODEL & SCALER
classifier = joblib.load('support_vector_machine.pkl')
scaler = joblib.load('scaler.pkl')

# LOAD DATASET
dataset = pd.read_csv('Social_Network_Ads.csv')

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# ORIGINAL DATA
x_original = x
y_original = y

# HOME
@app.route('/')
def home():

    return jsonify({

        'success': True,

        'message':
            'Support Vector Machine API Running'
    })

# PREDICT
@app.route('/predict', methods=['POST'])
def predict():

    try:

        data = request.get_json()

        age = float(data['age'])

        estimated_salary = float(
            data['estimated_salary']
        )

        # ORIGINAL INPUT
        original_input = [[
            age,
            estimated_salary
        ]]

        # SCALE INPUT
        scaled_input = scaler.transform(
            original_input
        )

        # PREDICTION
        prediction = classifier.predict(
            scaled_input
        )[0]

        # LABEL
        label = (

            'BUY_SUV'

            if prediction == 1

            else 'NOT_BUY_SUV'
        )

        # DESCRIPTION
        description = (

            'Customer diprediksi akan membeli SUV'

            if prediction == 1

            else
            'Customer diprediksi tidak membeli SUV'
        )

        # DECISION FUNCTION
        decision_score = float(

            classifier.decision_function(
                scaled_input
            )[0]
        )

        # RESPONSE
        response = {

            'success': True,

            'model':
                'Support Vector Machine',

            'kernel':
                'linear',

            'prediction':
                int(prediction),

            'label':
                label,

            'description':
                description,

            'input': {

                'age':
                    age,

                'estimated_salary':
                    estimated_salary
            },

            'scaled_input': {

                'scaled_age':
                    float(scaled_input[0][0]),

                'scaled_estimated_salary':
                    float(scaled_input[0][1])
            },

            'decision_score':
                decision_score,

            'visualization_info': {

                'x_axis':
                    'Age',

                'y_axis':
                    'Estimated Salary',

                'blue_region':
                    'BUY SUV',

                'salmon_region':
                    'NOT BUY SUV',

                'decision_boundary':
                    'Maximum Margin Hyperplane'
            }
        }

        return jsonify(response)

    except Exception as e:

        return jsonify({

            'success': False,

            'error': str(e)

        }), 500

# PLOT DATA
@app.route('/plot-data', methods=['GET'])
def plot_data():

    try:

        # ORIGINAL DATA
        x_set = x_original
        y_set = y_original

        # CREATE GRID
        x1, x2 = np.meshgrid(

            np.arange(

                start=x_set[:, 0].min() - 10,

                stop=x_set[:, 0].max() + 10,

                step=1
            ),

            np.arange(

                start=x_set[:, 1].min() - 1000,

                stop=x_set[:, 1].max() + 1000,

                step=1000
            )
        )

        # GRID POINTS
        grid_points = np.array([

            x1.ravel(),

            x2.ravel()

        ]).T

        # SCALE GRID
        scaled_grid = scaler.transform(
            grid_points
        )

        # PREDICT GRID
        predictions = classifier.predict(
            scaled_grid
        )

        # SUPPORT VECTORS
        support_vectors = scaler.inverse_transform(
            classifier.support_vectors_
        )

        # PREDICTION REGIONS
        prediction_regions = []

        for i in range(len(grid_points)):

            prediction_regions.append({

                'age':
                    float(grid_points[i][0]),

                'estimated_salary':
                    float(grid_points[i][1]),

                'prediction':
                    int(predictions[i]),

                'region_color': (

                    'dodgerblue'

                    if predictions[i] == 1

                    else 'salmon'
                )
            })

        # CUSTOMER POINTS
        customer_points = []

        for i in range(len(x_set)):

            customer_points.append({

                'age':
                    float(x_set[i][0]),

                'estimated_salary':
                    float(x_set[i][1]),

                'actual_class':
                    int(y_set[i]),

                'point_color': (

                    'dodgerblue'

                    if y_set[i] == 1

                    else 'salmon'
                )
            })

        # SUPPORT VECTOR POINTS
        support_vector_points = []

        for i in range(
            len(support_vectors)
        ):

            support_vector_points.append({

                'age':
                    float(
                        support_vectors[i][0]
                    ),

                'estimated_salary':
                    float(
                        support_vectors[i][1]
                    ),

                'point_color':
                    'yellow',

                'point_type':
                    'support_vector'
            })

        # DECISION BOUNDARY
        decision_boundary = []

        prediction_matrix = predictions.reshape(
            x1.shape
        )

        rows, cols = prediction_matrix.shape

        for i in range(rows - 1):

            for j in range(cols - 1):

                current_class = (
                    prediction_matrix[i][j]
                )

                right_class = (
                    prediction_matrix[i][j + 1]
                )

                bottom_class = (
                    prediction_matrix[i + 1][j]
                )

                # CLASS CHANGED
                if (

                    current_class != right_class

                    or

                    current_class != bottom_class
                ):

                    decision_boundary.append({

                        'age':
                            float(x1[i][j]),

                        'estimated_salary':
                            float(x2[i][j]),

                        'boundary_type':
                            'maximum_margin_hyperplane'
                    })

        # RESPONSE
        response = {

            'success': True,

            'model':
                'Support Vector Machine',

            'kernel':
                'linear',

            'axis': {

                'x_axis':
                    'Age',

                'y_axis':
                    'Estimated Salary'
            },

            'plot_range': {

                'x_min':
                    float(x1.min()),

                'x_max':
                    float(x1.max()),

                'y_min':
                    float(x2.min()),

                'y_max':
                    float(x2.max())
            },

            'customer_points':
                customer_points,

            'prediction_regions':
                prediction_regions,

            'support_vectors':
                support_vector_points,

            'decision_boundary':
                decision_boundary,

            'legend': {

                'salmon':
                    'NOT BUY SUV',

                'dodgerblue':
                    'BUY SUV',

                'yellow':
                    'SUPPORT VECTOR',

                'decision_boundary':
                    'Maximum Margin Hyperplane'
            }
        }

        return jsonify(response)

    except Exception as e:

        return jsonify({

            'success': False,

            'error': str(e)

        }), 500

# RUN SERVER
if __name__ == '__main__':

    app.run(
        debug=True
    )