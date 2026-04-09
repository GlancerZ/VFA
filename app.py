from flask import Flask, jsonify, render_template, request

from predictor import predict_task1_from_form, predict_task2_from_form

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['GET'])
def show_predict_page():
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        task_type = request.form.get('task_type', 'task1')
        form_data = request.form.to_dict()

        if task_type == 'task1':
            prediction = predict_task1_from_form(form_data)
            return render_template(
                'result_T1.html',
                task_type='task1',
                recommended_diet_name=prediction['recommended_diet_name'],
                vfa_reduction=prediction['vfa_reduction'],
                all_results=prediction['all_results'],
                prediction_mode=prediction['prediction_mode'],
                prediction_message=prediction['prediction_message'],
            )

        if task_type == 'task2':
            prediction = predict_task2_from_form(form_data)
            return render_template(
                'result_T2.html',
                task_type='task2',
                results=prediction['results'],
                prediction_mode=prediction['prediction_mode'],
                prediction_message=prediction['prediction_message'],
            )

        return jsonify({'error': 'Invalid task type'}), 400
    except ValueError as exc:
        return jsonify({'error': 'Invalid input: {0}'.format(exc)}), 400
    except Exception as exc:
        return jsonify({'error': 'Server error: {0}'.format(exc)}), 500

if __name__ == '__main__':
    app.run(debug=True)
