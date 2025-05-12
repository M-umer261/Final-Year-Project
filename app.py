from flask import Flask, render_template, request, jsonify, redirect, url_for, send_file
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
import os
import uuid
import io
import shutil
from datetime import datetime, timedelta

app = Flask(__name__)

# Configuration with automatic cleanup
UPLOAD_FOLDER = 'temp_uploads'
MODEL_FOLDER = 'saved_models'
RESULT_FOLDER = 'temp_results'
MAX_TEMP_FILE_AGE_HOURS = 24  # Files older than this will be deleted

# Create folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)


def cleanup_old_files():
    """Remove files older than MAX_TEMP_FILE_AGE_HOURS from temp folders"""
    now = datetime.now()
    cutoff = now - timedelta(hours=MAX_TEMP_FILE_AGE_HOURS)

    for folder in [UPLOAD_FOLDER, RESULT_FOLDER]:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                if file_time < cutoff:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")


# Run cleanup at startup
cleanup_old_files()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_id = f"{timestamp}_{uuid.uuid4().hex[:8]}"
        filepath = os.path.join(UPLOAD_FOLDER, f"{file_id}.csv")
        file.save(filepath)

        try:
            df = pd.read_csv(filepath)
            if df.empty:
                os.remove(filepath)  # Clean up empty file
                return jsonify({'error': 'The uploaded file is empty'}), 400

            columns = list(df.columns)
            return render_template('select_columns.html',
                                   columns=columns,
                                   file_id=file_id)
        except Exception as e:
            os.remove(filepath)  # Clean up invalid file
            return jsonify({'error': f'Error reading file: {str(e)}'}), 400


@app.route('/detect', methods=['POST'])
def detect_anomalies():
    file_id = request.form['file_id']
    selected_columns = request.form.getlist('columns')

    filepath = os.path.join(UPLOAD_FOLDER, f"{file_id}.csv")

    try:
        df = pd.read_csv(filepath)

        # Select only numeric columns from user selection
        numeric_cols = [col for col in selected_columns if pd.api.types.is_numeric_dtype(df[col])]

        if not numeric_cols:
            return jsonify({'error': 'No numeric columns selected for analysis'}), 400

        X = df[numeric_cols].values

        # Handle missing values
        if pd.isna(X).any():
            X = pd.DataFrame(X).fillna(pd.DataFrame(X).median()).values

        # Scale the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train model
        model = IsolationForest(contamination=0.05, random_state=42)
        model.fit(X_scaled)

        # Predict anomalies
        predictions = model.predict(X_scaled)
        anomaly_scores = model.decision_function(X_scaled)

        df['anomaly'] = ['Anomaly' if x == -1 else 'Normal' for x in predictions]
        df['anomaly_score'] = anomaly_scores

        # Save results with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        clean_path = os.path.join(RESULT_FOLDER, f"{timestamp}_clean.csv")
        anomalies_path = os.path.join(RESULT_FOLDER, f"{timestamp}_anomalies.csv")

        df[df['anomaly'] == 'Normal'].to_csv(clean_path, index=False)
        df[df['anomaly'] == 'Anomaly'].to_csv(anomalies_path, index=False)

        # Prepare results
        results = {
            'file_id': file_id,
            'timestamp': timestamp,
            'columns': numeric_cols,
            'all_columns': list(df.columns),
            'total_points': len(df),
            'anomalies': (df['anomaly'] == 'Anomaly').sum(),
            'normal': (df['anomaly'] == 'Normal').sum(),
            'data': df.to_dict(orient='records'),
            'sample_data': df.head(100).to_dict(orient='records'),
            'stats': df[numeric_cols].describe().to_dict()
        }

        # Clean up uploaded file after processing
        os.remove(filepath)

        return render_template('results.html', results=results)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/download/<timestamp>/<file_type>')
def download_file(timestamp, file_type):
    if file_type == 'clean':
        filename = f"{timestamp}_clean.csv"
        display_name = "anomaly_free_data.csv"
    elif file_type == 'anomalies':
        filename = f"{timestamp}_anomalies.csv"
        display_name = "anomalies_only.csv"
    else:
        return jsonify({'error': 'Invalid file type requested'}), 400

    filepath = os.path.join(RESULT_FOLDER, filename)

    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404

    return send_file(
        filepath,
        as_attachment=True,
        download_name=display_name
    )


# Scheduled cleanup route (can be called via PythonAnywhere tasks)
@app.route('/cleanup', methods=['POST'])
def cleanup_files():
    try:
        cleanup_old_files()
        return jsonify({'status': 'success', 'message': 'Cleanup completed'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)