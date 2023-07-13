# Convert scikit-learn Pipelines to ONNX Models

This code demonstrates how to convert scikit-learn pipelines to ONNX models, allowing you to deploy your scikit-learn models in frameworks that support ONNX.

## Prerequisites

Make sure you have the following dependencies installed:

- scikit-learn
- skl2onnx
- onnxmltools

You can install these dependencies using pip:

```bash
pip install scikit-learn skl2onnx onnxmltools
```

# Usage

1. Clone the repository or download the code.
2. Import the necessary libraries in your Python environment:

   ```python
   import skl2onnx
   from skl2onnx import convert_sklearn
   from sklearn.ensemble import RandomForestRegressor
   from sklearn.metrics import mean_squared_error, r2_score
   from sklearn.pipeline import Pipeline
   from sklearn.preprocessing import StandardScaler
   ```
   
3. Define your scikit-learn pipeline, including any preprocessing steps and the final model:
    ```python
   preprocessor = ...  # Define your preprocessing steps
    regressor = ...  # Define your final model (regressor, classifier, etc.)
    pipeline = Pipeline([ ('preprocessor', preprocessor),
    ('regressor', regressor)])
   ```
    
4. Fit your pipeline on the training data
  
5. Convert the scikit-learn pipeline to ONNX format:
   ```python
   initial_type = [('input', onnx_types.FloatTensorType([None, num_features]))]  # Specify the initial type of the input data
   onnx_model = convert_sklearn(pipeline, initial_types=initial_type)
   ```
6. Save the ONNX model to a file:
     ```python
   onnxmltools.utils.save_model(onnx_model, 'model.onnx')
         with open("newmodelclassifier.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())
    exit()
   ```
     
## Contributing
Contributions to this project are welcome. If you have any suggestions, improvements, or bug fixes, please submit a pull request.
   
   
