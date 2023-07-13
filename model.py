import skl2onnx
from skl2onnx import convert_sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_model_pipeline(df, target, spark):
    print(df.count())
    # Delete null values
    df = df.replace('?', None) \
        .dropna(how='any')

    # Create the feature vector
    df = df.toPandas()
    y = df[target]
    X = df.drop([target], axis=1)

    # Define the preprocessing steps for the features
    preprocessor = StandardScaler()

    # Define the regressor model
    regressor = RandomForestRegressor()

    # Create the pipeline with feature preprocessing and the regressor
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', regressor)
    ])

    model = pipeline.fit(X, y)

    # Show the predictions
    predictions = model.predict(X)
    df['predictions'] = predictions
    df = spark.createDataFrame(df)
    df.show()

    mse = mean_squared_error(y, predictions)  # Calculate Mean Squared Error
    r2 = r2_score(y, predictions)  # Calculate R-squared

    print("Mean Squared Error:", mse)
    print("R-squared:", r2)

    initial_type = [('label', skl2onnx.common.data_types.FloatTensorType([None, 1])),
                    # Add other labels for multiple inputs
                    ]

    # Convert the model to onnx, target_opset is optional
    onnx_model = convert_sklearn(model, initial_types=initial_type, target_opset={'': 14, 'ai.onnx.ml': 2})

    # Serialize the model into a file
    with open("newmodelclassifier.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())
    exit()

