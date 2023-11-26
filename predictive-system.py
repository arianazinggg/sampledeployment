import numpy as np
import pickle

# Load the saved model
loaded_model = pickle.load(open('trained_hybrid_model.sav', 'rb'))

input_data = (1.00, 25.15,28.36,20.88,1701.00,975.30,100.00,0.16,0.75,6.00,4.10,9.40,26.01,0.28,140.02, 70.89)

input_data_as_numpy_array = np.asarray(input_data)

input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)

print(f"Predicted output: {prediction}")