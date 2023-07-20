import pickle 

# Load the serialized model from the file
with open('text_nb.pkl', 'rb') as file:
    loaded_model = pickle.load(file)
# predictions = loaded_model.predict(new_data)
