import pickle, os
import random
import numpy as np
import pandas as pd
import spacy  
import tensorflow as tf
from flask_cors import CORS
from flask import Flask, jsonify, request, send_file

app = Flask(__name__)
CORS(app)

nlp = spacy.load(r'F:\New folder\choice-match-ai\ner-model')

feature_columns = [
    'Age',
    'Skin color',
    'FashionType',
    'Event',
    'Size',
    'Climate',
    'ColorMode',
    'Gender',
    'Color',
    'BudgetRange'
]

feature_store_path = "weights/FeatureStore.csv"
weight_path = 'weights/weights-fashion-rec.h5'
feature_path = 'weights/cloth-features.npz'
encoding_weight_dir = 'weights/encoding'


# Load the Model
model = tf.keras.models.load_model(weight_path)
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)
FeatureStore = pd.read_csv(feature_store_path)
FeatureStore = FeatureStore[feature_columns]
data = np.load(feature_path, allow_pickle=True)
FEATURES_BOTTOM = data['FEATURES_BOTTOM']
FEATURES_TOP = data['FEATURES_TOP']
BOTTOM_IDS = data['BOTTOM_IDS']
TOP_IDS = data['TOP_IDS']
# Read txt file
with open(os.path.join("femaleList.txt"), "r") as f:
    maleList = [line.strip().split(".")[0] for line in f.readlines()]

with open(os.path.join("maleList.txt"), "r") as f:
    femaleList = [line.strip().split(".")[0] for line in f.readlines()]


def MSE(img1, img2):
    diff = img1 - img2
    return (diff ** 2).sum()


def preprocess_Data(data_json):
    # Get empty fileds in json
    empty_fields = []
    for key, value in data_json.items():
        if value == '':
            empty_fields.append(key)

    df_inf = pd.DataFrame([data_json])

    # Known values
    age = ['A', 'B', 'C', 'D']
    skin_color = ['brown', 'dark', 'others', 'white']
    fashion_type = ['Simple cloth', 'Formal cloth', 'Professional cloth']
    event = ['General', 'Outdoor party', 'Indoor party', 'Wedding', 'Funeral']
    size = ['small', 'medium', 'Large', 'XL', '2XL']
    climate = ['Normal', 'Rainy', 'Sunny']
    color_mode = ['Dark', 'Light']
    gender = ["Male", "Female"]
    color = ['Black', 'Blue', 'Green', 'Others', 'Pink', 'Grey', 'Yellow', 'Red', 'White', 'Purple']
    budget_ranges = ['A', 'B', 'C']

    # Generate random values for empty fields
    for field in empty_fields:
        # Add random values for empty fields in df_inf
        if field == 'Age':
            df_inf['Age'] = np.random.choice(age)
        elif field == 'Skin color':
            df_inf['Skin color'] = np.random.choice(skin_color)
        elif field == 'FashionType':
            df_inf['FashionType'] = np.random.choice(fashion_type)
        elif field == 'Event':
            df_inf['Event'] = np.random.choice(event)
        elif field == 'Size':
            df_inf['Size'] = np.random.choice(size)
        elif field == 'Climate':
            df_inf['Climate'] = np.random.choice(climate)
        elif field == 'ColorMode':
            df_inf['ColorMode'] = np.random.choice(color_mode)
        elif field == 'Gender':
            df_inf['Gender'] = np.random.choice(gender)
        elif field == 'Color':
            df_inf['Color'] = np.random.choice(color)
        elif field == 'BudgetRange':
            df_inf['BudgetRange'] = np.random.choice(budget_ranges)

    Gender = df_inf['Gender'].values[0].strip()

    df_inf = df_inf[feature_columns]

    # Get vale for gender column from df_inf

    for feature in feature_columns:
        with open(os.path.join(encoding_weight_dir, feature + '.pkl'), 'rb') as f:
            encoder = pickle.load(f)
            print('feature: ', feature, 'encoder: ', encoder.classes_)
        df_inf[feature] = encoder.transform(df_inf[feature].str.strip().str.lower())

    nec_cols = ['Skin color', 'FashionType', 'Color', 'Event']

    match_idxs = None
    for col in nec_cols:
        m_idx = (FeatureStore[col] == df_inf[col].values[0])
        if match_idxs is None:
            match_idxs = m_idx
        else:
            match_idxs = np.logical_and(match_idxs, m_idx)
    match_idxs = np.where(match_idxs)[0]

    if len(match_idxs) == 0:
        diffF = FeatureStore - df_inf.loc[df_inf.index.repeat(len(FeatureStore))].reset_index(drop=True)
        max_sim = max((diffF == 0).sum(axis=1).values)
        match_idxs = ((diffF == 0).sum(axis=1) == max_sim).values
        match_idxs = np.where(match_idxs)[0]

    Xinf = df_inf.values.reshape(1, -1)
    Finf = model.predict(Xinf)
    Finf_bottom, Finf_top = Finf

    FEATURES_BOTTOM_FIL = FEATURES_BOTTOM[match_idxs]
    FEATURES_TOP_FIL = FEATURES_TOP[match_idxs]

    BOTTOM_IDS_FIL = BOTTOM_IDS[match_idxs]
    TOP_IDS_FIL = TOP_IDS[match_idxs]

    bottom_sims = np.zeros(len(match_idxs))
    top_sims = np.zeros(len(match_idxs))

    for i in range(len(match_idxs)):
        bottom_sims[i] = MSE(Finf_bottom.squeeze(), FEATURES_BOTTOM_FIL[i].squeeze()).squeeze()
        top_sims[i] = MSE(Finf_top.squeeze(), FEATURES_TOP_FIL[i].squeeze()).squeeze()

    full_sims = (bottom_sims + top_sims) / 2
    full_sims = np.argsort(full_sims)

    if len(full_sims) >= 5:
        full_sims = full_sims[:5]
    bottom_ids = BOTTOM_IDS_FIL[full_sims]
    top_ids = TOP_IDS_FIL[full_sims]

    print("Bottom ids initial: ", bottom_ids)

    # If gender is male exlude bottom ids that are in maleList
    if Gender == "Male":
        print("Gender is male")
        bottom_ids = [id for id in bottom_ids if id not in maleList]
        top_ids = [id for id in top_ids if id not in maleList]
    elif Gender == "Female":
        # Remove top ids that are in femaleList
        print("Gender is female")
        top_ids = [id for id in top_ids if id not in femaleList]
        bottom_ids = [id for id in bottom_ids if id not in femaleList]

    print("Bottom ids final: ", bottom_ids)

    response = []
    for idx, (bottom_id, top_id) in enumerate(zip(bottom_ids, top_ids)):
        response.append({
            "bottom_id": f"{bottom_id}",
            "top_id": f"{top_id}"
        })
    return response


@app.route('/predict', methods=['POST'])
def predict():
    values = []
    data_json = request.get_json()
    gender = 1 if data_json['Gender'] == 'Male' else 0
    for i in range(3):
        for e in preprocess_Data(data_json):
            values.append(e)

    unique_tuples = {tuple(item.items()) for item in values}
    resp = [dict(item) for item in unique_tuples]
    if len(resp) == 0:
        df = pd.read_csv(feature_store_path)
        df = df.groupby('Gender').get_group(gender)[['TopDress', 'BottomDress']]
        df.columns = ['top_id', 'bottom_id']
        tem = df.to_dict('records')
        for i in range(5):
            resp.append(random.choice(tem))

    return jsonify(resp)

@app.route('/extract-entities', methods=['POST'])
def extract_entities():
    data = request.get_json()
    print("Received data:", data)  
    text = data['text']
    gender = data.get('gender', 'Unspecified') 
    print("Processing text for gender:", gender) 
    
    doc = nlp(text)
    entities = [{'text': ent.text, 'label': ent.label_} for ent in doc.ents]
    print("Extracted entities:", entities)

    form_data = map_entities_to_form(entities)
    print("Mapped form data:", form_data)
    return jsonify({'entities': entities, 'form_data': form_data, 'gender': gender})

def map_entities_to_form(entities):
    form_data = {
        "Age": "",
        "Skin color": "",
        "FashionType": "",
        "Event": "",
        "Size": "",
        "Climate": "",
        "ColorMode": "",
        "Gender": "",
        "Color": "",
        "BudgetRange": ""
    }
    entity_conversion = {
        'SKIN COLOR': 'Skin color',
        'FASHIONTYPE': 'FashionType',
        'EVENT': 'Event',
         'AGE' : 'Age,',
        'SIZE' : 'Size',
        'CLIMATE':'Climate',
        'COLORMODE': 'ColorMode',
        'GENDER':'Gender',
        'COLOR':  'Color',
        'BUDGETRANGE':'BudgetRange'
    }

    for entity in entities:
        if entity['label'] in entity_conversion:
            form_key = entity_conversion[entity['label']]
            form_data[form_key] = entity['text']
    return form_data


@app.route('/images/<dress_type>/<filename>', methods=['GET'])
def serve_image(dress_type, filename):
    file_path = f'images/{dress_type}/{filename}'
    # print(f"Serving image from: {file_path}") 
    return send_file(file_path, mimetype='image/jpeg')


if __name__ == '__main__':
    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000,
        threaded=True
    )