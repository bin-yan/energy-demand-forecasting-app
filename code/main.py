import os
from flask import Flask, render_template, jsonify, request, redirect, flash, url_for, make_response
from flask_pymongo import PyMongo
from werkzeug.utils import secure_filename
import pandas as pd
import json, bson, random, string
from datetime import datetime

import GPflow
import numpy as np
import tensorflow as tf

from sklearn.metrics import r2_score

from datetime import datetime
from dateutil import parser, tz


app = Flask(__name__, static_url_path='/static')
app.secret_key = 'some_secret'

app.config['MONGO_DBNAME'] = 'mydb'
mongo = PyMongo(app)

ALLOWED_EXTENSIONS = set(['csv'])

rand_str = lambda n: ''.join([random.choice(string.ascii_letters + string.digits) for i in range(n)])


@app.route('/static')
def run_index():
    return render_template("index.html")


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    # create a collection called "models" if not exist
    if 'models' not in mongo.db.collection_names():
        models = mongo.db['models']
        # TODO: createIndex on collection

    if request.method == 'POST':
        # If enter a collection/model name(id)
        if request.form['collection']:
            collection = request.form['collection']
            if collection not in mongo.db.collection_names():
                flash('Model does not exist.')
                return redirect(request.url)
            else:
                model = mongo.db['models'].find({'collection': collection}, {'_id': 0})[0]

                if 'target' in model.keys():
                    target = model['target']
                    return redirect(url_for('visualize', collection=collection, target=target))
                else:
                    return redirect(url_for('train_and_predict', collection=collection))

        else:
            # check if the post request has the file part
            if 'file' not in request.files:
                flash('No file part')
                return redirect(request.url)
            file = request.files['file']
            # if user does not select file, browser also
            # submit a empty part without filename
            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)
            if file:
                # import data to pandas and add some additional datetime features
                df = pd.read_csv(file)
                if 'date' not in df.columns:
                    flash('A date column with the name of "date" is required.')
                    return redirect(request.url)

                try:
                    dt = df['date'].apply(lambda x: parser.parse(x))
                except ValueError:
                    flash('Date format of one or more data values is not acceptable.')
                    return redirect(request.url)

                isNumeric = df.applymap(np.isreal).all(0)
                pass_numeric_test = True
                for column in df.columns:
                    if column != 'date':
                        if not isNumeric[column]:
                            pass_numeric_test = False
                            flash('Column "' + column + '" contains non-numeric value(s). '
                                                        'All values of features and target should be numeric.')

                if not pass_numeric_test:
                    return redirect(request.url)

                df['timestamp'] = dt.apply(lambda x: x.timestamp() * 1000)
                df['day_of_week'] = dt.apply(lambda x: x.isoweekday())   # 1 is Monday and 7 is Sunday
                df['day_of_month'] = dt.apply(lambda x: x.day)
                df['week_of_year'] = dt.apply(lambda x: int(x.strftime("%W")))
                df['day_of_year'] = dt.apply(lambda x: x.dayofyear)
                df['year'] = dt.apply(lambda x: x.year)
                df['month'] = dt.apply(lambda x: x.month)

                data_json = json.loads(df.to_json(orient='records'))
                # assign a random string as the name of the collection
                collection = rand_str(6)
                while collection in mongo.db.collection_names():
                    collection = rand_str(6)
                col = mongo.db[collection]
                col.insert(data_json)
                mongo.db['models'].insert({'collection': collection})

                return redirect(url_for('train_and_predict', collection=collection))

    return render_template("upload.html")


def get_keys(collection, train_only=False):

    col = mongo.db[collection]
    sample = col.find_one()
    keys = list(sample.keys())

    to_be_removed = ['_id', 'timestamp', 'date']
    if train_only:
        to_be_removed = to_be_removed + ['predictedMean', 'predictedStd', 'abnormal']

    for element in to_be_removed:
        if element in keys:
            keys.remove(element)

    return keys


def get_range(collection, field):

    result = {"min": None, "max": None}
    col = mongo.db[collection]
    result['min'] = col.find().sort(field, 1).limit(1)[0][field]
    result['max'] = col.find().sort(field, -1).limit(1)[0][field]

    return result


def get_data(collection, selected_fields=None, excluded_fields=None, sort_by = None):

    col = mongo.db[collection]
    fields = {'_id': 0}

    if selected_fields:
        for sf in selected_fields:
            fields[sf] = 1

    if excluded_fields:
        for ef in excluded_fields:
            fields[ef] = 0

    items = col.find({}, fields)

    if sort_by:
        items.sort([(sort_by, 1)])

    # keys = get_keys(collection)
    # data = []

    # for d in items:
    #     item = {}
    #     for k in keys:
    #         item[k] = d[k]
    #     data.append(item)

    data = list(items)  # This is fine for relatively small result sets, as you are pulling everything into memory.

    return data


@app.route('/get_model/<collection>')
def get_model(collection):

    model = mongo.db['models'].find({'collection': collection}, {'_id': 0})[0]

    return jsonify(model)


@app.route('/format/<type_>/<collection>')
def format_data(type_, collection):

    excluded_fields = request.args.getlist('excluded_fields[]')

    x = request.args.get('x')
    y = request.args.get('y')

    id_ = request.args.get('id_')
    normalColor = request.args.get('normalColor')
    abnormalColor = request.args.get('abnormalColor')
    showAbnormal = request.args.get('showAbnormal')

    measuredMean = request.args.get('measuredMean')
    predictedMean = request.args.get('predictedMean')
    predictedStd = request.args.get('predictedStd')

    if type_ == 'timeline':
        data = get_data(collection, excluded_fields=excluded_fields, sort_by='timestamp')
    else:
        data = get_data(collection, excluded_fields=excluded_fields)

    keys = list(data[0].keys())

    info = {}

    if type_ == 'scatter':
        formattedData = []
        for d in data:
            if x and d[x]:
                d["x"] = d[x]
                #print(d["x"])
            if y and d[y]:
                d["y"] = d[y]
            if id_ and d[id_]:
                d["id"] = d[id_]
            if 'abnormal' in keys:
                if d['abnormal'] and abnormalColor:
                    d['marker'] = {"fillColor": abnormalColor}
                if (not d['abnormal']) and normalColor:
                    d['marker'] = {"fillColor": normalColor}
            elif normalColor:
                d['marker'] = {"fillColor": normalColor}

            formattedData.append(d)

    elif type_ == 'timeline':
        formattedData = {"area": [], "measured": [], "predicted": []}
        marker = {}
        for d in data:
            if 'abnormal' in keys:
                if d['abnormal'] and abnormalColor:
                    marker = {"fillColor": abnormalColor}
                elif (not d['abnormal']) and normalColor:
                    marker = {"fillColor": normalColor}
            elif normalColor:
                marker = {"fillColor": normalColor}
            if predictedMean and predictedStd and d[predictedMean]:
                formattedData['area'].append({
                    "id": d[id_],
                    "x": d[x],
                    "low": d[predictedMean] - 2 * d[predictedStd],
                    "high": d[predictedMean] + 2 * d[predictedStd]
                })
                formattedData['predicted'].append({
                    "id": d[id_],
                    "x": d[x],
                    "y": d[predictedMean]
                })
            if measuredMean:

                d["id"] = d[id_] + '-measured'
                d["x"] = d[x]
                d["y"] = d[measuredMean]
                d["marker"] = marker

                formattedData['measured'].append(d)

    elif type_ == 'heatmap':
        formattedData = {}

        timestamp_range = get_range(collection, 'timestamp')

        info['year'] =  {"min": None, "max": None}
        info['year']['min'] = int(datetime.fromtimestamp(timestamp_range['min']/1000).year)
        info['year']['max'] = int(datetime.fromtimestamp(timestamp_range['max'] / 1000).year)

        for year in range(info['year']['min'], info['year']['max'] + 1, 1):
            formattedData[year] = {}
            for month in range(1, 13, 1):
                formattedData[year][month] = []
        for d in data:
            item = d

            item["id"] = d[id_]
            item["x"] = d["day_of_week"]
            item["y"] = d["week_of_year"]
            item["value"] = d[measuredMean]
            item["name"] = d['day_of_month']  # day of month, used as data labels

            if predictedMean and d[predictedMean]:
                item["predicted"] = d[predictedMean]
            if showAbnormal and 'abnormal' in keys and d['abnormal']:
                item["dataLabels"] = {
                    "borderColor": 'red',
                    "borderWidth": 2,
                    "shape": 'circle',
                    "style":
                        {
                            "color": "black",
                            "fontSize": "12px",
                            "fontWeight": "bold",
                            "textOutline": "1px 1px white"
                        }
                }
            formattedData[d['year']][d['month']].append(item)
        # since formattedData keys have int, it cannot have string keys. Otherwise jsonify will not work.

    results = {"keys": keys, "formattedData": formattedData, "info": info}

    return jsonify(results)


@app.route('/visualize/<collection>/<target>')
def visualize(collection, target):

    keys = get_keys(collection)

    return render_template("test.html", keys=keys, target=target, collection=collection)


def gp_train(X_train, Y_train):

    scale_x = np.amax(np.absolute(X_train), axis=0)
    scale_y = np.amax(np.absolute(Y_train))

    x_train = X_train / scale_x
    y_train = Y_train / scale_y

    n_features = x_train.shape[1]
    k = GPflow.kernels.RBF(n_features, lengthscales=1, ARD=True)
    m = GPflow.gpr.GPR(x_train, y_train, kern=k)

    message = m.optimize()

    trained_model = {"model": m, "scale_x": scale_x, "scale_y": scale_y, "message": message}

    return trained_model


def gp_predict(X_test, trained_model):

    x_test = X_test / trained_model["scale_x"]

    mean, var = trained_model["model"].predict_y(x_test)

    predicted_mean = mean * trained_model["scale_y"]
    predicted_std = np.sqrt(var) * trained_model["scale_y"]

    return predicted_mean, predicted_std

def gp_predict_nextDay(X_train, Y_train, X_test, Y_test, trained_model):

    x_train = X_train / trained_model["scale_x"]
    y_train = Y_train / trained_model["scale_y"]
    x_test = X_test / trained_model["scale_x"]
    y_test = Y_test / trained_model["scale_y"]

    fs = trained_model['model'].get_parameter_dict()
    predicted_mean_nextday = []
    predicted_var_nextday = []

    k = GPflow.kernels.RBF(len(x_train[0, :]), lengthscales=1, ARD=True)

    for i in range(len(y_test)):
        m2 = GPflow.gpr.GPR(x_train, y_train, kern=k)
        m2.set_parameter_dict(fs)
        y_mean, y_var = m2.predict_y([x_test[i]])
        x_train = np.append(x_train, [x_test[i]], axis=0)
        y_train = np.append(y_train, [y_test[i]], axis=0)
        if len(predicted_mean_nextday):
            predicted_mean_nextday = np.append(predicted_mean_nextday, y_mean, axis=0)
            predicted_var_nextday = np.append(predicted_var_nextday, y_var, axis=0)
        else:
            predicted_mean_nextday = y_mean
            predicted_var_nextday = y_var

    predicted_mean = predicted_mean_nextday * trained_model["scale_y"]
    predicted_std = np.sqrt(predicted_var_nextday) * trained_model["scale_y"]

    return predicted_mean, predicted_std


def insert_predictions(collection, df):

    col = mongo.db[collection]
    for index, row in df.iterrows():
        #print(row['predictedMean'])
        result = col.update_one({'timestamp': int(index)}, {
            '$set': {'predictedMean': row['predictedMean'], 'predictedStd': row['predictedStd'], 'abnormal': row['abnormal']}})
        # have to use int(index) because mongo db cannot recognize numpy.int64

    return None


def gp_get_feature_rank(trained_model, features, X_train):

    x_train = X_train/trained_model['scale_x']

    parameters = trained_model['model'].get_parameter_dict()
    print(parameters)
    lengthScale = parameters['model.kern.lengthscales']
    std = np.std(x_train, axis=0)
    importance = std / lengthScale
    #importance = 1 / lengthScale
    print(importance)

    index = np.argsort(importance)
    rank = []

    n = len(features)

    for i in range(n):
        item = {'name': features[index[i]], 'x': n-i, 'y': importance[index[i]]}
        rank.append(item)

    return rank


@app.route('/train_and_predict/<collection>', methods=['GET', 'POST'])
def train_and_predict(collection):

    keys = get_keys(collection, train_only=True)

    if request.method == 'POST':

        #results = {"trained": {}, "tested": {}}

        features = request.form.getlist('features')   # features must be a list, others are string/int
        target = request.form.get('target')

        forecasting_type = request.form.get('forecastingType')
        features_timeSeries = request.form.getlist('features_timeSeries')
        print(forecasting_type, features_timeSeries)

        trainStart = float(parser.parse(request.form.get("trainStart")).timestamp()*1000)
        trainEnd = float(parser.parse(request.form.get("trainEnd")).timestamp()*1000)

        testStart = float(parser.parse(request.form.get("testStart")).timestamp()*1000)
        testEnd = float(parser.parse(request.form.get("testEnd")).timestamp()*1000)

        data = get_data(collection, features + [target] + ['timestamp'])
        df = pd.DataFrame(data)
        df.sort_values(by=['timestamp'], inplace=True)
        df.set_index('timestamp', inplace=True)

        if forecasting_type == 'nextDay':
            if 'i' in features_timeSeries:
                df['i'] = 0
                for i in range(len(df)):
                    df.set_value(df.index[i], 'i', i)
            if 'y_t-1' in features_timeSeries:
                df['y_t-1'] = df[target].shift(1)
                df.set_value(df.index[0], 'y_t-1', df.loc[df.index[2], target])

        df_train = df[np.array(df.index >= trainStart) * np.array(df.index <= trainEnd)]
        df_test = df[np.array(df.index >= testStart) * np.array(df.index <= testEnd)]

        X_train = df_train[features+features_timeSeries].values
        Y_train = df_train[target].values[:, None]

        X_test = df_test[features+features_timeSeries].values
        Y_test = df_test[target].values[:, None]

        train_test_ratio = len(Y_test) / len(Y_train)

        trained_model = gp_train(X_train, Y_train)

        predicted_mean, predicted_std = gp_predict(X_train, trained_model)
        training_accuracy = r2_score(predicted_mean, Y_train)
        df_train['predictedMean'] = predicted_mean  # cannot use .loc[:,''] here. If first time, broadcast error.
        df_train['predictedStd'] = predicted_std
        df_train['abnormal'] = np.abs(df_train['predictedMean'] - df_train[target]) >= 1.96 * df_train['predictedStd']

        insert_predictions(collection, df_train)

        if forecasting_type == 'baseline':
            predicted_mean, predicted_std = gp_predict(X_test, trained_model)
        elif forecasting_type == 'nextDay':
            predicted_mean, predicted_std = gp_predict_nextDay(X_train, Y_train, X_test, Y_test, trained_model)

        test_accuracy = r2_score(predicted_mean, Y_test)
        df_test['predictedMean'] = predicted_mean
        df_test['predictedStd'] = predicted_std
        df_test['abnormal'] = np.abs(df_test['predictedMean'] - df_test[target]) >= 1.96 * df_test['predictedStd']
        insert_predictions(collection, df_test)

        feature_rank = gp_get_feature_rank(trained_model, features+features_timeSeries, X_train)
        print(feature_rank)

        mongo.db['models'].update_one({'collection': collection}, {
            '$set': {'features': features + features_timeSeries, 'target': target, 'trainStart': trainStart, 'trainEnd': trainEnd,
                     'testStart': testStart, 'testEnd': testEnd, 'trainAccuracy': training_accuracy,
                     'testAccuracy': test_accuracy, 'trainTestRatio': train_test_ratio, 'featureRank': feature_rank}})

        return redirect(url_for('visualize', target=target, collection=collection))
        # Here target and collection are parameters of visualize function, not template parameters

    return render_template("train.html", keys=keys, collection=collection)

@app.route('/download/<collection>/<target>')
def download(collection, target):

    data = get_data(collection, ['timestamp', 'date', 'predictedMean', 'predictedStd', target])
    df = pd.DataFrame(data)
    df.sort_values(by=['timestamp'], inplace=True)
    df.set_index('timestamp', inplace=True)

    resp = make_response(df.to_csv())
    resp.headers["Content-Disposition"] = "attachment; filename=export.csv"
    resp.headers["Content-Type"] = "text/csv"

    return resp


# with app.app_context():
#
#      col = mongo.db['FeFPNf']
#      # print(col.find().limit(1)[0])
#      # print(col.find().sort("timestamp", -1).limit(1)[0]['timestamp'])
#      # print(mongo.db['models'].find()[0])
#      print(get_data('FeFPNf', selected_fields=['y_t-1', 'day_of_month']))


if __name__ == '__main__':
    app.run(debug=True)


