from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load saved files
model = joblib.load("model.pkl")
encoders = joblib.load("encoders.pkl")
columns = joblib.load("columns.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:

        data = {
            "CODE_GENDER": request.form["CODE_GENDER"],
            "FLAG_OWN_CAR": request.form["FLAG_OWN_CAR"],
            "FLAG_OWN_REALTY": request.form["FLAG_OWN_REALTY"],
            "CNT_CHILDREN": int(request.form["CNT_CHILDREN"]),
            "AMT_INCOME_TOTAL": float(request.form["AMT_INCOME_TOTAL"]),
            "NAME_INCOME_TYPE": request.form["NAME_INCOME_TYPE"],
            "NAME_EDUCATION_TYPE": request.form["NAME_EDUCATION_TYPE"],
            "NAME_FAMILY_STATUS": request.form["NAME_FAMILY_STATUS"],
            "NAME_HOUSING_TYPE": request.form["NAME_HOUSING_TYPE"],
            "DAYS_BIRTH": int(request.form["DAYS_BIRTH"]),
            "DAYS_EMPLOYED": int(request.form["DAYS_EMPLOYED"]),
            "FLAG_MOBIL": int(request.form["FLAG_MOBIL"]),
            "FLAG_WORK_PHONE": int(request.form["FLAG_WORK_PHONE"]),
            "FLAG_PHONE": int(request.form["FLAG_PHONE"]),
            "FLAG_EMAIL": int(request.form["FLAG_EMAIL"]),
            "OCCUPATION_TYPE": request.form["OCCUPATION_TYPE"],
            "CNT_FAM_MEMBERS": float(request.form["CNT_FAM_MEMBERS"])
        }

        df = pd.DataFrame([data])

        # Encode categorical columns
        for col in encoders:
            value = str(df.loc[0, col])

            if value not in encoders[col].classes_:
                value = encoders[col].classes_[0]

            df[col] = encoders[col].transform([value])

        # Arrange columns
        df = df[columns]

        prediction = model.predict(df)[0]

        if prediction == 0:
            result = "Credit Card Approved"
        else:
            result = "Credit Card Rejected"

        return render_template("result.html", prediction=result)

    except Exception as e:
        return str(e)


if __name__ == "__main__":
    app.run(debug=True)