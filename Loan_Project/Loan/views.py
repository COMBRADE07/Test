import pandas as pd
from django.shortcuts import render, HttpResponse, redirect
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .models import Employee, User, loan


# Create your views here.
def signup(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        userid = request.POST.get('userid')
        email = request.POST.get('email')
        password = request.POST.get('password')

        user = User(name=name, userid=userid, email=email, password=password)
        user.save()
        # if successful then return to loginpage
        return redirect("login/")
    elif request.method == 'GET':
        return render(request, 'signuppage.html')
    else:
        return HttpResponse("An Exception Occured! please try again")


def login(request):
    if request.method == 'POST':
        userid = request.POST.get('userid')
        password = request.POST.get('password')

        if userid and password:
            try:
                user = User.objects.get(userid=userid)
                if user.password == password:
                    # Credentials match, perform login
                    request.session['user_id'] = user.id
                    return render(request, 'index.html')
                else:
                    return HttpResponse("Invalid credentials! Please try again.")
            except User.DoesNotExist:
                return HttpResponse("User does not exist!")
        else:
            return HttpResponse("Invalid form data! Please fill in all required fields.")

    elif request.method == 'GET':
        return render(request, 'loginpage.html')

    return HttpResponse("An exception occurred! Please try again.")


def index(request):
    return render(request, 'index.html')



def predict(request):
    if request.method == 'POST':
        first_name = request.POST['first_name']
        last_name = request.POST['last_name']
        salary = int(request.POST['salary'])
        bonus = int(request.POST['bonus'])
        phone = int(request.POST['phone'])
        dept = request.POST['dept']
        role = int(request.POST['role'])
        new_emp = Employee(first_name=first_name, last_name=last_name, salary=salary, bonus=bonus, phone=phone,
                           dept=dept, role=role)
        new_emp.save()
        return HttpResponse('Employee added Successfully')
    elif request.method == 'GET':
        return render(request, 'predict.html')
    else:
        return HttpResponse("An Exception Occured! Employee Has Not Been Added")


def about(request):
    return render(request, 'about.html')


def about_ds(request):
    return render(request, 'about_dataset.html')


def loan_application(request):
    global prediction
    if request.method == 'POST':
        loan_id = request.POST.get('loan_id')
        gender = request.POST.get('gender')
        married = request.POST.get('married')
        dependents = request.POST.get('dependents')
        education = request.POST.get('education')
        self_employed = request.POST.get('self_employed')
        applicant_income = request.POST.get('applicant_income')
        coapplicant_income = request.POST.get('coapplicant_income')
        loan_amount = request.POST.get('loan_amount')
        loan_term = request.POST.get('loan_term')
        credit_history = request.POST.get('credit_history')
        property_area = request.POST.get('property_area')

        input_data = [loan_id, gender, married, dependents, education, self_employed, applicant_income,
                      coapplicant_income, loan_amount, loan_term, credit_history, property_area]

        # Perform further processing or database operations with the form data

        custom_input = {
            'Gender': gender,  # 1 represents 'Male'
            'Married': married,  # 0 represents 'No'
            'Dependents': dependents,
            'Education': education,  # 0 represents 'Graduate'
            'Self_Employed': self_employed,  # 0 represents 'No'
            'ApplicantIncome': applicant_income,
            'CoapplicantIncome': coapplicant_income,
            'LoanAmount': loan_amount,
            'Loan_Amount_Term': loan_term,
            'Credit_History': credit_history,
            'Property_Area': property_area  # 2 represents 'Urban'
        }

    return render(request, 'input.html')


def input_data(request):
    if request.method == 'POST':
        gender_mapping = {'Female': 0, 'Male': 1}
        married_mapping = {'No': 0, 'Yes': 1}
        education_mapping = {'Graduate': 0, 'Not Graduate': 1}
        self_employed_mapping = {'No': 0, 'Yes': 1}
        property_area_mapping = {'Rural': 0, 'Semiurban': 1, 'Urban': 2}
        loan_status_mapping = {'N': 0, 'Y': 1}

        gender = request.POST.get('gender')
        married = request.POST.get('married')
        dependents = request.POST.get('dependents')
        education = request.POST.get('education')
        self_employed = request.POST.get('self_employed')
        applicant_income = request.POST.get('applicant_income')
        coapplicant_income = request.POST.get('coapplicant_income')
        loan_amount = request.POST.get('loan_amount')
        loan_term = request.POST.get('loan_term')
        credit_history = request.POST.get('credit_history')
        property_area = request.POST.get('property_area')

        userdata = {
            'Gender': [gender_mapping.get(gender)],
            'Married': [married_mapping.get(married)],
            'Dependents': [dependents],
            'Education': [education_mapping.get(education)],
            'Self_Employed': [self_employed_mapping.get(self_employed)],
            'ApplicantIncome': [applicant_income],
            'CoapplicantIncome': [coapplicant_income],
            'LoanAmount': [loan_amount],
            'Loan_Amount_Term': [loan_term],
            'Credit_History': [credit_history],
            'Property_Area': [property_area_mapping.get(property_area)]
        }

        request.session['userdata'] = userdata
        return redirect('result')
    elif request.method == 'GET':
        return render(request, 'input.html')
    else:
        return HttpResponse("An Exception Occured! Employee Has Not Been Added")


# algorithm
def algorithm(data,custom_input):
    # Dropping Loan_ID column because there is no use for prediction
    data.drop(['Loan_ID'], axis=1, inplace=True)

    # encoding categorical columns into int
    obj = (data.dtypes == 'object')
    label_encoder = LabelEncoder()
    obj = (data.dtypes == 'object')
    for col in list(obj[obj].index):
        data[col] = label_encoder.fit_transform(data[col])

    # taking action on missing values in dataset
    for col in data.columns:
        data[col] = data[col].fillna(data[col].mean())

    # extract dependent and independent variable in dataset
    X = data.drop(['Loan_Status'], axis=1)
    Y = data['Loan_Status']

    # scaling is important to improve accuracy of model but,
    # in case of random forest algorithmRandom Forests are robust to,
    # variations in feature scales

    # spliting dataset into training and testing dataset
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

    # create instance of random forest model
    random_forest_classifier = RandomForestClassifier(n_estimators=7, criterion='entropy', random_state=7)

    # train model using random forest
    random_forest_classifier.fit(X, Y)

    # make prediction :)
    # Convert the custom input to a DataFrame
    custom_input_df = pd.DataFrame(custom_input,index=[0])


    result = random_forest_classifier.predict(custom_input_df)

    # accuracy of random forest
    # accuracy = accuracy_score(Y_test,result)*100
    # accuracy = random_forest_classifier.score(X_test,Y_test)*100    #0.9586776859504132

    return result

def fetch_data(request):
    userdata = request.session.get('userdata', {})
    # Retrieve data from the database
    queryset = loan.objects.all()

    # Create a list of dictionaries containing the data
    data = []
    for obj in queryset:
        data.append({
            'Loan_ID': obj.Loan,
            'Gender': obj.Gender,
            'Married': obj.Married,
            'Dependents': obj.Dependents,
            'Education': obj.Education,
            'Self_Employed': obj.Self_Employed,
            'ApplicantIncome': obj.ApplicantIncome,
            'CoapplicantIncome': obj.CoapplicantIncome,
            'LoanAmount': obj.LoanAmount,
            'Loan_Amount_Term': obj.Loan_Amount_Term,
            'Credit_History': obj.Credit_History,
            'Property_Area': obj.Property_Area,
            'Loan_Status': obj.Loan_Status,
        })
    column_list = ['Loan_ID', 'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome',
                   'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area',
                   'Loan_Status']

    # Create a DataFrame using the data
    data = pd.DataFrame(data,columns=column_list)

    from django.contrib.staticfiles import finders

    # Resolve the file path
    file_path = finders.find('dataset/LoanApprovalPrediction.csv')

    # Load the dataset
    data = pd.read_csv(file_path)
    result = algorithm(data,userdata)
    print(result)
    userdata['Loan_Status'] = result
    return render(request, 'result.html', {'userdata': userdata})
