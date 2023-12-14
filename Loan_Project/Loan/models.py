from django.contrib.auth.models import AbstractUser
from django.db import models


class Employee(models.Model):
    first_name = models.CharField(max_length=20)
    last_name = models.CharField(max_length=20)
    dept = models.CharField(max_length=15)
    salary = models.IntegerField()
    bonus = models.IntegerField()
    role = models.IntegerField()
    phone = models.IntegerField()


class User(models.Model):
    name = models.CharField(max_length=20)
    userid = models.CharField(max_length=20, unique=True)
    email = models.CharField(max_length=30)
    password = models.CharField(max_length=20)


class loan(models.Model):
    Loan = models.AutoField(primary_key=True)
    Gender = models.CharField(max_length=30)
    Married = models.CharField(max_length=10)
    Dependents = models.IntegerField()
    Education = models.CharField(max_length=30)
    Self_Employed = models.CharField(max_length=30)
    ApplicantIncome = models.IntegerField()
    CoapplicantIncome = models.IntegerField()
    LoanAmount = models.IntegerField()
    Loan_Amount_Term = models.IntegerField()
    Credit_History = models.IntegerField()
    Property_Area = models.CharField(max_length=20)
    Loan_Status = models.CharField(max_length=10)
