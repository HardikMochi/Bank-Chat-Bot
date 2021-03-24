from django.db import models

# Create your models here.
class Customers(models.Model):
    name = models.CharField(max_length=100)
    user_id = models.AutoField(primary_key=True)
    password = models.CharField(max_length=50)
    number_of_account = models.IntegerField(verbose_name="nuber_of_account")
    loan_status =models.CharField(max_length=15)

class Accounts(models.Model):
    user_id = models.ForeignKey(Customers, on_delete=models.CASCADE)
    account_type = models.CharField(max_length=100)
    avilable_balance = models.IntegerField()