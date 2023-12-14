import mysql.connector as connection
import pandas as pd

class Connection:

    def create_connection(self):
        try:
            mydb = connection.connect(host="localhost", database='loan', user="root", passwd="")
            query = "Select * from loandata;"
            data = pd.read_sql(query, mydb)
            mydb.close()  # close the connection
        except Exception as e:
            print(str(e))

        return data


