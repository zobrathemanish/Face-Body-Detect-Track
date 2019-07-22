import mysql.connector

connection = mysql.connector.connect(host='localhost',
                                         database='users',
                                         user='phpmyadmin',
                                         password='phpmyadmin')
cursor = connection.cursor()

def insertPerson(age, gender, image_path):
    sql_insert_query = "INSERT INTO person(age,gender,image_path) VALUES ("+age+",'"+gender+"','"+image_path+"')"
    print(sql_insert_query)

    cursor.execute(sql_insert_query)
    connection.commit()
    print("Record inserted successfully into python_users table")

    return

def getPerson():
    return

