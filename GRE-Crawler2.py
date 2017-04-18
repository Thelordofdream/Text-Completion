# coding=utf-8
import urllib2
import re
import time
import pymysql.cursors
import requests
from bs4 import BeautifulSoup

# XDF红皮 单选 230-275 等价 234-315
# Connect to the database
connection = pymysql.connect(user='root', password='root',
                             database='GRE')

cursor = connection.cursor()
commit = "CREATE TABLE IF NOT EXISTS GREQ3 (No int, Former VARCHAR(500), Later VARCHAR(500));"
cursor.execute(commit)
connection.commit()
commit = "CREATE TABLE IF NOT EXISTS GREA3 (No int, A VARCHAR(50), B VARCHAR(50), C VARCHAR(50), D VARCHAR(50), E VARCHAR(50), Answer VARCHAR(50));"
cursor.execute(commit)
connection.commit()

cursor = connection.cursor()
commit = "CREATE TABLE IF NOT EXISTS GREQ4 (No int, Former VARCHAR(500), Later VARCHAR(500));"
cursor.execute(commit)
connection.commit()
commit = "CREATE TABLE IF NOT EXISTS GREA4 (No int, A VARCHAR(50), B VARCHAR(50), C VARCHAR(50), D VARCHAR(50), E VARCHAR(50), F VARCHAR(50), Answer1 VARCHAR(50), Answer2 VARCHAR(50));"
cursor.execute(commit)
connection.commit()

ba = 0
No = 0
No1 = 0
for i in range(25):
    url = "http://gre.kmf.com/subject/lib?&s=10&t=0&p=5&p=%d" % (1 + i)
    content = urllib2.urlopen(url).read()
    bases = re.findall(r'<b>.*?<a href="(.*?)">.*?</a></b>', content)
    set = 0
    for j in range(len(bases)):
        set += 1
        url1 = "http://gre.kmf.com" + bases[j]
        response = requests.get(url1)
        soup = BeautifulSoup(response.text, "lxml")
        # print soup
        question = soup.find_all("div", class_="mb20")[1].find("div", class_="mb20").string
        print "Base：%d   Set：%d" % (i + 1, set)
        print question.replace("\n", "")
        Right = soup.find("div", class_="exa-correctanswer", id="ShowSiderAnswer").find("span").string
        if len(Right) == 1:
            No += 1
            if No >= 0:
                question = question.replace("    ", "").replace("\n", "")
                question = question.replace(u"\u2018", "'").replace(u"\u2019", "'").replace(u'\u2014',u'-').replace(
                    u"\u201c", "\"").replace(u"\u201d", "\"").replace(u"\uff0c", ",").replace(u'\u2013', '-')
                question = question.split("_____")
                former = question[0].replace(".", "").replace(",", "")
                later = question[1].replace(".", "").replace(",", "")
                options = soup.find_all("li", class_="clearfix")
                A = re.findall(r'<span><strong>.*?</strong>(.*?)</span>', str(options[0]))[0]
                B = re.findall(r'<span><strong>.*?</strong>(.*?)</span>', str(options[1]))[0]
                C = re.findall(r'<span><strong>.*?</strong>(.*?)</span>', str(options[2]))[0]
                D = re.findall(r'<span><strong>.*?</strong>(.*?)</span>', str(options[3]))[0]
                E = re.findall(r'<span><strong>.*?</strong>(.*?)</span>', str(options[4]))[0]
                print A, B, C, D, E
                print Right
                with connection.cursor() as cursor:
                    # Create a new record
                    sql = "INSERT INTO GREQ3 "
                    sql += "(No, Former, Later) VALUES (%s, %s, %s)"
                    cursor.execute(sql, (No, former, later))
                connection.commit()
                with connection.cursor() as cursor:
                    # Create a new record
                    sql = "INSERT INTO GREA3 "
                    sql += "(No, A, B, C, D, E, Answer) VALUES (%s, %s, %s, %s, %s, %s, %s)"
                    cursor.execute(sql, (No, A, B, C, D, E, Right))
                connection.commit()
        elif len(Right) == 3:
            question = question.replace("    ", "").replace("\n", "")
            question = question.replace(u"\u2018", "'").replace(u"\u2019", "'").replace(u'\u2014', u'-').replace(
                u"\u201c", "\"").replace(u"\u201d", "\"").replace(u"\uff0c", ",").replace(u'\u2013', '-')
            question = question.split("_____")
            if len(question) == 2:
                No1 += 1
                if No1 >= 0:
                    former = question[0].replace(".", "").replace(",", "")
                    later = question[1].replace(".", "").replace(",", "")
                    options = soup.find_all("li", class_="clearfix")
                    A = re.findall(r'<span><strong>.*?</strong>(.*?)</span>', str(options[0]))[0]
                    B = re.findall(r'<span><strong>.*?</strong>(.*?)</span>', str(options[1]))[0]
                    C = re.findall(r'<span><strong>.*?</strong>(.*?)</span>', str(options[2]))[0]
                    D = re.findall(r'<span><strong>.*?</strong>(.*?)</span>', str(options[3]))[0]
                    E = re.findall(r'<span><strong>.*?</strong>(.*?)</span>', str(options[4]))[0]
                    F = re.findall(r'<span><strong>.*?</strong>(.*?)</span>', str(options[5]))[0]
                    print A, B, C, D, E, F
                    print Right
                    Right1 = Right.split(" ")[0]
                    Right2 = Right.split(" ")[1]
                    with connection.cursor() as cursor:
                        # Create a new record
                        sql = "INSERT INTO GREQ4 "
                        sql += "(No, Former, Later) VALUES (%s, %s, %s)"
                        cursor.execute(sql, (No1, former, later))
                    connection.commit()
                    with connection.cursor() as cursor:
                        # Create a new record
                        sql = "INSERT INTO GREA4 "
                        sql += "(No, A, B, C, D, E, F, Answer1, Answer2) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
                        cursor.execute(sql, (No1, A, B, C, D, E, F, Right1, Right2))
                    connection.commit()
        time.sleep(0.05)
connection.close()