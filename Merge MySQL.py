import pymysql.cursors

connection = pymysql.connect(user='root', password='root',
                             database='GRE')

cursor = connection.cursor()

cursor.execute("CREATE TABLE IF NOT EXISTS GRES (No int, Sentence VARCHAR(1000), Label int);")
connection.commit()

commit = "select count(*) from GREA1"
cursor.execute(commit)
row = cursor.fetchall()[0][0]
No = 0
for i in range(row):
    cursor = connection.cursor()
    commit = "select Answer from GREA1 where No = %d;" % (i + 1)
    cursor.execute(commit)
    right = cursor.fetchall()[0][0]
    for a in ['A', 'B', 'C', 'D', 'E']:
        label = 0
        cursor = connection.cursor()
        commit = "select %s from GREA1 where No = %d;" % (a, (i + 1))
        cursor.execute(commit)
        answer = cursor.fetchall()[0][0]
        commit = "select Former from GREQ1 where No = %d;" % (i + 1)
        cursor.execute(commit)
        former = cursor.fetchall()[0][0]
        commit = "select Later from GREQ1 where No = %d;" % (i + 1)
        cursor.execute(commit)
        later = cursor.fetchall()[0][0]
        sentence = former + answer + later
        No += 1
        if a == right:
            label = 1
        with connection.cursor() as cursor:
            # Create a new record
            sql = "INSERT INTO GRES "
            sql += "(No, Sentence, Label) VALUES (%s, %s, %s)"
            cursor.execute(sql, (No, sentence, label))
        connection.commit()
        print No

cursor = connection.cursor()
commit = "select count(*) from GREA2"
cursor.execute(commit)
row = cursor.fetchall()[0][0]
for j in range(row):
    cursor = connection.cursor()
    commit = "select Answer%d from GREA2 where No = %d;" % (1, (j + 1))
    cursor.execute(commit)
    right1 = cursor.fetchall()[0][0]
    cursor = connection.cursor()
    commit = "select Answer%d from GREA2 where No = %d;" % (2, (j + 1))
    cursor.execute(commit)
    right2 = cursor.fetchall()[0][0]
    for a in ['A', 'B', 'C', 'D', 'E', 'F']:
        label = 0
        cursor = connection.cursor()
        commit = "select %s from GREA2 where No = %d;" % (a, (j + 1))
        cursor.execute(commit)
        answer = cursor.fetchall()[0][0]
        commit = "select Former from GREQ2 where No = %d;" % (j + 1)
        cursor.execute(commit)
        former = cursor.fetchall()[0][0]
        commit = "select Later from GREQ2 where No = %d;" % (j + 1)
        cursor.execute(commit)
        later = cursor.fetchall()[0][0]
        sentence = former + answer + later
        No += 1
        if a == right1 or a == right2:
            label = 1
        with connection.cursor() as cursor:
            # Create a new record
            sql = "INSERT INTO GRES "
            sql += "(No, Sentence, Label) VALUES (%s, %s, %s)"
            cursor.execute(sql, (No, sentence,label))
        connection.commit()
        print No

cursor = connection.cursor()
commit = "select count(*) from GREA3"
cursor.execute(commit)
row = cursor.fetchall()[0][0]
for i in range(row):
    cursor = connection.cursor()
    commit = "select Answer from GREA3 where No = %d;" % (i + 1)
    cursor.execute(commit)
    right = cursor.fetchall()[0][0]
    for a in ['A', 'B', 'C', 'D', 'E']:
        label = 0
        cursor = connection.cursor()
        commit = "select %s from GREA3 where No = %d;" % (a, (i + 1))
        cursor.execute(commit)
        answer = cursor.fetchall()[0][0]
        commit = "select Former from GREQ3 where No = %d;" % (i + 1)
        cursor.execute(commit)
        former = cursor.fetchall()[0][0]
        commit = "select Later from GREQ3 where No = %d;" % (i + 1)
        cursor.execute(commit)
        later = cursor.fetchall()[0][0]
        sentence = former + answer + later
        No += 1
        if a == right:
            label = 1
        with connection.cursor() as cursor:
            # Create a new record
            sql = "INSERT INTO GRES "
            sql += "(No, Sentence, Label) VALUES (%s, %s, %s)"
            cursor.execute(sql, (No, sentence, label))
        connection.commit()
        print No

cursor = connection.cursor()
commit = "select count(*) from GREA4"
cursor.execute(commit)
row = cursor.fetchall()[0][0]
for j in range(row):
    if not (j == 14 or j == 33):
        cursor = connection.cursor()
        commit = "select Answer%d from GREA4 where No = %d;" % (1, (j + 1))
        cursor.execute(commit)
        right1 = cursor.fetchall()[0][0]
        cursor = connection.cursor()
        commit = "select Answer%d from GREA4 where No = %d;" % (2, (j + 1))
        cursor.execute(commit)
        right2 = cursor.fetchall()[0][0]
        for a in ['A', 'B', 'C', 'D', 'E', 'F']:
            label = 0
            cursor = connection.cursor()
            commit = "select %s from GREA4 where No = %d;" % (a, (j + 1))
            cursor.execute(commit)
            answer = cursor.fetchall()[0][0]
            commit = "select Former from GREQ4 where No = %d;" % (j + 1)
            cursor.execute(commit)
            former = cursor.fetchall()[0][0]
            commit = "select Later from GREQ4 where No = %d;" % (j + 1)
            cursor.execute(commit)
            later = cursor.fetchall()[0][0]
            sentence = former + answer + later
            No += 1
            if a == right1 or a == right2:
                label = 1
            with connection.cursor() as cursor:
                # Create a new record
                sql = "INSERT INTO GRES "
                sql += "(No, Sentence, Label) VALUES (%s, %s, %s)"
                cursor.execute(sql, (No, sentence,label))
            connection.commit()
            print No
