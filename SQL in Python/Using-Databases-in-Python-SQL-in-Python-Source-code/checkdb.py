import sqlite3

conn = sqlite3.connect("contacts.sqlite")
update_sql = "UPDATE contacts SET email = 'update@update.com'"
update_cursor = conn.cursor()
update_cursor.execute(update_sql)
print("{} rows updated".format(update_cursor.rowcount))
for row in conn.execute("SELECT * FROM contacts"):
    print(row)

conn.close()
