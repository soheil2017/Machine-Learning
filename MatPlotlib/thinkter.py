#https://datatofish.com/import-csv-file-python-using-pandas/
import tkinter as tk
from tkinter import filedialog
import pandas as pd
from matplotlib import pyplot as plt

root= tk.Tk()

canvas1 = tk.Canvas(root, width = 300, height = 300, bg = 'lightsteelblue2', relief = 'raised')
canvas1.pack()

def getCSV ():
    global df

    import_file_path = filedialog.askopenfilename()
    df = pd.read_csv(import_file_path)
    print(df)

browseButton_CSV = tk.Button(text="      Import your CSV File     ", command=getCSV, bg='green', fg='white',
                                 font=('helvetica', 12, 'bold'))
canvas1.create_window(150, 150, window=browseButton_CSV)

root.mainloop()


df = df.rename(columns={'field.point.x': 'X'})
df = df.rename(columns={'field.point.y': 'Y'})
df = df.rename(columns={'field.point.z': 'Z'})

plt.plot(df.X, df.Y, "g--", linewidth=2, markersize=1)

#plt.legend("easy_MH01")
plt.xlabel("X(meter)")
plt.ylabel("Y(meter)")
plt.title("easy_MH01")
plt.grid(True)
plt.show()


