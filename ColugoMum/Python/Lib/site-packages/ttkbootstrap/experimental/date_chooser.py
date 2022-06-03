# check into this project for more info on implementation: https://pypi.org/project/tkcalendar/

import tkinter as tk
from tkinter import ttk
from ttkbootstrap import Style
import calendar

c = calendar.Calendar()
c.setfirstweekday(calendar.SUNDAY)

md = c.monthdayscalendar(2021, 4)
print(md)

root = tk.Tk()
style = Style()
style.theme_use('flatly')

root.columnconfigure(1, weight=1)

ttk.Button(root, text='<').grid(row=0, column=0, sticky='nsw')
ttk.Label(root, text='April 2021', anchor='center', style='primary.Inverse.TLabel').grid(row=0, column=1, sticky='nswe')
ttk.Button(root, text='>').grid(row=0, column=2, sticky='nse')

day_grid = ttk.Frame()
for i in range(7):
    day_grid.columnconfigure(i, weight=1)
for j in range(len(md)):
    day_grid.rowconfigure(j, weight=1)

for i, m in enumerate(['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']):
    ttk.Label(day_grid, text=m, style='secondary.Inverse.TLabel', padding=5, anchor='center').grid(row=0, column=i,sticky='nswe')

for w, week in enumerate(md):
    for d, day in enumerate(week):
        if d in [0, 6]:
            ttk.Label(day_grid, text=day if day != 0 else '', style='secondary.Inverse.TLabel', padding=5, anchor='center').grid(row=w+1, column=d, sticky='nswe')
        else:
            ttk.Label(day_grid, text=day if day != 0 else '', padding=5, anchor='center').grid(row=w+1, column=d, sticky='nswe')


day_grid.grid(row=1, column=0, columnspan=3, sticky='nswe')
day_grid.mainloop()





