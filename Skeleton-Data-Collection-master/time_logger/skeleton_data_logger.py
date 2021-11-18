import tkinter as tk
from tkinter import ttk
from csv import DictWriter
import os
import datetime
from tkinter import messagebox

from info import action_list


def get_current_time():
    currentDT = datetime.datetime.now()
    return currentDT


class SkeletonDataGui:
    def __init__(self):
        self.win = tk.Tk()
        self.win.title('Skeleton Data Collector')  # give a title name
        self.win.geometry("600x200")

        # create a csv file
        self.csv_name = str(datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")) + '_action_log.csv'
        save_dir = 'saved_csv'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        self.csv_name = os.path.join(save_dir, self.csv_name)

        # create labels
        # name label
        self.name_label = ttk.Label(self.win, text="Action ID: ")
        self.name_label.config(font=(None, 40))
        self.name_label.grid(row=0, column=0, sticky=tk.W)

        # Create entry box
        self.action_var = tk.StringVar()
        self.action_entrybox = ttk.Entry(self.win, width=16, textvariable=self.action_var)
        self.action_entrybox.grid(row=0, column=1)
        self.action_entrybox.focus()

        # Current action
        self.current_action_id = 0
        self.current_action = action_list[self.current_action_id]
        self.action_entrybox.insert(tk.END, self.current_action)

        # active
        self.active_label = ttk.Label(self.win, text="Active or Passive")
        self.active_label.grid(row=1, column=0, sticky=tk.W)

        self.active_var = tk.StringVar()
        self.active_button = ttk.Radiobutton(self.win, text='Active', value='Active', variable=self.active_var)
        self.active_button.grid(row=1, column=1)

        self.passive_button = ttk.Radiobutton(self.win, text='Passive', value='Passive', variable=self.active_var)
        self.passive_button.grid(row=1, column=2)

        # orientation
        self.orientation_label = ttk.Label(self.win, text="Forward or Backward")
        self.orientation_label.grid(row=2, column=0, sticky=tk.W)

        self.orientation_var = tk.StringVar()
        self.forward_button = ttk.Radiobutton(self.win, text='Forward', value='Forward', variable=self.orientation_var)
        self.forward_button.grid(row=2, column=1)

        self.backward_button = ttk.Radiobutton(self.win, text='Backward', value='Backward', variable=self.orientation_var)
        self.backward_button.grid(row=2, column=2)

        self.start_button = tk.Button(self.win, text="Start", command=self.start_action)
        self.start_button.grid(row=4, column=0)

        self.end_button = tk.Button(self.win, text="End", command=self.end_action)
        self.end_button.grid(row=4, column=2)

        # Switch action button
        self.prev_button = tk.Button(self.win, text="Previous Action", command=self.prev_action)
        self.prev_button.grid(row=5, column=0)

        self.next_button = tk.Button(self.win, text="Next Action", command=self.next_action)
        self.next_button.grid(row=5, column=2)

        self.already_start = False

        # Action ID

        self.win.mainloop()

    def save_action(self, a_start_time, a_end_time, a_last_time, a_orientation_var, a_active_var):
        action_name = self.action_var.get()
        if action_name == '' or action_name is None:
            messagebox.showerror("Error", "No action ID given. ")
            return
        if a_active_var == '' or a_orientation_var == '':
            messagebox.showerror("Error", "No forward or backward is given. ")
            return

        with open(self.csv_name, 'a', newline='') as f:
            self.dict_writer = DictWriter(f, fieldnames=['Action ID', 'Active', 'Orientation',
                                                    'Start Time', 'End Time', 'Last Time'
                                                    ])

            # dict_writer = DictWriter(f, fieldnames=['Action ID', 'Start Time', 'End Time', 'Last Time'
            #                                         ])
            if os.stat(self.csv_name).st_size == 0:  # if file is not empty than header write else not
                self.dict_writer.writeheader()

            self.dict_writer.writerow({
                'Action ID': action_name,
                'Active': a_active_var,
                'Orientation': a_orientation_var,
                'Start Time': a_start_time,
                'End Time': a_end_time,
                'Last Time': a_last_time,
            })
        # Change color after submit button
        messagebox.showinfo("Saved", "Saved successful")

    def get_current_time(self):
        currentDT = datetime.datetime.now()
        return currentDT

    def start_action(self):
        self.start_time = get_current_time()
        self.already_start = True

    def end_action(self):
        if not self.already_start:
            messagebox.showerror(title='Error', message='Press start first. ')
            return
        end_time = get_current_time()
        last_time = (end_time - self.start_time)
        print('last time: ', last_time)
        self.already_start = False

        orientation_var_ = self.orientation_var.get()

        active_var_ = self.active_var.get()

        self.save_action(self.start_time, end_time, last_time, orientation_var_, active_var_)

    def prev_action(self):
        # Go to next action
        if (self.current_action_id - 1) == -1:
            messagebox.showerror(title='Error', message='This is the first action. ')
            return
        self.action_entrybox.delete(0, tk.END)
        self.current_action_id = (self.current_action_id - 1)
        self.current_action = action_list[self.current_action_id]
        self.action_entrybox.insert(tk.END, self.current_action)

    def next_action(self):
        # Go to next action
        if (self.current_action_id + 1) == len(action_list):
            messagebox.showerror(title='Error', message='This is the last action. ')
            return
        self.action_entrybox.delete(0, tk.END)
        self.current_action_id = (self.current_action_id + 1)
        self.current_action = action_list[self.current_action_id]
        self.action_entrybox.insert(tk.END, self.current_action)


if __name__ == '__main__':
    omega = SkeletonDataGui()
