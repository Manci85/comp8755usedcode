import tkinter as tk
from tkinter import ttk
from csv import DictWriter
import csv 
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
        self.win.geometry("800x500")

        # create a csv file
        self.csv_name = str(datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")) + '_action_log.csv'
        save_dir = 'saved_csv'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        self.csv_name = os.path.join(save_dir, self.csv_name)

        # create labels
        # name label
        self.name_label = ttk.Label(self.win, text="Action ID: ")
        self.name_label.config(font=(None, 20))
        # self.name_label.grid(row=0,  sticky=tk.W)
        self.name_label.grid(row=0, column=0 ,  sticky=tk.W)

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
        self.forward_button = ttk.Radiobutton(self.win, text='Forward', value='Forward', variable=self.orientation_var )
        self.forward_button.grid(row=2, column=1)

        self.backward_button = ttk.Radiobutton(self.win, text='Backward', value='Backward', variable=self.orientation_var)
        self.backward_button.grid(row=2, column=2)

        self.start_button = tk.Button(self.win, text="Start", command=self.start_action)
        self.start_button.grid(row=4, column=0)
        self.redo_button = tk.Button(self.win, text="Redo", command=self.redo , state=tk.DISABLED)
        self.redo_button.grid(row=4, column=1)
        self.end_button = tk.Button(self.win, text="Save & Next", command=self.end_action , state=tk.DISABLED)
        self.end_button.grid(row=4, column=2)

        # Switch action button
        # self.prev_button = tk.Button(self.win, text="Previous Action", command=self.prev_action , bg="yellow" , state=tk.ACTIVE)
        self.prev_button = tk.Button(self.win, text="Previous Action", command=self.prev_action)
        self.prev_button.grid(row=0, column=2)

        self.next_button = tk.Button(self.win, text="Next Action", command=self.next_action )
        self.next_button.grid(row=0, column=3)

        self.already_start = False

        self.counter = 0 
        self.record1 = "----------   Click start to record   ----------"
        self.record2 = "---------- record will be shown here ----------"
        self.record3 = "---------- record will be shown here ----------"

        self.list_box_label = ttk.Label(self.win, text="The latest three records... ")
        # self.list_box_label.config(font=(None, 16))

        self.list_box_label.grid(row=8, column=1 )
        
        self.listbox = tk.Listbox(self.win , height = 5)  
        self.listbox.insert(0,self.record1)  
        self.listbox.insert(1,self.record2)  
        self.listbox.insert(2,self.record3)  
        self.listbox.grid(row=9 , sticky=tk.EW  , columnspan = 4  )

        self.del_button = tk.Button(self.win, text="Delete selected item", command=self.del_selected_item )
        self.del_button.grid(row=10 , sticky=tk.EW  , columnspan = 2  )
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

    def update_listbox(self, a_start_time, a_end_time, a_last_time, a_orientation_var, a_active_var):
        action_name = self.action_var.get()
        cur_record = '{} | {} | {} | Endtime:{}'.format(action_name,a_orientation_var,a_active_var, str(a_end_time) )

        self.record1 = self.record2
        self.record2 = self.record3
        self.record3 = cur_record
        self.listbox.delete(0,tk.END)
        self.listbox.insert(0,self.record1)  
        self.listbox.insert(1,self.record2)  
        self.listbox.insert(2,self.record3)  

    def remove_line_from_csv(self , line):
        path = self.csv_name
        lines = list()
        # G40A1: belt person,Passive,Backward,2021-05-17 16:00:33.877364,2021-05-17 16:00:38.259727,0:00:04.382363
        # G2A2: bow | Backward | Passive | Endtime:2021-05-24 12:19:00.308288

        with open(path, 'r') as readFile:
            reader = csv.reader(readFile)
            for row in reader:
                lines.append(row)
                if [row[0],row[2],row[1]] ==  list([i.strip() for i in line.split('|')[:3] ])     :
                    lines.remove(row)
        with open(path, 'w' ,  newline='') as writeFile:
            writer = csv.writer(writeFile)
            writer.writerows(lines)
    
    def del_selected_item(self):
        # locate the select item # 
        for i in self.listbox.curselection():
            selected_text = self.listbox.get(i)
            index = i 
            break 
        
        if 'Deleted record' in selected_text or 'Endtime' not in selected_text:
            messagebox.showerror(title='Error', message='This record cannot be removed from the csv log file')
            return None 

        # delete the item from the log file # 
        self.remove_line_from_csv(selected_text)

        # update the listbox # 
        self.listbox.delete(index)
        if index == 0: 
            self.record1 = '[Deleted record] ' + self.record1
            self.listbox.insert(0,self.record1) 
        if index == 1: 
            self.record2 = '[Deleted record] ' + self.record2
            self.listbox.insert(1,self.record2) 
        if index == 2: 
            self.record3 = '[Deleted record] ' + self.record3
            self.listbox.insert(2,self.record3) 
        return None 


    def get_current_time(self):
        currentDT = datetime.datetime.now()
        return currentDT

    def start_action(self):
        orientation_var_ = self.orientation_var.get()
        active_var_ = self.active_var.get()
        if not orientation_var_: 
            messagebox.showerror(title='Error', message='Error. Orientation radio empty. Please select Forward/Backward')
            return
        if not active_var_: 
            messagebox.showerror(title='Error', message='Error. Active radio empty. Please select Active/Passive')
            return
        self.start_button['state'] = tk.DISABLED
        self.start_button['text'] = 'recording....'
        self.redo_button['state'] =  tk.NORMAL 
        self.end_button['state'] =  tk.NORMAL 
        self.start_time = get_current_time()
        self.already_start = True

    def redo(self):
        if not self.already_start:
            messagebox.showerror(title='Error', message='Press start first. ')
            return
        self.start_button['state'] = tk.NORMAL
        self.start_button['text'] = 'start'
        self.start_time = get_current_time()
        self.already_start = False

    def end_action(self):
        if not self.already_start:
            messagebox.showerror(title='Error', message='Press start first. ')
            return
        self.start_button['state'] = tk.NORMAL
        self.start_button['text'] = 'start'
        end_time = get_current_time()
        last_time = (end_time - self.start_time)
        print('last time: ', last_time)
        self.already_start = False
        orientation_var_ = self.orientation_var.get()
        active_var_ = self.active_var.get()
        self.save_action(self.start_time, end_time, last_time, orientation_var_, active_var_)

        ### auto select next active & orientation ###
        lst = [            
            ("Active" , "Forward"), 
            ("Active" , "Backward"), 
            ("Passive" , "Forward"), 
            ("Passive" , "Backward") ]

        print ((active_var_ , orientation_var_ ) )
        assert ( (active_var_ , orientation_var_ )  in lst  )
        next_active_var_ , next_orientation_var_ = lst[ ( lst.index((active_var_ , orientation_var_ ) )  + 1 )% 4 ]
        print  (next_active_var_ , next_orientation_var_)
        self.active_var.set(next_active_var_)
        self.orientation_var.set(next_orientation_var_)

        ### auto switch to next action if needed ### 
        if ( next_active_var_ , next_orientation_var_ ) ==  ("Active" , "Forward"):
            self.next_action()

        ### disable redo button ### 
        self.redo_button['state'] =  tk.DISABLED 
        self.end_button['state'] =  tk.DISABLED 

        self.update_listbox(self.start_time, end_time, last_time, orientation_var_, active_var_)
        self.counter += 1 



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


    def swtich_to_next_step(self):
        self.next_action() 


    def undo(self):
        pass 

if __name__ == '__main__':
    omega = SkeletonDataGui()
