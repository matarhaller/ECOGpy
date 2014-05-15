## For the image positions thing
from Tkinter import *
from tkFileDialog import askopenfilename
import Image, ImageTk
import numpy as np

def getImagePositions(file_path):
    """
    Takes as input the path to a picture.  Displays the picture and lets you click on it.  Stores the xy coordinates of each click, so now you can superimpose something on top of it.

    Returns:
    xyCoords            : An np array that is nClicks x 2 specifying the points that you clicked on the picture
    xyDict              : A dictionary that is nClicks long specifying the points that you clicked on the picture
    """

    if __name__ == "__main__":
        root = Tk()

        #setting up a tkinter canvas with scrollbars
        frame = Frame(root, bd=2, relief=SUNKEN)
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)
        xscroll = Scrollbar(frame, orient=HORIZONTAL)
        xscroll.grid(row=1, column=0, sticky=E+W)
        yscroll = Scrollbar(frame)
        yscroll.grid(row=0, column=1, sticky=N+S)
        canvas = Canvas(frame, bd=0, xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
        canvas.grid(row=0, column=0, sticky=N+S+E+W)
        xscroll.config(command=canvas.xview)
        yscroll.config(command=canvas.yview)
        frame.pack(fill=BOTH,expand=1)

        #adding the image
        File ='../../electrodeLocation.PNG'
        img = ImageTk.PhotoImage(Image.open(File))
        canvas.create_image(0,0,image=img,anchor="nw")
        canvas.config(scrollregion=canvas.bbox(ALL))

        #function to be called when mouse is clicked
        xyCoords = []
        def printcoords(event):
            #outputting x and y coords to console
            print (event.x,event.y)
            xyCoords.append([event.x, event.y])

        #mouseclick event
        canvas.bind("<Button 1>",printcoords)

        root.mainloop()

    root = Tk()

    #setting up a tkinter canvas with scrollbars
    frame = Frame(root, bd=2, relief=SUNKEN)
    frame.grid_rowconfigure(0, weight=1)
    frame.grid_columnconfigure(0, weight=1)
    xscroll = Scrollbar(frame, orient=HORIZONTAL)
    xscroll.grid(row=1, column=0, sticky=E+W)
    yscroll = Scrollbar(frame)
    yscroll.grid(row=0, column=1, sticky=N+S)
    canvas = Canvas(frame, bd=0, xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
    canvas.grid(row=0, column=0, sticky=N+S+E+W)
    xscroll.config(command=canvas.xview)
    yscroll.config(command=canvas.yview)
    frame.pack(fill=BOTH,expand=1)

    #adding the image
    File = file_path
    img = ImageTk.PhotoImage(file=file_path, master=root)
    canvas.create_image(0,0,image=img,anchor="nw")
    canvas.config(scrollregion=canvas.bbox(ALL))

    #function to be called when mouse is clicked
    rawXyCoords = []
    def printcoords(event):
        #outputting x and y coords to console
        rawXyCoords.append([event.x, event.y])

    #mouseclick event
    canvas.bind("<Button 1>",printcoords)

    root.mainloop()

    xyCoords = np.array(rawXyCoords)
    xyDict = dict([[i,(xyCoords[i,0],xyCoords[i,1])] for i in np.arange(xyCoords.shape[0])]) ## in case the user wants coordinates to be in dictionary format.
    return xyDict
